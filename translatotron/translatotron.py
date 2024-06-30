import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F
from translatotron.utils import play_audio


class SpeakerEncoder(nn.Module):
    def __init__(self, input_dim=80, output_dim=1024, hidden_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class StackedBLSTMEncoder(nn.Module):
    def __init__(self, input_dim=80, hidden_dim=1024, num_layers=8, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim // 2,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
        )

    def forward(self, x):
        return self.lstm(x)[0]


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.2):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

    def forward(self, query, key, value):
        return self.mha(query, key, value)[0]


class SpectrogramDecoder(nn.Module):
    def __init__(self, input_dim=1024, output_dim=80, hidden_dim=1024):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.decoder(x)


class LSTMDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out)


class AuxiliaryDecoder(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=256, output_dim=256, num_heads=1):
        super().__init__()
        self.attention = MultiheadAttention(input_dim, num_heads)
        self.lstm = LSTMDecoder(input_dim, hidden_dim, output_dim, num_layers=2)

    def forward(self, x):
        attention_out = self.attention(x, x, x)
        lstm_out = self.lstm(attention_out)
        return lstm_out


class Vocoder(nn.Module):
    def __init__(self, n_fft=2048, n_iter=60, hop_length=512):
        super().__init__()
        self.griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=n_fft,
            n_iter=n_iter,
            hop_length=hop_length,
            power=1.0,
            rand_init=False,
            momentum=0.99,
        )

    def forward(self, spectrogram):
        # Griffin-Lim expects a 3D tensor: (channels, freq, time)
        # Our spectrogram is (batch, time, freq), so we need to permute it
        spectrogram = spectrogram.permute(0, 2, 1)

        # Griffin-Lim works on batches, so we process each item in the batch
        waveforms = []
        for i in range(spectrogram.size(0)):
            waveform = self.griffin_lim(spectrogram[i])
            waveforms.append(waveform)

        # Stack the waveforms back into a batch
        return torch.stack(waveforms)


class Translatotron(nn.Module):
    def __init__(
        self,
        input_dim=80,
        encoder_hidden_dim=1024,
        encoder_layers=8,
        encoder_dropout=0.2,
        speaker_embedding_dim=1024,
        attention_heads=8,
        decoder_hidden_dim=1024,
        decoder_output_dim=1025,
        auxiliary_decoder_hidden_dim=256,
        auxiliary_decoder_output_dim=256,
        vocoder_n_fft=2048,
        vocoder_hop_length=512,
        vocoder_n_iter=60,
    ):
        super().__init__()

        self.stacked_blstm_encoder = StackedBLSTMEncoder(
            input_dim=input_dim,
            hidden_dim=encoder_hidden_dim,
            num_layers=encoder_layers,
            dropout=encoder_dropout,
        )

        self.speaker_encoder = SpeakerEncoder(
            input_dim=input_dim, output_dim=speaker_embedding_dim
        )

        combined_dim = encoder_hidden_dim + speaker_embedding_dim
        self.mha = MultiheadAttention(embed_dim=combined_dim, num_heads=attention_heads)

        self.spectrogram_decoder = SpectrogramDecoder(
            input_dim=combined_dim,
            output_dim=decoder_output_dim,
            hidden_dim=decoder_hidden_dim,
        )

        self.vocoder = Vocoder(
            n_fft=vocoder_n_fft, hop_length=vocoder_hop_length, n_iter=vocoder_n_iter
        )

        self.auxiliary_decoder_english = AuxiliaryDecoder(
            input_dim=combined_dim,
            hidden_dim=auxiliary_decoder_hidden_dim,
            output_dim=auxiliary_decoder_output_dim,
        )

        self.auxiliary_decoder_spanish = AuxiliaryDecoder(
            input_dim=combined_dim,
            hidden_dim=auxiliary_decoder_hidden_dim,
            output_dim=auxiliary_decoder_output_dim,
        )

    def forward(self, x, speaker_reference=None):
        encoder_output = self.stacked_blstm_encoder(x)

        if speaker_reference is not None:
            speaker_embedding = self.speaker_encoder(speaker_reference)
            speaker_embedding = speaker_embedding.unsqueeze(1).expand(
                -1, encoder_output.size(1), -1
            )
            encoder_output = torch.cat([encoder_output, speaker_embedding], dim=-1)
        else:
            batch_size, seq_len, _ = encoder_output.shape
            zero_padding = torch.zeros(
                batch_size,
                seq_len,
                self.speaker_encoder.encoder[-1].out_features,
                device=encoder_output.device,
            )
            encoder_output = torch.cat([encoder_output, zero_padding], dim=-1)

        mha_output = self.mha(encoder_output, encoder_output, encoder_output)

        spectrogram_out = self.spectrogram_decoder(mha_output)
        waveform = self.vocoder(spectrogram_out)

        phonemes_english = self.auxiliary_decoder_english(encoder_output)
        phonemes_spanish = self.auxiliary_decoder_spanish(encoder_output)

        return waveform, spectrogram_out, phonemes_english, phonemes_spanish
