import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import GriffinLim
from IPython.display import Audio


class BLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(BLSTM, self).__init__()
        self.layer = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, x):
        return self.layer(x)


class AuxiliaryDecoder(nn.Module):
    def __init__(
        self, in_size, hidden_size, num_layers, source_target_input_size, dropout_prob
    ):
        super(AuxiliaryDecoder, self).__init__()
        self.lstm = nn.LSTM(
            in_size,
            hidden_size,
            num_layers=num_layers,
            bidirectional=False,
            batch_first=True,
        )
        self.source_projection = nn.Linear(hidden_size, source_target_input_size)
        self.target_projection = nn.Linear(hidden_size, source_target_input_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x)
        source_output = self.source_projection(x)
        target_output = self.target_projection(x)
        return source_output, target_output


class Translatotron(nn.Module):
    def __init__(
        self,
        input_size=80,  # Assuming mel spectrogram input
        encoder_hidden_size=1024,
        encoder_num_layers=8,
        decoder_hidden_size=1024,
        decoder_num_layers=6,
        aux_decoder_hidden_size=256,
        aux_decoder_num_layers=2,
        aux_decoder_source_target_size=8,
        dropout_prob=0.2,
        n_fft=1024,
        win_length=1024,
        hop_length=256,
    ):
        super(Translatotron, self).__init__()
        self.encoder = BLSTM(input_size, encoder_hidden_size, encoder_num_layers)
        self.decoder = nn.LSTM(
            encoder_hidden_size * 2,
            decoder_hidden_size,
            num_layers=decoder_num_layers,
            bidirectional=False,
            batch_first=True,
        )
        self.aux_decoder = AuxiliaryDecoder(
            encoder_hidden_size * 2,
            aux_decoder_hidden_size,
            aux_decoder_num_layers,
            aux_decoder_source_target_size,
            dropout_prob,
        )

        self.output_projection = nn.Linear(decoder_hidden_size, n_fft // 2 + 1)

        self.griffin_lim = GriffinLim(
            n_fft=n_fft, win_length=win_length, hop_length=hop_length, power=1.0
        )

    def forward(self, x):
        encoder_output, _ = self.encoder(x)

        decoder_output, _ = self.decoder(encoder_output)
        spectrogram = self.output_projection(decoder_output)

        waveform = self.griffin_lim(spectrogram.transpose(1, 2))

        aux_source, aux_target = self.aux_decoder(encoder_output)

        return waveform, aux_source, aux_target


# Hyperparameters
input_sample_rate = 16000  # For Conversational model
output_sample_rate = 24000
learning_rate = 0.002
encoder_hidden_size = 1024
encoder_num_layers = 8
decoder_hidden_size = 1024
decoder_num_layers = 6
aux_decoder_hidden_size = 256
aux_decoder_num_layers = 2
aux_decoder_source_target_size = 8
dropout_prob = 0.2

# Create the model
model = Translatotron(
    input_size=80,  # Assuming 80-dim mel spectrogram input
    encoder_hidden_size=encoder_hidden_size,
    encoder_num_layers=encoder_num_layers,
    decoder_hidden_size=decoder_hidden_size,
    decoder_num_layers=decoder_num_layers,
    aux_decoder_hidden_size=aux_decoder_hidden_size,
    aux_decoder_num_layers=aux_decoder_num_layers,
    aux_decoder_source_target_size=aux_decoder_source_target_size,
    dropout_prob=dropout_prob,
)


x = torch.randn(32, 100, 80)  # (batch_size, sequence_length, input_features)
waveform, aux_source, aux_target = model(x)


print(waveform.shape)
