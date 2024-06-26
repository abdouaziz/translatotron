import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from torchaudio.transforms import GriffinLim
from IPython.display import Audio

class BLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=8):
        super(BLSTM, self).__init__()
        self.layer = nn.LSTM(
            input_size, hidden_size, num_layers=num_layers, bidirectional=True
        )

    def forward(self, x):
        return self.layer(x)


class Attention(nn.Module):
    def __init__(self, feature_size):
        super(Attention, self).__init__()
        self.feature_size = feature_size

        # Linear transformations for Q, K, V from the same source
        self.key = nn.Linear(feature_size, feature_size)
        self.query = nn.Linear(feature_size, feature_size)
        self.value = nn.Linear(feature_size, feature_size)

    def forward(self, x, mask=None):
        # Apply linear transformations
        keys = self.key(x)
        queries = self.query(x)
        values = self.value(x)

        # Scaled dot-product attention
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.feature_size, dtype=torch.float32)
        )

        # Apply mask (if provided)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)

        # Multiply weights with values
        output = torch.matmul(attention_weights, values)

        return output, attention_weights


class SpeakerEncoder(nn.Module):
    def __init__(self, input_size, out_size):
        super(SpeakerEncoder, self).__init__()
        self.layer = nn.Linear(input_size, out_size)
        self.norm = nn.LayerNorm(out_size)

    def forward(self, x):
        x = self.layer(x)
        x = self.norm(x)
        return x


class AuxillaryDecoder(nn.Module):
    def __init__(self, in_size, out_size, n_layers=2):
        super(AuxillaryDecoder, self).__init__()
        self.attention = Attention(in_size)
        self.layers = nn.ModuleList(
            [
                nn.LSTM(in_size, out_size, num_layers=n_layers, bidirectional=False)
                for _ in range(2)
            ]
        )

    def forward(self, x):
        x, _ = self.attention(x)

        for layer in self.layers:
            x, _ = layer(x)

        return x, x


class DecoderWaveforme(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(DecoderWaveforme, self).__init__()
        self.multihead = MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)
        self.lstm = nn.LSTM(embed_dim, embed_dim, bidirectional=False)

    def forward(self, x, speaker_embedding):
        x = torch.cat([x, speaker_embedding], dim=-1)

        x, _ = self.multihead(x, x, x)

        x = self.norm(x)
        x, _ = self.lstm(x)
        return x


class Translatotron(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=8, feature_size=10 , 
                n_fft=512, win_length=400, hop_length=160, power=2):
        super(Translatotron, self).__init__()
        self.encoder = BLSTM(input_size, hidden_size, num_layers)
        self.speaker_encoder = SpeakerEncoder(hidden_size * 2, feature_size * 2)
        self.aux_decoder = AuxillaryDecoder(hidden_size * 2, feature_size * 2)
        self.decoder_waveform = DecoderWaveforme(feature_size * 4, 8)

        self.pre_grif = nn.Linear(40, 257)

        self.griffin_lim = GriffinLim(
            n_fft=n_fft, win_length=win_length, hop_length=hop_length, power=power
        )

    def forward(self, x, speaker_embedding=None):

        x, _ = self.encoder(x)

        phonem_A, phonem_B = self.aux_decoder(x)

        if speaker_embedding is None:
            speaker_embedding = torch.zeros_like(x)

        speaker_embedding = self.speaker_encoder(speaker_embedding)

        x = self.decoder_waveform(x, speaker_embedding)

        x = self.pre_grif(x)

        wave = self.griffin_lim(x.transpose(1, 2))

        return wave, phonem_A, phonem_B


if __name__ == "__main__":
    model = Translatotron(input_size=40, hidden_size=10)
    x = torch.randn(10, 40, 40)
    wave, phonem_A, phonem_B = model(x)
    print(wave)
 