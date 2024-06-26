import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import GriffinLim
from IPython.display import Audio
import os


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

    @classmethod
    def from_pretrained(cls, model_path, device="cpu"):

        if not model_path.endswith(".pth"):
            raise ValueError("Model path should have a .pth extension")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No file found at {model_path}")

        # Load the state dict
        state_dict = torch.load(model_path, map_location=device)

        # Extract model parameters from the state dict
        model_params = state_dict.get("model_params", {})

        # Create a new model instance with the saved parameters
        model = cls(**model_params)

        # Load the model weights
        model.load_state_dict(state_dict["model_state_dict"])

        model.to(device)
        model.eval()  # Set the model to evaluation mode

        return model

    def save_pretrained(self, save_directory, model_name="translatotron_model"):

        os.makedirs(save_directory, exist_ok=True)
        save_path = os.path.join(save_directory, f"{model_name}.pth")

        # Prepare the state dict with both the model parameters and weights
        state_dict = {
            "model_params": {
                "input_size": self.encoder.layer.input_size,
                "encoder_hidden_size": self.encoder.layer.hidden_size,
                "encoder_num_layers": self.encoder.layer.num_layers,
                "decoder_hidden_size": self.decoder.hidden_size,
                "decoder_num_layers": self.decoder.num_layers,
                "aux_decoder_hidden_size": self.aux_decoder.lstm.hidden_size,
                "aux_decoder_num_layers": self.aux_decoder.lstm.num_layers,
                "aux_decoder_source_target_size": self.aux_decoder.source_projection.out_features,
                "dropout_prob": self.encoder.layer.dropout,
                "n_fft": self.griffin_lim.n_fft,
                "win_length": self.griffin_lim.win_length,
                "hop_length": self.griffin_lim.hop_length,
            },
            "model_state_dict": self.state_dict(),
        }
        # Save the state dict
        torch.save(state_dict, save_path)