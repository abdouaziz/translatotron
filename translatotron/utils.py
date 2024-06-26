import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import numpy as np
import os
import IPython.display as ipd
import soundfile as sf
 

class SpeechToSpeechDataset(Dataset):
    def __init__(self, root_dir, csv_file, split='train', source_lang='en', target_lang='fr', max_audio_length=10):
        """
        Args:
            root_dir (string): Directory with all the audio files.
            csv_file (string): Path to the csv file with annotations.
            split (string): 'train', 'val', or 'test'
            source_lang (string): Source language code
            target_lang (string): Target language code
            max_audio_length (int): Maximum audio length in seconds
        """
        self.root_dir = root_dir
        self.data = pd.read_csv(csv_file)
        self.data = self.data[self.data['split'] == split]
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.max_audio_length = max_audio_length
        
        # Create dictionaries for text-to-index conversion
        self.source_vocab = self.create_vocabulary(self.data[f'{source_lang}_text'])
        self.target_vocab = self.create_vocabulary(self.data[f'{target_lang}_text'])

    def create_vocabulary(self, text_series):
        vocab = set()
        for text in text_series:
            vocab.update(text)
        return {char: idx for idx, char in enumerate(sorted(vocab))}

    def text_to_index(self, text, vocab):
        return torch.tensor([vocab[char] for char in text if char in vocab])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load audio files
        source_audio_path = os.path.join(self.root_dir, self.data.iloc[idx][f'{self.source_lang}_audio'])
        target_audio_path = os.path.join(self.root_dir, self.data.iloc[idx][f'{self.target_lang}_audio'])

        source_audio, source_sr = torchaudio.load(source_audio_path)
        target_audio, target_sr = torchaudio.load(target_audio_path)

        # Ensure mono audio
        source_audio = source_audio.mean(dim=0) if source_audio.shape[0] > 1 else source_audio.squeeze(0)
        target_audio = target_audio.mean(dim=0) if target_audio.shape[0] > 1 else target_audio.squeeze(0)

        # Trim or pad audio to max_audio_length
        max_length = int(self.max_audio_length * source_sr)
        if source_audio.shape[0] > max_length:
            source_audio = source_audio[:max_length]
        else:
            source_audio = torch.nn.functional.pad(source_audio, (0, max_length - source_audio.shape[0]))

        if target_audio.shape[0] > max_length:
            target_audio = target_audio[:max_length]
        else:
            target_audio = torch.nn.functional.pad(target_audio, (0, max_length - target_audio.shape[0]))

        # Get text data
        source_text = self.data.iloc[idx][f'{self.source_lang}_text']
        target_text = self.data.iloc[idx][f'{self.target_lang}_text']

        # Convert text to indices
        source_text_indices = self.text_to_index(source_text, self.source_vocab)
        target_text_indices = self.text_to_index(target_text, self.target_vocab)

        return source_audio, target_audio, source_text_indices, target_text_indices
    
    
    
def play_audio(waveform, sample_rate=24000, filename=None):
    """
    Play or save the audio waveform.
    
    Args:
    waveform (torch.Tensor or np.ndarray): The audio waveform to play/save.
    sample_rate (int): The sample rate of the audio (default: 24000).
    filename (str, optional): If provided, save the audio to this file instead of playing it.
    
    Returns:
    IPython.display.Audio or None: Audio widget if in a notebook environment, None otherwise.
    """
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.detach().cpu().numpy()
    
    if waveform.ndim == 2:
        waveform = waveform.squeeze(0)  # Remove batch dimension if present
    
    if filename:
        sf.write(filename, waveform, sample_rate)
        print(f"Audio saved to {filename}")
        return None
    
    try:
 
        return ipd.Audio(waveform, rate=sample_rate, autoplay=False)
    except:
      
        temp_file = "temp_audio.wav"
        sf.write(temp_file, waveform, sample_rate)
        print(f"Audio saved to temporary file: {os.path.abspath(temp_file)}")
        print("You can play this file using your system's audio player.")
        return None
