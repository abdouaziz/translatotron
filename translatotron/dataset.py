import torch 
import torch.nn as nn 
import torchaudio 
from torch.utils.data import Dataset

import numpy as np 
from pathlib import Path
from typing import List , Union 

import librosa

from datasets import load_dataset , Audio
from log import get_logger , setup_logging

from tokenizer import Tokenizer
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

setup_logging()
logger = get_logger("Dataset")
    


class AudioException(Exception):
    pass 


class AudioProcessing:

    def load_wav(self,file_path:Union[str , Path]):
        return NotImplementedError(f"This method should be implemented")
    
    def amplitude_to_db(self, x , min_db=-100):
        clip_val = 10**(min_db/20)
        return 20*np.log10(torch.clamp(x , min=clip_val))
    
    def db_to_amplitude(self, x):
        return 10**(x/20)
    
    def normalize(self, x , min_db=-100 , max_abs_val=4):
        # Normalize x 
        # centered valours into [-max_abs_val , max_abs_val]

        x = (x-min_db)/-min_db
        x = 2* max_abs_val*x - max_abs_val
        x = torch.clip(x, min=-max_abs_val , max=max_abs_val)

    def denormlize(self ,x , min_db=-100 , max_abs_val=4):
        # centered values between -max_abs_val , max_abs_val
        # Denormalize to have the x 
        # Decentralized x 
        # return x
        x = torch.clip(x , min=-max_abs_val , max=max_abs_val)
        x = x * max_abs_val / 2*max_abs_val
        x = x*-min_db + min_db
        return x 
    


class AudioConversion(AudioProcessing):
    def __init__(self ,sampling_rate=22050,n_fft=1024,n_mels=80,fmin=0,
            fmax=8000 ,window_size=1024, hop_size=256, center=False , min_db=-100 , max_scaled_abs=4):
        super(AudioConversion,self).__init__()
        self.sampling_rate = sampling_rate
        self.n_fft=n_fft
        self.n_mels=n_mels
        self.fmin=fmin
        self.fmax=fmax
        self.hop_size= hop_size
        self.window_size = window_size
        self.center = center
        self.min_db =min_db
        self.max_scaled_abs=max_scaled_abs

        self.spec2mel = self._get_spec2mel_proj()
        self.mel2spec = torch.linalg.pinv(self.spec2mel)


    def load_wav(self, file_path):

        if not Path(file_path).exists:
            raise AudioException(f"The path of audio {file_path} doesnt exist")
        
        path_audio = Path(file_path)

        if not path_audio.suffix.lower() in [".mp3" ,".wav" ,".ogg" ,".flax"]:
            AudioException(f"Unsupported audio format: {path_audio.suffix}")

        try:
            audio , sr = torchaudio(path_audio)

            if sr != self.sampling_rate:
                audio = torchaudio.functional.resample(
                    waveform=audio,
                    orig_freq=sr,
                    new_freq=self.sampling_rate
                ) 

                audio = audio.squeeze(0)

                sr = self.sampling_rate

            return audio  , sr 
    
        except Exception as e :
            raise AudioException(f"Error of load file audio {e}")
        

    def _get_spec2mel_proj(self,):

        mel = librosa.filters.mel(
            sr=self.sampling_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )
        return torch.from_numpy(mel) 

    
    def audio2mel(self, audio , do_norm=False):

        if not isinstance(audio, torch.Tensor):
            audio = torch.tensor(audio, dtype=torch.float32)

        spectrogram = torch.stft(
                input=audio,
                n_fft=self.n_fft,
                hop_length=self.hop_size,
                win_length=self.window_size,
                window=torch.hann_window(self.window_size).to(audio.device),
                center=self.center,
                pad_mode="reflect",
                normalized=False,
                onesided=True,
                return_complex=True,
            )
        
        spectrogram = torch.abs(spectrogram)

        mel = torch.matmul(self.spec2mel.to(spectrogram.device) , spectrogram)

        mel = self.amplitude_to_db(mel , min_db=self.min_db)

        if do_norm:
            mel = self.normalize(mel , min_db=self.min_db , max_abs_val=self.max_scaled_abs)

        return mel 
    

    def mel2audio(self,mel , do_denorm=False , griffin_lim_iters=60):

        if do_denorm:

            mel = self.denormlize(mel , min_db=self.min_db , max_abs_val=self.max_scaled_abs)

        mel = self.db_to_amplitude(mel)

        spectogram = torch.matmul(self.mel2spec(mel.device) , mel).cpu().numpy()

        audio = librosa.griffinlim(
            S=spectogram,
            n_iter=griffin_lim_iters ,
            hop_length=self.hop_size,
            win_length=self.window_size ,
            n_fft=self.n_fft,
            window="hann")

        audio *= 32767 / max(0.01, np.max(np.abs(audio)))
        
        audio = audio.astype(np.int16)

        return audio
    

class TTSDataset(Dataset):
    def __init__(self,dataset_name_or_path ,
                 sampling_rate=22050,
                 n_fft=1024,n_mels=80,
                 fmin=0,fmax=8000,
                 window_size=1024,
                 hop_size=256,
                 center=False,
                 min_db=-100,
                 max_scaled_abs=4 , 
                 split="train+validation+test" ,
                 max_duration_in_seconds=30):
        
        self.sampling_rate=sampling_rate
        try:
            self.dataset = load_dataset(dataset_name_or_path ,split=split)
            logger.info(f"dataset {dataset_name_or_path} charged")

        except Exception as e :
            logger.warning(f"Impossible de load the dataset {dataset_name_or_path} from HF .")
            raise AudioException(f"Impossible to load the dataset {dataset_name_or_path} witht the split of {split}")

        if self.dataset[0]["audio"]["sampling_rate"] != sampling_rate:
            self.dataset = self.dataset.cast_column("audio" , Audio(sampling_rate=sampling_rate))

        #self.dataset = self.filter_dataset(max_duration_in_seconds=max_duration_in_seconds)

        self.tokenizer = Tokenizer(path_or_name=dataset_name_or_path , sampling_rate=sampling_rate ,split=split)

        self.audio_conversion= AudioConversion(
            sampling_rate=sampling_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            window_size=window_size,
            hop_size=hop_size,
            center=center,
            min_db=min_db,
            max_scaled_abs=max_scaled_abs
        )


    def filter_dataset(self,max_duration_in_seconds):
        logger.info(f"Filtering Dataset with max duration in second : {max_duration_in_seconds}")
        try:
            dataset = self.dataset.filter(lambda x: int(len(x["audio"]["array"])/self.sampling_rate)< max_duration_in_seconds)
            
            return dataset
        except Exception as e :
            raise AudioException(f"Error filtering dataset ")

    def __len__(self,):
        return len(self.dataset) 

    def __getitem__(self, idx):

        audio = self.dataset[idx]["audio"]["array"]

        transcription = self.dataset[idx]["transcription"]

        input_ids = self.tokenizer.encode(text=transcription)

        mel = self.audio_conversion.audio2mel(audio=audio)

        return {
            "transcription":transcription,
            "input_ids":input_ids.squeeze(0),
            "mel":mel.squeeze(0)
        }


def build_padding_mask(lengths):

    B = lengths.size(0)
    T = torch.max(lengths).item()

    mask = torch.zeros(B, T)
    for i in range(B):
        mask[i, lengths[i]:] = 1

    return mask.bool()


def TTSCollator():

    tokenizer = Tokenizer()

    def _collate_fn(batch):
        
        texts = [tokenizer.encode(b["transcription"]) for b in batch]
        mels = [b["mel"] for b in batch]
        
        ### Get Lengths of Texts and Mels ###
        input_lengths = torch.tensor([t.shape[0] for t in texts], dtype=torch.long)
        output_lengths = torch.tensor([m.shape[1] for m in mels], dtype=torch.long)

        ### Sort by Text Length (as we will be using packed tensors later) ###
        input_lengths, sorted_idx = input_lengths.sort(descending=True)
        texts = [texts[i] for i in sorted_idx]
        mels = [mels[i] for i in sorted_idx]
        output_lengths = output_lengths[sorted_idx]

        ### Pad Text ###
        text_padded = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=tokenizer.pad_token_id)

        ### Pad Mel Sequences ###
        max_target_len = max(output_lengths).item()
        num_mels = mels[0].shape[0]
        
        ### Get gate which tells when to stop decoding. 0 is keep decoding, 1 is stop ###
        mel_padded = torch.zeros((len(mels), num_mels, max_target_len))
        gate_padded = torch.zeros((len(mels), max_target_len))

        for i, mel in enumerate(mels):
            t = mel.shape[1]
            mel_padded[i, :, :t] = mel
            gate_padded[i, t-1:] = 1
        
        mel_padded = mel_padded.transpose(1,2)

        return text_padded, input_lengths, mel_padded, gate_padded, build_padding_mask(input_lengths), build_padding_mask(output_lengths)


    return _collate_fn


if __name__=="__main__":

    from torch.utils.data import DataLoader

    dataset = TTSDataset(
        dataset_name_or_path="abdouaziiz/alffa"
    )

    dataloader = DataLoader(dataset=dataset , batch_size=2  , collate_fn=TTSCollator())

    text_padded, input_lengths, mel_padded, gate_padded, encoder_mask, decoder_mask = next(iter(dataloader))

    print(mel_padded.shape)




