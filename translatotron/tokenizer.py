import torch
import pickle
import os
from datasets import load_dataset, Audio
from log import get_logger , setup_logging


setup_logging()
logger = get_logger("Tokenizer")
    

class Tokenizer:
    def __init__(self, path_or_name="abdouaziiz/alffa", sampling_rate=16000, split="train+validation+test", cache_dir="./cache"):
        self.path_or_name = path_or_name
        self.sampling_rate = sampling_rate
        self.split = split
        
        os.makedirs(cache_dir, exist_ok=True)
        
        cache_filename = f"{path_or_name.replace('/', '_')}_{split}_{sampling_rate}.pkl"
        self.cache_path = os.path.join(cache_dir, cache_filename)
        
        if self._load_from_cache():
            logger.info(f"Loaded from cache: {cache_filename}")
        else:
            logger.info("Building vocabulary from dataset...")
            self._build_vocabulary()
            self._save_to_cache()
            logger.info(f"Saved to cache: {cache_filename}")
    
    def _build_vocabulary(self):

        self.eos_token = "<EOS>"
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        
        ds = load_dataset(self.path_or_name, split=self.split)
        ds = ds.cast_column("audio", Audio(sampling_rate=self.sampling_rate))
        
        vocab_chars = set()
        for item in ds["transcription"]:
            vocab_chars.update(item.lower())
        
        self.chars = [self.pad_token, self.eos_token, self.unk_token] + sorted(list(vocab_chars))
        self.char2id = {char: i for i, char in enumerate(self.chars)}
        self.id2char = {i: char for i, char in enumerate(self.chars)}
        
        self.eos_token_id = self.char2id[self.eos_token]
        self.pad_token_id = self.char2id[self.pad_token]
        self.unk_token_id = self.char2id[self.unk_token]
        self.vocab_size = len(self.chars)

        logger.info(f"Tokenizer size : {self.vocab_size} of characters .")

    
    def _save_to_cache(self):

        cache_data = {
            'eos_token': self.eos_token,
            'pad_token': self.pad_token,
            'unk_token': self.unk_token,
            'chars': self.chars,
            'char2id': self.char2id,
            'id2char': self.id2char,
            'eos_token_id': self.eos_token_id,
            'pad_token_id': self.pad_token_id,
            'unk_token_id': self.unk_token_id,
            'vocab_size': self.vocab_size
        }
        
        with open(self.cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
    
    def _load_from_cache(self):

        if not os.path.exists(self.cache_path):
            return False
        
        
        
        try:
            with open(self.cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            self.eos_token = cache_data['eos_token']
            self.pad_token = cache_data['pad_token']
            self.unk_token = cache_data['unk_token']
            self.chars = cache_data['chars']
            self.char2id = cache_data['char2id']
            self.id2char = cache_data['id2char']
            self.eos_token_id = cache_data['eos_token_id']
            self.pad_token_id = cache_data['pad_token_id']
            self.unk_token_id = cache_data['unk_token_id']
            self.vocab_size = cache_data['vocab_size']

            logger.info(f"Tokenizer size : {self.vocab_size} of characters .")

            
            return True
        except:
            return False
    
    def encode(self, text, return_tensor=True):

        tokens = [self.char2id.get(char, self.unk_token_id) for char in text] + [self.eos_token_id]

        if return_tensor:
            tokens = torch.tensor(tokens, dtype=torch.long)
        
        return tokens
    

    def decode(self, token_ids, include_special_tokens=False):

        chars = []
        for token_id in token_ids:
            char = self.id2char.get(token_id.item(), self.unk_token)
            if include_special_tokens or char not in [self.eos_token, self.pad_token, self.unk_token]:
                chars.append(char)
        return ''.join(chars)
    
    def clear_cache(self):

        if os.path.exists(self.cache_path):
            os.remove(self.cache_path)
            logger.info(f"Cache cleared: {self.cache_path}")



# if __name__=="__main__":

#     tranform = Tokenizer(path_or_name="abdouaziiz/alffa")

#     text ="dafa ma ko wax may rabat lu ma waroona def"

#     ids = tranform.encode(text)
#     print(ids)       
#     print("je suis la ",tranform.decode(ids ,include_special_tokens=True))

