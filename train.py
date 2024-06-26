import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchaudio import transforms as T
import numpy as np
from translatotron import Translatotron
from translatotron import SpeechToSpeechDataset


# Hyperparameters
batch_size = 32
num_epochs = 100
learning_rate = 0.002
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

 
model = Translatotron().to(device)

 
mse_loss = nn.MSELoss()
ce_loss = nn.CrossEntropyLoss()

 
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
 
train_dataset = SpeechToSpeechDataset(split='train')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Mel spectrogram converter
mel_converter = T.MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=256, n_mels=80).to(device)

def train_one_epoch(model, loader, optimizer, epoch):
    model.train()
    total_loss = 0

    for batch_idx, (source_audio, target_audio, source_text, target_text) in enumerate(loader):
        source_audio, target_audio = source_audio.to(device), target_audio.to(device)
        source_text, target_text = source_text.to(device), target_text.to(device)

      
        source_mel = mel_converter(source_audio)
        target_mel = mel_converter(target_audio)

        optimizer.zero_grad()

        # Forward pass
        output_waveform, aux_source, aux_target = model(source_mel)

    
        waveform_loss = mse_loss(output_waveform, target_audio)
        source_text_loss = ce_loss(aux_source.view(-1, aux_source.size(-1)), source_text.view(-1))
        target_text_loss = ce_loss(aux_target.view(-1, aux_target.size(-1)), target_text.view(-1))

        #  losses
        total_loss = waveform_loss + source_text_loss + target_text_loss

 
        total_loss.backward()
        optimizer.step()

 
        if batch_idx % 10 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}/{len(loader)}, '
                  f'Loss: {total_loss.item():.4f}')

    return total_loss / len(loader)

 
for epoch in range(num_epochs):
    avg_loss = train_one_epoch(model, train_loader, optimizer, epoch)
    
 
    scheduler.step(avg_loss)
    
    print(f'Epoch {epoch} completed. Average Loss: {avg_loss:.4f}')

 
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f'translatotron_model_epoch_{epoch+1}.pth')

print("Training completed!")