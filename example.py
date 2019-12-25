from pytorch_griffinlim.griffin_lim import griffinlim 
from torch_stft import STFT
import numpy as np
import librosa 
import torch

def main():
    # Define constant
    device = 'cuda'
    filter_length = 1024
    hop_length = 256
    win_length = 1024
    window = 'hann'
    n_iter = 50
    duration = 23

    # Load audio and form tensor
    audio, sr = librosa.load("your_music.mp3", duration=duration, offset=0)
    audio = torch.FloatTensor(audio)
    audio = audio.unsqueeze(0)
    audio = audio.to(device)

    # STFT (input size is [BATCH, N], N is the number of sample points)
    stft = STFT(filter_length=filter_length, hop_length=hop_length, win_length=win_length, window=window).to(device)
    magnitude, phase = stft.transform(audio)

    # Grifflin-lim to reconstruct the wave without phase
    wave = griffinlim(magnitude, sr*duration, n_iter=n_iter, angles=None, hop_length=hop_length, win_length=win_length, device=device)
    wave = wave.cpu().numpy()
    librosa.output.write_wav('out.wav', wave[0], sr, norm=True)

if __name__ == '__main__':
    main()