from .torch_stft import STFT
import numpy as np
import torch
import math

"""
    This script define the proceeding of Griffin-lim algorithm
    We refer torch_stft package to do the STFT and ISTFT in the procedure
    Package url: https://github.com/pseeth/torch-stft

    Author: SunnerLi
"""

def griffinlim(S, length, n_iter=32, hop_length=None, win_length=None, window='hann',
               center=True, momentum=0.99, init='random', device='cpu'):
    """
        The Pytorch implementation of Griffin-lim algorithm
        Since the Pytorch doesn't support complex number yet, we use two channel to represent complex number
        S[:, :, 0] is the magnitude of spectral, and S[:, :, 1] is the phase
        Ref: https://librosa.github.io/librosa/_modules/librosa/core/spectrum.html#griffinlim

        Arg:    S               (torch.Tensor)      - The spectral you want to reconstruct. Size is [BIN, TIME]
                length          (int)               - The number of sample in original waveform
                hop_length      (int)               - The hop length of the STFT
                win_length      (int)               - The window length of the STFT
                window          (str)               - The name of window function
                center          (bool)              - Use centered frame or left-aligned frames
                momentum        (float)             - The momentum parameter for fast Griffin-Lim
                init            (str)               - None or random. The method to initialize the phase
                device          (str)               - The device you want to use to compute. cpu | cuda
    """
    # Check if the momentum is valid
    if momentum > 1:
        print('Griffin-Lim with momentum={} > 1 can be unstable. Proceed with caution!'.format(momentum))
    elif momentum < 0:
        raise Exception('griffinlim() called with momentum={} < 0'.format(momentum))

    # using complex64 will keep the result to minimal necessary precision
    angles = torch.empty(S.size()).cuda()
    if init == 'random':
        # randomly initialize the phase
        angles[:] = torch.exp(2 * math.pi * torch.randn(*S.shape))
    elif init is None:
        # Initialize an all ones complex matrix
        angles[:] = 1.0
    else:
        raise Exception("init={} must either None or 'random'".format(init))

    # And initialize the previous iterate to 0
    rebuilt = 0.

    # Iterative update angles
    stft = STFT(filter_length=win_length, hop_length=hop_length, win_length=win_length, window=window, length=length).to(device)
    S = S.cuda()
    for _ in range(n_iter):
        # Store the previous iterate
        tprev = rebuilt

        # Invert with our current estimate of the phases
        inverse = stft.inverse(S, angles)

        # Rebuild the spectrogram
        _, rebuilt = stft.transform(inverse)

        # Update our phase estimates
        angles[:] = rebuilt - (momentum / (1 + momentum)) * tprev
        angles[:] /= torch.abs(angles) + 1e-16

    # Return the final phase estimates
    return stft.inverse(S, angles)