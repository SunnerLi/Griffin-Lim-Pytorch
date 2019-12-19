# Gifflin-Lim-Pytorch 

[![Packagist](https://img.shields.io/badge/Pytorch-1.3.0-red.svg)]()
[![Packagist](https://img.shields.io/badge/Python-3.7.0-blue.svg)]()

Introduction
---
In this repository, we try to re-implement the Griffin-Lim algorithm with PyTorch. Since the official PyTorch doesn't support complex number, we use [this](https://github.com/pseeth/torch-stft) to deal with STFT and ISTFT. Check out the example below to see how to use it.

Install
---
Clone this project and move `pytorch_griffinlim` folder to the current folder. 
* Notice: You should install  [torch-stft](https://github.com/pseeth/torch-stft) if you want to run the example code.

Result
---
Compare to numpy version, the result is still slightly worse but MUCH MUCH faster. Feel free to open issue to make the program more better :-)

Example
---
```python
from pytorch_griffinlim.griffinlim import griffinlim
from torch_stft import STFT

stft = STFT(filter_length=1024, hop_length=256, win_length=1024)
magnitude, phase = stft.transform(audio)
wave = griffinlim(magnitude, 205000, n_iter=32, hop_length=256, win_length=1024)
```