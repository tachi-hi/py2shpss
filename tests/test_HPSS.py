#!/usr/bin/env python

import pytest

import os
import numpy as np
import scipy.io.wavfile as wavfile
from py2shpss.STFT import STFT
from py2shpss.HPSS import HPSS
from py2shpss import metric

here = os.path.dirname(os.path.abspath(__file__))
wavpath = os.path.abspath(here + "/../sampleSounds/doremi.wav")
fft_sizes = [128, 384, 1024]
qs = [(0.001, 1), (0.1, 0.1), (100, 0.01), (1e3, 1e3), (1e-5, 1e-5)]
iter=30

@pytest.mark.parametrize("fft_size", fft_sizes)
def test_hm21_obj_decrease(fft_size):
    # load sig
    sr, sig = wavfile.read(wavpath)
    assert sr == 8000
    # stft
    amp, phase = STFT(fft_size).forward(sig)
    # hpss
    hpss = HPSS(mode='hm21', eval_obj=True, iter=iter)
    _, _, obj = hpss(amp)
    # check loss
    loss = [np.sum(_) for _ in obj]
    for x, y in zip(loss[:-1], loss[1:]):
        assert x >= y

@pytest.mark.parametrize("fft_size", fft_sizes)
@pytest.mark.parametrize("qH,qP", qs)
def test_idiv_obj_decrease(fft_size, qH, qP):
    # load sig
    sr, sig = wavfile.read(wavpath)
    assert sr == 8000
    # stft
    amp, phase = STFT(fft_size).forward(sig)
    # hpss
    hpss = HPSS(mode='idiv', eval_obj=True, qH = qH, qP = qP, iter=iter)
    _, _, obj = hpss(amp)
    # check loss
    loss = [h/qH + p/qP + idiv for h, p, idiv in obj]
    for x, y in zip(loss[:-1], loss[1:]):
        assert x >= y

