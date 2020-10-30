#!/usr/bin/env python

import pytest

import numpy as np
from py2shpss import STFT
from py2shpss import metric

np.random.seed(123)
@pytest.mark.parametrize("frame", [128, 256, 512, 1024])
@pytest.mark.parametrize("siglen", np.random.randint(16000, 48000, (10,)).tolist() + [6400, 16000])
def test_STFT_size(frame, siglen):
    # stft instance
    stft = STFT.STFT(frame)
    # create random signal
    sig = np.random.normal(0, 1, siglen)
    # stft
    amp, phase = stft.forward(sig)
    # check size
    T = -(-siglen // (frame // 2)) + 1
    F = frame // 2 + 1
    assert amp.shape == (F, T)

@pytest.mark.parametrize("frame", [128, 256, 512, 1024])
@pytest.mark.parametrize("siglen", np.random.randint(16000, 48000, (10,)).tolist() + [6400, 16000])
def test_STFT_reconstruction(frame, siglen):
    # stft instance
    stft = STFT.STFT(frame)
    # create random signal
    sig = np.random.normal(0, 1, siglen)
    # stft and istft
    amp, phase = stft.forward(sig)
    sig_ = stft.inverse(amp, phase)
    # check length
    assert len(sig_) >= siglen
    # evaluate sisdr
    sig_ = sig_[:siglen]
    sisdr = metric.SISDR(sig_, sig)
    assert sisdr > 50 # infty
