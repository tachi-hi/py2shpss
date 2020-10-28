#!/usr/bin/env python

import unittest

import numpy as np
from py2shpss import STFT
from py2shpss import samprate
from py2shpss import metric

np.random.seed(123)
class TestPy2shpss(unittest.TestCase):
    def test_STFT_size(self):
        frames = [128, 256, 512, 1024]
        siglens = np.random.randint(16000, 48000, (10,)).tolist() + [6400, 16000]
        for frame in frames:
            for siglen in siglens:
                # stft instance
                stft = STFT.STFT(frame)
                # create random signal
                sig = np.random.normal(0, 1, siglen)
                # stft
                amp, phase = stft.forward(sig)
                # check size
                T = -(-siglen // (frame // 2)) + 1
                F = frame // 2 + 1
                self.assertEqual(amp.shape, (F, T))

    def test_STFT_reconstruction(self):
        frames = [128, 256, 512, 1024]
        siglens = np.random.randint(16000, 48000, (10,)).tolist() + [6400, 16000]
        for frame in frames:
            for siglen in siglens:
                # stft instance
                stft = STFT.STFT(frame)
                # create random signal
                sig = np.random.normal(0, 1, siglen)
                # stft and istft
                amp, phase = stft.forward(sig)
                sig_ = stft.inverse(amp, phase)
                # check length
                self.assertTrue(len(sig_) >= siglen)
                # evaluate sisdr
                sig_ = sig_[:siglen]
                sisdr = metric.SISDR(sig_, sig)
                self.assertTrue(sisdr > 50) # infty

    def test_FFTsize(self):
        self.assertEqual(samprate.SampRate2FFTSize(16000), (512, 4096))
        self.assertEqual(samprate.SampRate2FFTSize(44100), (1024, 16384))

if __name__ == '__main__':
    unittest.main()