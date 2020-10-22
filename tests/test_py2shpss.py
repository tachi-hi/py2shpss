#!/usr/bin/env python

import unittest

import numpy as np
from py2shpss import py2shpss

np.random.seed(123)
class TestPy2shpss(unittest.TestCase):
    def test_SISDR(self):
        x = np.random.normal(0, 1, 16000)
        y = np.random.normal(0, 1, 16000)
        for SDR in [-10, -5, 0, 5, 10]:
            ratio = 10 ** (SDR / 20)
            mix = y + ratio * x
            SISDR = py2shpss.SISDR(mix, x)
            print(SDR, SISDR)
            self.assertTrue(np.abs(SISDR - SDR) < 1)

    def test_STFT_size(self):
        frames = [128, 256, 512, 1024]
        siglens = np.random.randint(16000, 48000, (10,)).tolist() + [6400, 16000]
        for frame in frames:
            for siglen in siglens:
                # stft instance
                stft = py2shpss.STFT(frame)
                # create random signal
                sig = np.random.normal(0, 1, siglen)
                # stft
                amp, phase = stft.STFT(sig)
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
                stft = py2shpss.STFT(frame)
                # create random signal
                sig = np.random.normal(0, 1, siglen)
                # stft and istft
                amp, phase = stft.STFT(sig)
                sig_ = stft.iSTFT(amp, phase)
                # check length
                self.assertTrue(len(sig_) >= siglen)
                # evaluate sisdr
                sig_ = sig_[:siglen]
                sisdr = py2shpss.SISDR(sig_, sig)
                print(sisdr)
                self.assertTrue(sisdr > 50) # infty

    def HPSS_test(self):
        if False:
            hpss = py2shpss.HPSS("hm21")
            h, p = hpss(s)
            h_sig = py2shpss.STFT.iSTFT()
            py2shpss.SISDR()
            flag = np.abs(np.sum(h**2 + p**2 - s**2)) < np.abs(s**2)

if __name__ == '__main__':
    unittest.main()