#!/usr/bin/env python

import unittest

import os
import numpy as np
import scipy.io.wavfile as wavfile
from py2shpss import py2shpss
from py2shpss.HPSS import HPSS
from py2shpss import metric

class TestObjDecrease(unittest.TestCase):
    here = os.path.dirname(os.path.abspath(__file__))
    wavpath = os.path.abspath(here + "/../sampleSounds/doremi.wav")
    fft_sizes = [128, 384, 1024]
    q = [0.001, 0.1, 100]

    def test_hm21_obj_decrease(self):
        # load sig
        sr, sig = wavfile.read(self.wavpath)
        self.assertEqual(sr, 8000)
        
        for fft_size in self.fft_sizes:
            # stft
            amp, phase = py2shpss.STFT(fft_size).STFT(sig)
            # hpss
            hpss = HPSS(mode='hm21', eval_obj=True, iter=100)
            _, _, obj = hpss(amp)
            # check loss
            loss = [np.sum(_) for _ in obj]
            for x, y in zip(loss[:-1], loss[1:]):
                self.assertGreaterEqual(x, y)

    def test_idiv_obj_decrease(self):
        # load sig
        sr, sig = wavfile.read(self.wavpath)
        self.assertEqual(sr, 8000)
        
        for fft_size in self.fft_sizes:
            for q in self.q:
                # stft
                amp, phase = py2shpss.STFT(fft_size).STFT(sig)
                # hpss
                qH = q
                qP = q
                hpss = HPSS(mode='idiv', eval_obj=True, qH = qH, qP = qP, iter=100)
                _, _, obj = hpss(amp)
                # check loss
                loss = [h/qH + p/qP + idiv for h, p, idiv in obj]
                for x, y in zip(loss[:-1], loss[1:]):
                    self.assertGreaterEqual(x, y)

if __name__ == '__main__':
    unittest.main()