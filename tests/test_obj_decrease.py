#!/usr/bin/env python

import unittest

import numpy as np
import scipy.io.wavfile as wavfile
from py2shpss import py2shpss
from py2shpss import metric

np.random.seed(123)
class TestObjDecrease(unittest.TestCase):
    def test_hm21_obj_decrease(self):
        # load sig
        sr, sig = wavfile.read("../sampleSounds/doremi.wav")
        amp, phase = py2shpss.STFT(512).STFT(sig)
        # hpss
        hpss = py2shpss.HPSS(mode='hm21', eval_obj=True, iter=100)
        _, _, obj = hpss(amp)
        # check loss
        loss = [np.sum(_) for _ in obj]
        for x, y in zip(loss[:-1], loss[1:]):
            self.assertGreaterEqual(x, y)

    def test_idiv_obj_decrease(self):
        # load sig
        sr, sig = wavfile.read("../sampleSounds/doremi.wav")
        amp, phase = py2shpss.STFT(512).STFT(sig)
        # hpss
        qH = 0.1
        qP = 0.1
        hpss = py2shpss.HPSS(mode='idiv', eval_obj=True, qH = qH, qP = qP, iter=100)
        _, _, obj = hpss(amp)
        # check loss
        loss = [h/qH + p/qP + idiv for h, p, idiv in obj]
        for x, y in zip(loss[:-1], loss[1:]):
            self.assertGreaterEqual(x, y)

if __name__ == '__main__':
    unittest.main()