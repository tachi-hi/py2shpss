#!/usr/bin/env python

import unittest

import numpy as np
from py2shpss import metric

np.random.seed(123)
class TestMetric(unittest.TestCase):
    def test_SISDR(self):
        x = np.random.normal(0, 1, 16000)
        y = np.random.normal(0, 1, 16000)
        for SDR in [-10, -5, 0, 5, 10]:
            ratio = 10 ** (SDR / 20)
            mix = y + ratio * x
            SISDR = metric.SISDR(mix, x)
            print(SDR, SISDR)
            self.assertTrue(np.abs(SISDR - SDR) < 1)

if __name__ == '__main__':
    unittest.main()