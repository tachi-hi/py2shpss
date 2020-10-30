#!/usr/bin/env python

import pytest

import numpy as np
from py2shpss import metric

@pytest.mark.parametrize("SDR", [-10, -5, 0, 5, 10])
def test_SISDR(SDR):
    np.random.seed(123)
    x = np.random.normal(0, 1, 16000)
    y = np.random.normal(0, 1, 16000)
    ratio = 10 ** (SDR / 20)
    mix = y + ratio * x
    SISDR = metric.SISDR(mix, x)
    assert np.abs(SISDR - SDR) < 1

