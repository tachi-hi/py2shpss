#!/usr/bin/env python

import pytest

import numpy as np
from py2shpss import samprate

def test_samprate():
    assert samprate.SampRate2FFTSize(16000) == (512, 4096)
    assert samprate.SampRate2FFTSize(44100) == (1024, 16384)

np.random.seed(123)
@pytest.mark.parametrize("sr", np.random.randint(4000, 96000, (100,)).tolist())
def test_samprate2(sr):
    short, long = samprate.SampRate2FFTSize(sr)
    assert short < long and long < sr
    assert short <= sr * 0.04 and sr * 0.04 < short * 2
    assert long  <= sr * 0.4  and sr * 0.4  < long  * 2