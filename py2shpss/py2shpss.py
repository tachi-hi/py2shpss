import sys
import numpy as np
import scipy.signal
from typing import Optional

from py2shpss.STFT import STFT
from py2shpss.HPSS import HPSS
from py2shpss import samprate as samprate_lib
from py2shpss import metric

class twostageHPSS(object):
    def __init__(self, mode="idiv", 
                samprate : Optional[int] = 16000 , 
                fft_short : Optional[int] = None, 
                fft_long : Optional[int] = None, 
                h_size : int = 1, 
                p_size : int = 1, 
                iter : int = 100, 
                *args, **kwargs):

        if samprate is None:
            pass
        else:
            assert(samprate >= 1)
            fft_short_, fft_long_ = samprate_lib.SampRate2FFTSize(samprate)
            if fft_short is not None:
                assert(fft_short_ == fft_short)
            if fft_long is not None:
                assert(fft_long_ == fft_long)
            fft_short, fft_long = fft_short_, fft_long_

        assert(fft_short >= 2)
        assert(fft_long >= 2)
        assert(fft_long > fft_short)

        self.stft_short = STFT(fft_short)
        self.stft_long = STFT(fft_long)
        self.hpss_short = HPSS(mode=mode, iter=iter, h_size=h_size, p_size=p_size, *args, **kwargs)
        self.hpss_long = HPSS(mode=mode, iter=iter, h_size=h_size, p_size=p_size, *args, **kwargs)

    def __call__(self, signal):
        s, phase = self.stft_short.forward(signal)
        hv, p, obj = self.hpss_short(s)
        hv = self.stft_short.inverse(hv, phase)
        p = self.stft_short.inverse(p, phase)

        s, phase = self.stft_long.forward(hv)
        h, v, obj = self.hpss_long(s)
        h = self.stft_long.inverse(h, phase)
        v = self.stft_long.inverse(v, phase)

        return h, v, p
