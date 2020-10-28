import sys
import numpy as np
import scipy.signal
from typing import Optional
from py2shpss.HPSS import HPSS
from py2shpss import samprate as samprate_lib
from py2shpss import metric

class STFT(object):
    def __init__(self, fft_size : int):
        self.frame = fft_size
        self.win = np.sin(np.arange(fft_size) / fft_size * np.pi)

    def STFT(self, signal):
        compspec = scipy.signal.stft(
                signal,
                window=self.win,
                nperseg=self.frame,
                noverlap=self.frame//2,
                nfft=self.frame)[-1]
        return np.abs(compspec), np.angle(compspec)

    def iSTFT(self, x, phase):
        compspec = x * np.exp(1j * phase)
        return scipy.signal.istft(compspec,
            window=self.win,
            nperseg=self.frame,
            noverlap=self.frame // 2,
            nfft=self.frame)[-1]

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
        s, phase = self.stft_short.STFT(signal)
        hv, p, obj = self.hpss_short(s)
        hv = self.stft_short.iSTFT(hv, phase)
        p = self.stft_short.iSTFT(p, phase)

        s, phase = self.stft_long.STFT(hv)
        h, v, obj = self.hpss_long(s)
        h = self.stft_long.iSTFT(h, phase)
        v = self.stft_long.iSTFT(v, phase)

        return h, v, p
