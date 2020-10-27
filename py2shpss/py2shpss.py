import sys
import numpy as np
import scipy.signal
from py2shpss import samprate as samprate_lib
from py2shpss import metric

class HPSS(object):
    def __init__(self, mode='hm21', iter=30, h_size=1, p_size=1, eval_obj=False, *args, **kwargs):
        self.h_filter, self.p_filter = self.__create_filter(h_size, p_size)
        self.iter = iter
        self.eval_obj = eval_obj

        if mode == 'hm21':
            self.call = self._call_hm21
        elif mode == 'idiv':
            self.call = self._call_idiv
            self.qH = kwargs["qH"] if "qH" in kwargs.keys() else 0.1
            self.qP = kwargs["qP"] if "qP" in kwargs.keys() else 0.1
        else:
            self.call = self._call_hm21
            print("Caution: mode should be either hm21 or idiv. (hm21 is set.)", file=sys.stderr)

    def __create_filter(self, h_size, p_size):
        h_filter = np.ones(1 + 2 * h_size)
        p_filter = np.ones(1 + 2 * p_size)
        h_filter[h_size] = 0
        p_filter[p_size] = 0
        h_filter = h_filter / np.sum(h_filter)
        p_filter = p_filter / np.sum(p_filter)
        h_filter = np.expand_dims(h_filter, 0)
        p_filter = np.expand_dims(p_filter, 1)
        return h_filter, p_filter

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def _call_hm21(self, Y):
        H = Y / np.sqrt(2)
        P = Y / np.sqrt(2)
        if self.eval_obj:
            obj = []
        for i in range(self.iter):
            H_tmp = scipy.signal.convolve2d(H, self.h_filter, boundary='fill', mode='same', fillvalue=0)
            P_tmp = scipy.signal.convolve2d(P, self.p_filter, boundary='fill', mode='same', fillvalue=0)
            d = np.sqrt(H_tmp ** 2 + P_tmp ** 2)
            H = Y * H_tmp / d
            P = Y * P_tmp / d
            if self.eval_obj:
                h_smoothness = metric.spectral_smoothness(H)[0]
                p_smoothness = metric.spectral_smoothness(P)[1]
                obj.append([h_smoothness, p_smoothness])
        return H, P, (obj if self.eval_obj else None)

    def _call_idiv(self, Y):
        H = Y / np.sqrt(2)
        P = Y / np.sqrt(2)
        M = H * 0 + 0.5
        if self.eval_obj:
            obj = []
        for i in range(self.iter):
            ah = 2*(1 + self.qH)
            ap = 2*(1 + self.qP)
            bh = scipy.signal.convolve2d(H, self.h_filter, boundary='fill', mode='same', fillvalue=0)
            bp = scipy.signal.convolve2d(P, self.p_filter, boundary='fill', mode='same', fillvalue=0)
            ch = 2 * self.qH * M * Y**2
            cp = 2 * self.qP * (1 - M) * Y**2
            H = (bh + np.sqrt(bh ** 2 + ah * ch)) /ah
            P = (bp + np.sqrt(bp ** 2 + ap * cp)) /ap
            M = H**2/(H**2 + P**2 + 1e-10)
            if self.eval_obj:
                h_smoothness = metric.spectral_smoothness(H)[0]
                p_smoothness = metric.spectral_smoothness(P)[1]
                idiv = metric.i_divergence(Y**2, H**2 + P**2)
                obj.append([h_smoothness, p_smoothness, idiv])
        H = M * Y
        P = (1 - M) * Y
        return H, P, (obj if self.eval_obj else None)

class STFT(object):
    def __init__(self, fft_size):
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
    def __init__(self, mode="idiv", samprate=16000, fft_short=None, fft_long=None, h_size=1, p_size=1, iter=100, *args, **kwargs):
        if samprate is None:
            fft_short = fft_short
            fft_long = fft_long
        else:
            fft_short_, fft_long_ = samprate_lib.SampRate2FFTSize(samprate)
            if fft_short is not None:
                assert(fft_short_ == fft_short)
            if fft_long is not None:
                assert(fft_long_ == fft_long)
            fft_short, fft_long = fft_short_, fft_long_

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

