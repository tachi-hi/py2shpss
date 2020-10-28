import sys
import numpy as np
import scipy.signal

from nptyping import NDArray
from typing import Any

from py2shpss import metric

class HPSS(object):
    def __init__(self, 
                mode : str = 'hm21', 
                iter : int = 30, 
                h_size : int = 1, 
                p_size : int = 1,
                eval_obj : bool =False, 
                *args, **kwargs):
        assert(iter >= 1)
        assert(h_size >= 1)
        assert(p_size >= 1)

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

    def __call__(self, spec : NDArray[(Any, Any), float], *args, **kwargs):
        """Run the HPSS algorithm on the given spectrogram.

        Args:
            spec (numpy 2D array): input spectrogram

        Returns:
            H (numpy 2D array): Harmonic spectrogram.
            P (numpy 2D array): Percussive spectrogram.
            obj (numpy 2D array, or None): Objective function log, if eval_obj is True
        """
        return self.call(spec, *args, **kwargs)

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
