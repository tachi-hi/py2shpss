import sys
import numpy as np
import scipy.signal

class STFT(object):
    def __init__(self, fft_size : int):
        self.frame = fft_size
        self.win = np.sin(np.arange(fft_size) / fft_size * np.pi)

    def forward(self, signal):
        compspec = scipy.signal.stft(
                signal,
                window=self.win,
                nperseg=self.frame,
                noverlap=self.frame//2,
                nfft=self.frame)[-1]
        return np.abs(compspec), np.angle(compspec)

    def inverse(self, x, phase):
        compspec = x * np.exp(1j * phase)
        return scipy.signal.istft(compspec,
            window=self.win,
            nperseg=self.frame,
            noverlap=self.frame // 2,
            nfft=self.frame)[-1]

