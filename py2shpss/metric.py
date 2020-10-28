import numpy as np

from nptyping import NDArray
from typing import Tuple, Any

def SISDR(
        x : NDArray[(Any,), float], 
        y : NDArray[(Any,), float]) -> float:
    """Evaluate SI-SDR between two signals.

    Args:
        x ([float] or numpy.array): a signal
        y ([float] or numpy.array): another signal

    Returns:
        float: SI-SDR between x and y
    """

    cos2_num = np.sum(x * y) ** 2
    cos2_den = np.sum(x ** 2) * np.sum(y ** 2)
    tan2_num = cos2_den - cos2_num
    tan2_den = cos2_num
    with np.errstate(divide='ignore'):
        log_abs_tan2 = np.log(np.abs(tan2_num)) - np.log(tan2_den)
    SISDR = -10 * log_abs_tan2 / np.log(10)
    return SISDR

def i_divergence(
            s1 : NDArray[(Any, Any), float], 
            s2 : NDArray[(Any, Any), float], 
            eps : float = 1e-100) -> float:
    with np.errstate(divide='ignore'):
        kl = - s1 * (np.log(s1 + eps) - np.log(s2 + eps))
    lin = - s1 + s2
    idiv = - np.mean(kl + lin)
    return idiv

def spectral_smoothness(
        spec: NDArray[(Any, Any), float]) -> Tuple[float, float]:
    # spec: (freq, time)
    t_diff = spec[:,1:] - spec[:,:-1]
    f_diff = spec[1:,:] - spec[:-1,:]
    t_diff = np.mean(t_diff ** 2)
    f_diff = np.mean(f_diff ** 2)
    return t_diff, f_diff