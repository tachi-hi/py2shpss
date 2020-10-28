import sys
import numpy as np
from typing import Tuple

def SampRate2FFTSize(sr : int) -> Tuple[int, int]:
    """ Find an appropriate FFT size.
    Find numbers 2^n (0 < n), which are close to int(sr * factor) 
    factor = 0.4 and 0.04.

    Args:
        sr (int): sampling rate

    Returns:
        (int, int): FFT_short and FFT_long
    """
    assert(sr >= 1)
    frame_long = sr * 0.4
    frame_short = sr * 0.04
    factor_long = int(np.log(frame_long)/np.log(2))
    factor_short = int(np.log(frame_short)/np.log(2))
    frame_long = 2 ** factor_long
    frame_short = 2 ** factor_short
    return frame_short, frame_long

