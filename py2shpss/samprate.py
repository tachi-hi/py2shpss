import sys
import numpy as np

def SampRate2FFTSize(sr):
    ''' 
    Find a number 2^n (0 < n), which is close to int(sr * factor) 
    (factor = 0.4 and 0.04)
    '''
    frame_long = sr * 0.4
    frame_short = sr * 0.04
    factor_long = int(np.log(frame_long)/np.log(2))
    factor_short = int(np.log(frame_short)/np.log(2))
    frame_long = 2 ** factor_long
    frame_short = 2 ** factor_short
    return frame_short, frame_long

