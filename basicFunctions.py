import numpy as np
import scipy as sp

def fivePointDer(signal,h,axis=-1,returnIndex=False):
    # takes the five point stencil as a principled estimate of the derivative
    assert isinstance(signal,np.ndarray), "signal must be a numpy array"
    assert isinstance(axis,(int,np.integer)), "axis must be an integer"
    assert -signal.ndim <= axis <= signal.ndim, "requested axis does not exist"
    N = signal.shape[axis]
    assert N >= 4*h+1, "h is too large for the given array -- it needs to be less than (N-1)/4!"
    signal = np.moveaxis(signal, axis, 0)
    n2 = slice(0,N-4*h)
    n1 = slice(h,N-3*h)
    p1 = slice(3*h,N-h)
    p2 = slice(4*h,N)
    fpd = (1/(12*h)) * (-signal[p2] + 8*signal[p1] - 8*signal[n1] + signal[n2])
    fpd = np.moveaxis(fpd, 0, axis)
    if returnIndex: return fpd, slice(2*h,N-2*h) # index of central points for each computation
    return fpd

def edge2center(edges):
    return np.mean(np.stack((edges[:-1],edges[1:])),axis=0)

















































