from __future__ import division

__author__ = "Manu Airaksinen, manu.airaksinen@aalto.fi"

import os
import numpy as np
import tensorflow as tf

def get_frames(x,wl,hop):
    Nframes = int(np.ceil(x.shape[0]/hop))
    X = np.zeros((Nframes,int(wl)),dtype=np.float32)
    pad = np.zeros(int(wl/2),dtype=np.float32)
    x = np.concatenate((pad, x, pad))
    for i in range(Nframes):
        X[i,:] = x[i*hop:i*hop+wl]

    return X


def onehot(f0_vec,minf0=50,maxf0=500,Nbins=351):
    f0_onehot = np.zeros((f0_vec.shape[0],Nbins),dtype=np.float32)
    f0vec = np.exp(np.linspace(np.log(minf0),np.log(maxf0),Nbins-1))
    for i in range(f0_vec.shape[0]):
        if f0_vec[i] > 0.0:
            IND = np.argmin(np.abs(f0vec-f0_vec[i]))
            f0_onehot[i,IND] = 1.0
        else:
            f0_onehot[i,-1] = 1.0

    f0_onehot = np.reshape(f0_onehot,[1,-1,Nbins])
    return f0_onehot

def getF0fromActivations(f0_act,minf0=50,maxf0=500,Nbins=351):
    f0_gen = np.zeros((f0_act.shape[0]),dtype=np.float32)
    f0vec = np.exp(np.linspace(np.log(minf0),np.log(maxf0),Nbins-1))
    for i in range(f0_act.shape[0]):
        ind = np.argmax(f0_act[i,:])
        if ind < Nbins-1:
            f0_gen[i] = f0vec[ind]
        else:
            f0_gen[i] = 0
    return f0_gen