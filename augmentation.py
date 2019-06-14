from __future__ import division

__author__ = "Manu Airaksinen, manu.airaksinen@aalto.fi"

import os
import numpy as np

import soundfile
import sys


def load_noise_samples(wav_scp='noiselist.scp'):
    if wav_scp == None:
        return None

    with open(wav_scp) as wavlist:
        wavs = wavlist.read().splitlines()

    noise_samples = []
    for wav in wavs:
        y, _ = soundfile.read(wav)
        noise_samples.append(y)
    return noise_samples

 
def sampled_noise(N,noise_samples):
    # Random noise sample
    i_sample = np.random.randint(0,len(noise_samples))
    noise = noise_samples[i_sample]
    # Random starting location
    i_start = np.random.randint(0,len(noise)-N-1)
    noise = noise[i_start:i_start+N]
    return noise


# Controlled augmentation
def add_noise_file_controlled(X,snr,noise_type='white',noise_sample=None,run_codec=False):
    # Add additive noise
    e = np.linalg.norm(X)
    if noise_type == 'white':
        noise = np.random.randn(X.shape[0])
    elif noise_type == 'babble':
        # Random starting location
        N = X.shape[0]
        i_start = np.random.randint(0,len(noise_sample)-N-1)
        noise = noise_sample[i_start:i_start+N]

    en = np.linalg.norm(noise)
    gain = 10.0**(-1.0*snr/20.0)
    noise = gain * noise * e / en
    X += noise

    return X

# Random augmentation
def add_noise_file(X,noise_samples=None): 

    # Add channel noise (random impulse response)
    if np.random.rand() > 0.5:
        imp = np.random.randn(17,)
        gain_imp = np.random.rand()
        imp *= gain_imp
        imp[8] = 1.0
        X = np.convolve(X,imp,'same')

    # Add additive noise
    if np.random.rand() > 0.5:
        e = np.linalg.norm(X)
        if noise_samples == None:
            noise = np.random.randn(X.shape[0])
        else:
            noise = sampled_noise(X.shape[0],noise_samples)
        en = np.linalg.norm(noise)
        # Random snr for batch
        snr = np.float32(np.random.randint(-10, 20))
        gain = 10.0**(-1.0*snr/20.0)
        noise = gain * noise * e / en
        X += noise
        X *= 10.0**(-1.0*(np.random.rand()-0.5))
    
    return X


