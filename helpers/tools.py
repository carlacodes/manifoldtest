
import numpy as np

''' Modified from Jules Lebert's code
spks was a numpy arrray of size trial* timebins*neuron, and bhv is  a pandas dataframe where each row represents a trial, the trial is the index '''
import os
import scipy
import pickle as pkl
from sklearn.base import BaseEstimator
from scipy.signal.windows import gaussian
import matplotlib.pyplot as plt
from scipy.signal import lfilter

def apply_lfads_smoothing(data_in):
    std_sec = 0.25
    bin_width_sec = 0.25
    # Scale the width of the Gaussian by our bin width
    std = std_sec / bin_width_sec
    # We need a window length of 3 standard deviations on each side (x2)
    M = std * 3 * 2
    window = gaussian(M, std)
    # Normalize so the window sums to 1
    window = window / window.sum()
    # _ = plt.stem(window)

    # Remove convolution artifacts
    invalid_len = len(window) // 2

    # smth_spikes = {}
    # for session in spikes:
    #     # Convolve each session with the gaussian window
    #     smth_spikes[session] = lfilter(window, 1, spikes[session], axis=1)[:, invalid_len:, :]
    X_umap_lfads = lfilter(window, 1, data_in, axis=0)[invalid_len:, :]

    removed_row_indices = np.arange(0, invalid_len)


    # then z score
    return X_umap_lfads, removed_row_indices