# from pathlib import Path
import copy
from datetime import datetime
from sklearn.model_selection import ParameterSampler
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
# from helpers.datahandling import DataHandler
from scipy.stats import randint
from sklearn.neighbors import KNeighborsRegressor
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
from umap import UMAP
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from sklearn.pipeline import Pipeline
import logging
import sys

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