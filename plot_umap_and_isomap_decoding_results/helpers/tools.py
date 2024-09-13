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
from scipy.signal import lfilter

import matplotlib.pyplot as plt
import numpy as np

def create_folds(n_timesteps, num_folds=5, num_windows=10):
    n_windows_total = num_folds * num_windows
    window_size = n_timesteps / n_windows_total

    # window_start_ind = np.arange(0, n_windows_total) * window_size
    window_start_ind = np.round(np.arange(0, n_windows_total) * window_size)

    folds = []

    for i in range(num_folds):
        test_windows = np.arange(i, n_windows_total, num_folds)
        test_ind = []
        for j in test_windows:
            test_ind.extend(np.arange(window_start_ind[j], window_start_ind[j] + np.round(window_size)))

        train_ind = list(set(range(n_timesteps)) - set(test_ind))
        # convert test_ind to int
        test_ind = [int(i) for i in test_ind]

        folds.append((train_ind, test_ind))
        # print the ratio
        ratio = len(train_ind) / len(test_ind)
        print(f'Ratio of train to test indices is {ratio}')

    return folds

def format_params(params):
    formatted_params = {}
    for key, value in params.items():
        if key.startswith('estimator__'):
            # Add another 'estimator__' prefix to the key
            formatted_key = 'estimator__estimator__' + key[len('estimator__'):]
        else:
            formatted_key = key
        formatted_params[formatted_key] = value
    return formatted_params

def plot_isomap_mosaic(X_test_transformed, actual_angle, actual_distance, n_components, savedir, count):
    for i in range(n_components):
        # Create a mosaic layout
        layout = {
            'Angle': [f'ax_angle_{j}' for j in range(i + 1, n_components)],
            'Distance': [f'ax_distance_{j}' for j in range(i + 1, n_components)]
        }
        fig, axes = plt.subplot_mosaic([layout['Angle'], layout['Distance']], figsize=(15, 5))  # Landscape orientation
        fig.subplots_adjust(hspace=0.4, wspace=0.4)

        for j in range(i + 1, n_components):
            ax_angle = axes[f'ax_angle_{j}']
            sc_angle = ax_angle.scatter(X_test_transformed[:, i], X_test_transformed[:, j], c=actual_angle,
                                        cmap='twilight', s=10)
            ax_angle.set_xlabel(f'isomap {i + 1}')
            ax_angle.set_ylabel(f'isomap {j + 1}')
            fig.colorbar(sc_angle, ax=ax_angle)
            ax_angle.set_title(f'Angle: Component {i + 1} vs {j + 1}')

            ax_distance = axes[f'ax_distance_{j}']
            sc_distance = ax_distance.scatter(X_test_transformed[:, i], X_test_transformed[:, j], c=actual_distance,
                                              cmap='viridis', s=10)
            ax_distance.set_xlabel(f'isomap {i + 1}')
            ax_distance.set_ylabel(f'isomap {j + 1}')
            fig.colorbar(sc_distance, ax=ax_distance)
            ax_distance.set_title(f'Distance: Component {i + 1} vs {j + 1}')

        plt.savefig(f'{savedir}/isomap_embeddings_mosaic_fold_{count}_component_{i + 1}.png', dpi=300,
                    bbox_inches='tight')
        plt.show()
        plt.close('all')




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
