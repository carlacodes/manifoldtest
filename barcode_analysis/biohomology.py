import copy
from datetime import datetime
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from pathlib import Path
from sklearn.metrics import r2_score
from manifold_neural.helpers.datahandling import DataHandler
import matplotlib.pyplot as plt
import gudhi as gd
from umap import UMAP
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
import sys
import os
from sklearn.base import BaseEstimator
from sklearn.model_selection import BaseCrossValidator
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.manifold import Isomap
import logging
from ripser import ripser
import persim
from gtda.homology import VietorisRipsPersistence, SparseRipsPersistence
from gtda.plotting import plot_diagram
import matplotlib.pyplot as plt
from data.generate_datasets import make_point_clouds
def plot_barcode(diag, dim, **kwargs):
    """
    Plot the barcode for a persistence diagram using matplotlib
    ----------
    diag: np.array: of shape (num_features, 3), i.e. each feature is
           a triplet of (birth, death, dim) as returned by e.g.
           VietorisRipsPersistence
    dim: int: Dimension for which to plot
    **kwargs
    Returns
    -------
    None.

    """
    diag_dim = diag[diag[:, 2] == dim]
    birth = diag_dim[:, 0]; death = diag_dim[:, 1]
    finite_bars = death[death != np.inf]
    if len(finite_bars) > 0:
        inf_end = 2 * max(finite_bars)
    else:
        inf_end = 2
    death[death == np.inf] = inf_end
    plt.figure(figsize=kwargs.get('figsize', (10, 5)))
    for i, (b, d) in enumerate(zip(birth, death)):
        if d == inf_end:
            plt.plot([b, d], [i, i], color='k', lw=kwargs.get('linewidth', 2))
        else:
            plt.plot([b, d], [i, i], color=kwargs.get('color', 'b'), lw=kwargs.get('linewidth', 2))
    plt.title(kwargs.get('title', 'Persistence Barcode'))
    plt.xlabel(kwargs.get('xlabel', 'Filtration Value'))
    plt.yticks([])
    plt.tight_layout()
    plt.show()

def run_persistence_analysis(folder_str):
    for i in range(5):
        print('at count ', i)
        reduced_data = np.load(folder_str + '/X_test_transformed_fold_' + str(i) + '.npy')
        subsample_size = 1000  # Define the size of the subsample
        if reduced_data.shape[0] > subsample_size:
            indices = np.random.choice(reduced_data.shape[0], subsample_size, replace=False)
            reduced_data = reduced_data[indices, :]

        # Compute persistence using SparseRipsPersistence
        persistence = SparseRipsPersistence(homology_dimensions=[0, 1, 2], n_jobs=-1, epsilon=1)
        reduced_data_input = [reduced_data]
        diagrams = persistence.fit_transform(reduced_data_input)

        plot_barcode(diagrams, 1)
        # plt.show()
        # plt.close('all')
    return



def main():
    #load the already reduced data
    base_dir = 'C:/neural_data/'
    for dir in [f'{base_dir}/rat_10/23-11-2021',f'{base_dir}/rat_8/15-10-2019', f'{base_dir}/rat_9/10-12-2021', f'{base_dir}/rat_3/25-3-2019', f'{base_dir}/rat_7/6-12-2019']:
        sub_folder = dir + '/plot_results/'
        #get list of files in the directory
        files = os.listdir(sub_folder)
        #check if more than two dirs
        if len(files) > 2:
            #choose the most recently modified directory
            files.sort(key=lambda x: os.path.getmtime(sub_folder + x))
            #get the most recently modified directory
            savedir = sub_folder + files[-1]
        else:
            savedir = sub_folder + files[0]

        run_persistence_analysis(savedir)










if __name__ == '__main__':
    main()
