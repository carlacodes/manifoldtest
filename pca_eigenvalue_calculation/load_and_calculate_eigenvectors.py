# from pathlib import Path
import copy
from datetime import datetime
from tqdm import tqdm
from joblib import Parallel, delayed
# from extractlfpandspikedata import load_theta_data
from sklearn.model_selection import ParameterSampler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from scipy.stats import randint
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, TimeSeriesSplit, permutation_test_score, \
    GridSearchCV, \
    RandomizedSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.dummy import DummyClassifier
from pathlib import Path
import numpy as np
import pandas as pd
# mpl.use('Qt5Agg')  # or can use 'TkAgg', whatever you have/prefer
''' Modified from Jules Lebert's code
spks was a numpy arrray of size trial* timebins*neuron, and bhv is  a pandas dataframe where each row represents a trial, the trial is the index '''
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import ParameterGrid
from sklearn.gaussian_process.kernels import WhiteKernel, ConstantKernel, RBF
import os
import scipy
import pickle as pkl


def main():
    data_dir = 'C:/neural_data/rat_7/6-12-2019/'
    spike_dir = os.path.join(data_dir, 'physiology_data')
    dlc_dir = os.path.join(data_dir, 'positional_data')

    # load labels
    # labels = np.load(f'{dlc_dir}/labels_0403_with_dist2goal_scale_data_False_zscore_data_False.npy')

    labels = np.load(f'{dlc_dir}/labels_1103_with_dist2goal_scale_data_False_zscore_data_False_overlap_False.npy')

    spike_data = np.load(f'{spike_dir}/inputs_overlap_False.npy')
    # extract the 10th trial

    spike_data_trial = spike_data


    data_dir_path = Path(data_dir)



    # check for neurons with constant firing rates
    tolerance = 1e-10  # or any small number that suits your needs
    if np.any(np.abs(np.std(spike_data_trial, axis=0)) < tolerance):
        print('There are neurons with constant firing rates')
        # remove those neurons
        spike_data_trial = spike_data_trial[:, np.abs(np.std(spike_data_trial, axis=0)) >= tolerance]
    # THEN DO THE Z SCORE
    X_for_umap = scipy.stats.zscore(spike_data_trial, axis=0)
    if np.isnan(X_for_umap).any():
        print('There are nans in the data')

    X_for_umap = scipy.ndimage.gaussian_filter(X_for_umap, 2, axes=0)

    # as a check, plot the firing rates for a single neuron before and after smoothing
    # fig, ax = plt.subplots(1, 2)
    # ax[0].plot(X_for_umap[:, 0])
    # ax[0].set_title('Before smoothing')
    # ax[1].plot(X_for_umap_smooth[ :, 0])
    # ax[1].set_title('After smoothing')
    # plt.show()

    labels_for_umap = labels[:, 0:6]
    # apply the same gaussian smoothing to the labels
    labels_for_umap = scipy.ndimage.gaussian_filter(labels_for_umap, 2, axes=0)

    label_df = pd.DataFrame(labels_for_umap,
                            columns=['x', 'y', 'dist2goal', 'angle_sin', 'angle_cos', 'dlc_angle_zscore'])
    label_df['time_index'] = np.arange(0, label_df.shape[0])

    explained_variance_list = []
    explained_variance_ratio_list = []
    # for n_components in range(1, 100):
    #     pca = PCA(n_components=n_components)
    #     pca.fit(X_for_umap)
    #     explained_variance = pca.explained_variance_
    #     explained_variance_ratio = pca.explained_variance_ratio_
    #
    #     print(explained_variance)
    #     print(np.sum(explained_variance))
    #     print(n_components)
    #     explained_variance_list.append(np.sum(explained_variance))
    #     explained_variance_ratio_list.append(np.sum(explained_variance_ratio))
    # fig, ax = plt.subplots()
    # plt.plot(np.arange(1, 100), explained_variance_list)
    # plt.title('Explained variance vs number of components')
    # plt.savefig('figures/explained_variance_vs_num_components.png')
    # plt.show()
    # fig, ax = plt.subplots()
    # plt.plot(np.arange(1, 100), explained_variance_ratio_list)
    # plt.title('Explained variance ratio vs number of components')
    # plt.savefig('figures/explained_variance_ratio_vs_num_components.png')
    # plt.show()
    # pca = PCA(n_components=3)

    pca = PCA(n_components=112)
    pca.fit(X_for_umap)
    explained_variance = pca.explained_variance_
    explained_variance_ratio = pca.explained_variance_ratio_

    print(explained_variance)
    print(np.sum(explained_variance))
    fig, ax = plt.subplots()
    PC_values = np.arange(pca.n_components_) + 1
    plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
    plt.title('Scree Plot')
    plt.xticks(np.arange(1, 101, 5))
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained')
    plt.show()
    #calculatie the inflection point
    #calculate the gradient of the explained variance ratio
    explained_variance_ratio_gradient = np.gradient(pca.explained_variance_ratio_)
    fig, ax = plt.subplots()
    plt.plot(np.arange(1, 113), explained_variance_ratio_gradient)
    plt.title('Gradient of explained variance ratio')
    plt.show()







if __name__ == '__main__':
    #
    main()