# from pathlib import Path
import copy
import sys
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

sys.path.append('C:/Users/Jake/Documents/python_code/robot_maze_analysis_code')
from utilities.get_directories import get_data_dir


def main():
    animal = 'Rat46'
    session = '19-02-2024'
    data_dir = get_data_dir(animal, session)
  
    goal = 52
    window_sizes = [25, 50, 100, 250, 500]

    for window_size in window_sizes: 

        ############ LOAD SPIKE DATA #####################
        # load numpy array of neural data
        spike_dir = os.path.join(data_dir, 'spike_sorting')
        inputs_path = os.path.join(spike_dir, 'inputs_for_embedding_and_decoding')
        inputs_file_name = f'inputs_goal{goal}_ws{window_size}'
        inputs = np.load(os.path.join(inputs_path, inputs_file_name + '.npy'))
        pca = PCA(n_components=100)
        pca.fit(inputs)
        explained_variance = pca.explained_variance_
        # explained_variance_ratio = pca.explained_variance_ratio_

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






if __name__ == '__main__':
    #
    main()