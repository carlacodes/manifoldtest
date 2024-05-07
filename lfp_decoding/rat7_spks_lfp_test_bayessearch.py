# from pathlib import Path
import copy
from datetime import datetime
from sklearn.model_selection import ParameterSampler
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from manifold_neural.helpers.datahandling import DataHandler
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
os.environ['JOBLIB_TEMP_FOLDER'] = 'C:/tmp'


# TODO: 1. change hyperparameters to normalise y = True and kernel = (constant kernel * RBF) + white kernel
# 2. change the regressor to GaussianProcessRegressor
# 3. should the umap X_training data be 2d rather than 3d? Also need to z-score the X input data
# 4. in the 2021 sci advances paper they used 2 fold cross validation
# 5. for the isomap they used n_neighbours = 20 #
# 6. they used the gaussian-filtered (omega = 2-time bins) square root of instantenous firing rates for the isomap decomposition
# 7. bin duration = 512 ms, so about the same as what I have
# 8. target position was smoothed using a gaussian filter


class CustomUMAP(BaseEstimator):
    def __init__(self, n_neighbors=15, n_components=2, metric='euclidean',
                 n_epochs=None, learning_rate=1.0, init='spectral',
                 min_dist=0.1, spread=1.0, low_memory=False,
                 random_state=None, verbose=False):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.metric = metric
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.init = init
        self.min_dist = min_dist
        self.spread = spread
        self.low_memory = low_memory
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y=None):
        self.model_ = UMAP(n_neighbors=self.n_neighbors,
                           n_components=self.n_components,
                           metric=self.metric,
                           n_epochs=self.n_epochs,
                           learning_rate=self.learning_rate,
                           init=self.init,
                           min_dist=self.min_dist,
                           spread=self.spread,
                           low_memory=self.low_memory,
                           random_state=self.random_state,
                           verbose=self.verbose)

        self.model_.fit(X)
        return self

    def transform(self, X):
        return self.model_.transform(X)


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

def train_and_test_on_umap_randcv(
        spks,
        bhv,
        regress,
        regressor,
        regressor_kwargs,
        reducer,
        reducer_kwargs, logger, save_dir_path, use_bayes_search=False, manual_params=None
):


    y = bhv[regress].values

    random_search_results = []

    # Create your custom folds
    n_timesteps = spks.shape[0]

    custom_folds = create_folds(n_timesteps, num_folds=10, num_windows=1000)
    # Example, you can use your custom folds here
    pipeline = Pipeline([
        ('reducer', CustomUMAP()),
        ('estimator', MultiOutputRegressor(regressor()))
    ])

    # Define the parameter grid
    # param_grid = {
    #     'estimator__n_neighbors': [2, 5, 10, 30, 40, 50, 60, 70],
    #     'reducer__n_components': [2],
    #     'estimator__metric': ['euclidean', 'cosine', 'minkowski'],
    #     'reducer__n_neighbors': [10, 20, 30, 40, 50, 60, 70],
    #     'reducer__min_dist': [0.0001, 0.001, 0.01, 0.1, 0.3],
    #     'reducer__random_state': [42]
    # }
    # get the date
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = open(f"{save_dir_path}/random_search_{now}.log", "w")

    # Save the original stdout
    original_stdout = sys.stdout

    # Redirect stdout to the log file
    sys.stdout = log_file

    param_grid = {
        'estimator__estimator__n_neighbors': (2, 70),  # range of values
        'reducer__n_components': [2],
        'estimator__estimator__metric': ['euclidean', 'cosine', 'minkowski'],
        'reducer__n_neighbors': (10, 70),  # range of values
        'reducer__min_dist': (0.0001, 0.3),  # range of values
        'reducer__random_state': [42]
    }

    if use_bayes_search:
        # Define the parameter grid
        param_grid = {
            'estimator__estimator__n_neighbors': (2, 70),  # range of values
            'reducer__n_components': [2],
            'estimator__estimator__metric': ['euclidean', 'cosine', 'minkowski'],
            'reducer__n_neighbors': (10, 70),  # range of values
            'reducer__min_dist': (0.0001, 0.3),  # range of values
            'reducer__random_state': [42]
        }

        # Initialize BayesSearchCV
        bayes_search = BayesSearchCV(
            pipeline,
            search_spaces=param_grid,
            n_iter=200,
            cv=custom_folds,
            verbose=2,
            n_jobs=-1,
            scoring='r2'
        )

        # Fit BayesSearchCV
        bayes_search.fit(spks, y)
        sys.stdout = original_stdout

        log_file.close()

        # Get the best parameters and score
        best_params = bayes_search.best_params_
        best_score = bayes_search.best_score_
    else:
        # Manually set the parameters

        # Initialize lists to store the scores
        train_scores = []
        test_scores = []

        # Loop over the custom folds
        for train_index, test_index in custom_folds:
            # Split the data into training and testing sets
            spks_train, spks_test = spks[train_index], spks[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Set the parameters
            formatted_params = format_params(manual_params)
            pipeline.set_params(**formatted_params)

            # Fit the pipeline on the training data
            pipeline.fit(spks_train, y_train)

            # Calculate the training score and append it to the list
            train_score = pipeline.score(spks_train, y_train)
            train_scores.append(train_score)

            # Calculate the test score and append it to the list
            test_score = pipeline.score(spks_test, y_test)
            test_scores.append(test_score)

        # Calculate the mean training and test scores
        mean_train_score = np.mean(train_scores)
        mean_test_score = np.mean(test_scores)

        # Print the mean scores
        print(f'Mean training score: {mean_train_score}')
        print(f'Mean test score: {mean_test_score}')
    return best_params, best_score


def main():
    data_dir = 'C:/neural_data/rat_7/6-12-2019'
    spike_dir = os.path.join(data_dir, 'physiology_data')
    dlc_dir = os.path.join(data_dir, 'positional_data')
    labels = np.load(
        f'{dlc_dir}/labels_1203_with_goal_centric_angle_scale_data_False_zscore_data_False_overlap_False_window_size_20.npy')
    lfp_data = np.load(f'{spike_dir}/theta_sin_and_cos_bin_overlap_False_window_size_20.npy')
    spike_data = np.load(f'{spike_dir}/inputs_overlap_False_window_size_300.npy')
    # print out the first couple of rows of the lfp_data
    previous_results, score_dict = DataHandler.load_previous_results('lfp_phase_manifold_withspkdata')
    rat_id = data_dir.split('/')[-2]
    manual_params = previous_results[rat_id]

    # make copies of each the spike data so it fits the shape of the lfp_data
    # find the ratio in length between the lfp_data and the spike_data
    ratio = lfp_data.shape[0] / spike_data.shape[0]
    # round to the nearest integer
    ratio = int(np.round(ratio))
    for i in range(0, len(spike_data)):
        # use np.tile to repeat the spike_data
        repeated_spike_data = np.tile(spike_data[i], (ratio, 1))
        # append to a list
        if i == 0:
            repeated_spike_data_array = repeated_spike_data
        else:
            repeated_spike_data_array = np.vstack((repeated_spike_data_array, repeated_spike_data))

    if len(repeated_spike_data_array) != len(lfp_data):
        print('The shape of the repeated spike data is not the same as the lfp data')
        if len(repeated_spike_data_array) > len(lfp_data):
            print('The shape of the repeated spike data is greater than the lfp data')
            repeated_spike_data_array = repeated_spike_data_array[:len(lfp_data), :]  #
        else:
            print('The shape of the repeated spike data is less than the lfp data')
            lfp_data = lfp_data[:len(repeated_spike_data_array), :]

    data_dir_path = Path(data_dir)

    # check for neurons with constant firing rates
    tolerance = 1e-10  # or any small number that suits your needs

    if np.any(np.abs(np.std(repeated_spike_data_array, axis=0)) < tolerance):
        print('There are neurons with constant firing rates')
        # remove those neurons
        repeated_spike_data_array = repeated_spike_data_array[:,
                                    np.abs(np.std(repeated_spike_data_array, axis=0)) >= tolerance]

    X_for_umap = scipy.stats.zscore(repeated_spike_data_array, axis=0)

    if np.isnan(X_for_umap).any():
        print('There are nans in the data')

    X_for_umap = scipy.ndimage.gaussian_filter(repeated_spike_data_array, 2, axes=0)
    X_for_umap = np.concatenate((X_for_umap, lfp_data), axis=1)

    # as a check, plot the firing rates for a single neuron before and after smoothing
    # fig, ax = plt.subplots(1, 2)
    # ax[0].plot(X_for_umap[:, 0])
    # ax[0].set_title('Before smoothing')
    # ax[1].plot(X_for_umap_smooth[ :, 0])
    # ax[1].set_title('After smoothing')
    # plt.show()

    labels_for_umap = labels[:, 0:9]
    labels_for_umap = scipy.ndimage.gaussian_filter(labels_for_umap, 2, axes=0)

    label_df = pd.DataFrame(labels_for_umap,
                            columns=['x', 'y', 'dist2goal', 'angle_sin', 'angle_cos', 'dlc_angle_zscore',
                                     'angle_rel_to_goal', 'angle_rel_to_goal_sin', 'angle_rel_to_goal_cos'])
    label_df['time_index'] = np.arange(0, label_df.shape[0])
    # crop the label_df to only take the first 20,000 samples
    print(f'The shape of the label_df is {label_df.shape}')
    label_df = label_df.iloc[:91817, :]

    # print the shape of the label_df
    print(f'The shape of the label_df is {label_df.shape}')
    # dpo the same for the X_for_umap
    X_for_umap = X_for_umap[:91817, :]

    regressor = KNeighborsRegressor
    regressor_kwargs = {'n_neighbors': 70, 'metric': 'euclidean'}

    reducer = UMAP

    reducer_kwargs = {
        'n_components': 3,
        # 'n_neighbors': 70,
        # 'min_dist': 0.3,
        'metric': 'euclidean',
        'n_jobs': 1,
    }

    regress = ['angle_sin', 'angle_cos']  # changing to two target variables

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    now_day = datetime.now().strftime("%Y-%m-%d")
    filename = f'params_lfp_all_trials_randomizedsearchcv_20bin_1000windows_jake_fold_sinandcos_{now}.npy'
    filename_mean_score = f'mean_lfp_score_all_trials_randomizedsearchcv_20bin_1000windows_jake_fold_sinandcos_{now_day}.npy'
    save_dir_path = data_dir_path / 'lfp_phase_manifold_withspkdata'
    save_dir_path.mkdir(parents=True, exist_ok=True)
    # initalise a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # create a file handler
    handler = logging.FileHandler(save_dir_path / f'lfp_phase_manifold_withspkdata_{now}.log')
    handler.setLevel(logging.INFO)
    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(handler)
    logger.info('Starting the training and testing of the lfp data with the spike data')
    #remove numpy array, just get mapping from manual_params
    manual_params = manual_params.item()

    best_params, mean_score = train_and_test_on_umap_randcv(
        X_for_umap,
        label_df,
        regress,
        regressor,
        regressor_kwargs,
        reducer,
        reducer_kwargs, logger, save_dir_path, use_bayes_search=False, manual_params=manual_params
    )
    np.save(save_dir_path / filename, best_params)
    np.save(save_dir_path / filename_mean_score, mean_score)


if __name__ == '__main__':
    #
    main()