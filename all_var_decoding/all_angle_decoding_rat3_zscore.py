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
import os
import scipy
import pickle as pkl
from sklearn.base import BaseEstimator


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
        reducer_kwargs, logger, save_dir_path, use_bayes_search=False, manual_params=None, rat_id=None, savedir=None
):


    y = bhv[regress].values

    random_search_results = []

    # Create your custom folds
    n_timesteps = spks.shape[0]

    custom_folds = create_folds(n_timesteps, num_folds=5, num_windows=340)
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

    if use_bayes_search:
        # Define the parameter grid
        param_grid = {
            'estimator__estimator__n_neighbors': (2, 70),  # range of values
            'reducer__n_components': (3, 10),
            'estimator__estimator__metric': ['euclidean', 'cosine', 'minkowski'],
            'reducer__n_neighbors': (2, 70),  # range of values
            'reducer__min_dist': (0.0001, 0.3),  # range of values
            'reducer__random_state': [42]
        }

        # Initialize BayesSearchCV
        bayes_search = BayesSearchCV(
            pipeline,
            search_spaces=param_grid,
            n_iter=200,
            cv=custom_folds,
            verbose=3,
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
        count = 0
        for train_index, test_index in custom_folds:
            # Split the data into training and testing sets
            spks_train, spks_test = spks[train_index], spks[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Set the parameters
            formatted_params = format_params(manual_params)
            pipeline.set_params(**formatted_params)

            # Fit the pipeline on the training data
            pipeline.fit(spks_train, y_train)
            fitted_reducer = pipeline.named_steps['reducer']
            X_test_reduced = fitted_reducer.transform(spks_test)

            # Calculate the training score and append it to the list
            train_score = pipeline.score(spks_train, y_train)
            train_scores.append(train_score)

            # Calculate the test score and append it to the list
            test_score = pipeline.score(spks_test, y_test)
            test_scores.append(test_score)
            actual_angle = np.arcsin(y_test[:, 0])


            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            sc = ax.scatter(X_test_reduced[:, 0], X_test_reduced[:, 1], c=actual_angle, cmap='viridis')
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            ax.set_zlabel('UMAP 3')
            #add a color bar
            cbar = plt.colorbar(sc, ax=ax)
            ax.set_title('UMAP test embeddings color-coded by head angle rel. \n  to goal for fold: ' + str(count) + 'rat id:' +str(rat_id))
            plt.savefig(f'{savedir}/umap_embeddings_fold_' + str(count) + '.png', dpi=300, bbox_inches='tight')
            count += 1

        # Calculate the mean training and test scores
        mean_train_score = np.mean(train_scores)
        mean_test_score = np.mean(test_scores)

        # Print the mean scores
        print(f'Mean training score: {mean_train_score}')
        print(f'Mean test score: {mean_test_score}')
    return best_params, best_score


def main():
    data_dir = '/ceph/scratch/carlag/honeycomb_neural_data/rat_3/25-3-2019/'
    spike_dir = os.path.join(data_dir, 'physiology_data')
    dlc_dir = os.path.join(data_dir, 'positional_data')
    labels = np.load(
        f'{dlc_dir}/labels_250.npy')
    lfp_data = np.load(f'{spike_dir}/theta_sin_and_cos_bin_overlap_False_window_size_20.npy')
    spike_data = np.load(f'{spike_dir}/inputs_10052024_250.npy')
    #check if the old data ussed is the same as the new spike_data
    old_spike_data = np.load(f'{spike_dir}/inputs_overlap_False_window_size_250.npy')

    # if np.array_equal(spike_data, old_spike_data):
    #     print('The spike data is the same as the old spike data')
    # else:
    #     print('The spike data is not the same as the old spike data')

    # print out the first couple of rows of the lfp_data
    # previous_results, score_dict = DataHandler.load_previous_results('lfp_phase_manifold_withspkdata')
    rat_id = data_dir.split('/')[-2]
    # manual_params = previous_results[rat_id]
        # check for neurons with constant firing rates
    spike_data_copy = copy.deepcopy(spike_data)
    tolerance = 1e-10  # or any small number that suits your needs
    if np.any(np.abs(np.std(spike_data_copy, axis=0)) < tolerance):
        print('There are neurons with constant firing rates')
        # remove those neurons
        spike_data_copy = spike_data_copy[:, np.abs(np.std(spike_data_copy, axis=0)) >= tolerance]


    X_for_umap = spike_data_copy
    if np.isnan(X_for_umap).any():
        print('There are nans in the data')

    X_for_umap = scipy.stats.zscore(X_for_umap, axis=0)
    X_for_umap = scipy.ndimage.gaussian_filter(X_for_umap, 2, axes=0)



    labels_for_umap = labels[:, 0:5]
    labels_for_umap = scipy.ndimage.gaussian_filter(labels_for_umap, 2, axes=0)

    label_df = pd.DataFrame(labels_for_umap,
                            columns=['x', 'y', 'dist2goal', 'hd', 'relative_direction_to_goal'])
    label_df['time_index'] = np.arange(0, label_df.shape[0])
    label_df['angle_sin'] = np.sin(label_df['hd'])
    label_df['angle_cos'] = np.cos(label_df['hd'])
    label_df['angle_sin_goal'] = np.sin(label_df['relative_direction_to_goal'])
    label_df['angle_cos_goal'] = np.cos(label_df['relative_direction_to_goal'])


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

    regress = ['x', 'y', 'dist2goal', 'angle_sin', 'angle_cos', 'angle_sin_goal', 'angle_cos_goal']  # changing to two target variables

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    now_day = datetime.now().strftime("%Y-%m-%d")
    filename = f'params_lfp_all_trials_bayessearch_250bin_340windows_jake_fold_sinandcos_{now}.npy'
    filename_mean_score = f'mean_lfp_score_all_trials_bayessearch_250bin_340windows_jake_fold_sinandcos_{now_day}.npy'
    save_dir_path = Path(f'{data_dir}/bayesearch_allvars_{now_day}')
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
    # manual_params = manual_params.item()

    best_params, mean_score = train_and_test_on_umap_randcv(
        X_for_umap,
        label_df,
        regress,
        regressor,
        regressor_kwargs,
        reducer,
        reducer_kwargs, logger, save_dir_path, use_bayes_search=True, manual_params=None, savedir=save_dir_path, rat_id=rat_id
    )
    np.save(save_dir_path / filename, best_params)
    np.save(save_dir_path / filename_mean_score, mean_score)


if __name__ == '__main__':
    #
    main()