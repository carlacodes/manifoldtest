# from pathlib import Path
import copy
from datetime import datetime
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from pathlib import Path
from umap import UMAP
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
import logging
import sys
''' Modified from Jules Lebert's code
spks was a numpy arrray of size trial* timebins*neuron, and bhv is  a pandas dataframe where each row represents a trial, the trial is the index '''
import os
from sklearn.base import BaseEstimator
from sklearn.model_selection import BaseCrossValidator
from sklearn.preprocessing import StandardScaler
import numpy as np

class ZScoreCV(BaseCrossValidator):
    def __init__(self, spks, custom_folds):
        self.spks = spks
        self.custom_folds = custom_folds

    def split(self, X, y=None, groups=None):
        for train_idx, test_idx in self.custom_folds:
            scaler = StandardScaler()
            # Fit on training data and transform both training and testing data
            self.spks[train_idx] = scaler.fit_transform(self.spks[train_idx])
            self.spks[test_idx] = scaler.transform(self.spks[test_idx])
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self.custom_folds)
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
        reducer_kwargs, logger, save_dir_path, use_rand_search=False, manual_params=None, rat_id=None, savedir=None
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

    if use_rand_search:


        param_grid = {
            'estimator__n_neighbors': [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 150, 200],
            'reducer__n_components': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            'reducer__random_state': [42],
            'estimator__metric': ['euclidean', 'cosine', 'minkowski'],
            'reducer__n_neighbors': [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 150, 200],
            'reducer__min_dist': [0.0001, 0.001, 0.01, 0.1, 0.3],
            'reducer__random_state': [42]
        }

        zscore_cv = ZScoreCV(spks, custom_folds)

        # Initialize BayesSearchCV
        random_search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_grid,
            n_iter=500,
            cv=zscore_cv,
            verbose=3,
            n_jobs=-1,
            scoring='r2'
        )

        # Fit BayesSearchCV
        random_search.fit(spks, y)
        sys.stdout = original_stdout
        log_file.close()

        # Get the best parameters and score
        best_params = random_search.best_params_
        best_score = random_search.best_score_
    else:

        train_scores = []
        test_scores = []

        count = 0
        for train_index, test_index in custom_folds:
            # Split the data into training and testing sets
            spks_train, spks_test = spks[train_index], spks[test_index]
            y_train, y_test = y[train_index], y[test_index]
            scaler = StandardScaler()
            spks_train_scaled = scaler.fit_transform(spks_train)
            spks_test_scaled = scaler.transform(spks_test)
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
    base_dir = '/ceph/scratch/carlag/honeycomb_neural_data/'

    for data_dir in [f'{base_dir}/rat_7/6-12-2019', f'{base_dir}/rat_10/23-11-2021',
                     f'{base_dir}/rat_8/15-10-2019', f'{base_dir}/rat_9/10-12-2021',
                     f'{base_dir}/rat_3/25-3-2019']:
        spike_dir = os.path.join(data_dir, 'physiology_data')
        dlc_dir = os.path.join(data_dir, 'positional_data')
        labels = np.load(f'{dlc_dir}/labels_250_raw.npy')
        col_list = np.load(f'{dlc_dir}/col_names_250_raw.npy')

        spike_data = np.load(f'{spike_dir}/inputs_10052024_250.npy')

        window_df = pd.read_csv(f'/ceph/scratch/carlag/honeycomb_neural_data/mean_p_value_vs_window_size_across_rats_grid_250_windows_scale_to_angle_range_False_allo_True.csv')
            #find the rat_id
        rat_id = data_dir.split('/')[-2]
        #filter for window_size
        window_df = window_df[window_df['window_size'] == 250]
        num_windows = window_df[window_df['rat_id'] == rat_id]['minimum_number_windows'].values[0]

        spike_data_copy = copy.deepcopy(spike_data)
        tolerance = 1e-10
        if np.any(np.abs(np.std(spike_data_copy, axis=0)) < tolerance):
            print('There are neurons with constant firing rates')
            spike_data_copy = spike_data_copy[:, np.abs(np.std(spike_data_copy, axis=0)) >= tolerance]

        percent_zeros = np.mean(spike_data_copy == 0, axis=0) * 100
        columns_to_remove = np.where(percent_zeros > 99.5)[0]
        spike_data_copy = np.delete(spike_data_copy, columns_to_remove, axis=1)
        X_for_umap = spike_data_copy

        labels_for_umap = labels
        label_df = pd.DataFrame(labels_for_umap, columns=col_list)

        regressor = KNeighborsRegressor
        regressor_kwargs = {'n_neighbors': 70}
        reducer = UMAP
        reducer_kwargs = {
            'n_components': 3,
            'metric': 'euclidean',
            'n_jobs': 1,
        }

        regress = ['x', 'y', 'cos_hd', 'sin_hd']

        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        now_day = datetime.now().strftime("%Y-%m-%d")
        filename = f'params_all_trials_randomizedsearchcv_250bin_1000windows_jake_fold_allvar_{now}.npy'
        filename_mean_score = f'mean_score_all_trials_randomizedsearchcv_250bin_1000windows_jake_fold_allvar_{now_day}.npy'
        save_dir_path = Path(f'{data_dir}/randsearch_sanitycheckallvarindepen_parallel2_{now_day}')
        save_dir = save_dir_path
        save_dir.mkdir(exist_ok=True)

        best_params, mean_score = train_and_test_on_umap_randcv(
            X_for_umap,
            label_df,
            regress,
            regressor,
            regressor_kwargs,
            reducer,
            reducer_kwargs, num_windows = num_windows
        )
        np.save(save_dir / filename, best_params)
        np.save(save_dir / filename_mean_score, mean_score)


if __name__ == '__main__':
    #
    main()