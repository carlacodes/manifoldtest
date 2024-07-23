# from pathlib import Path
import copy
from datetime import datetime
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from pathlib import Path
from sklearn.metrics import r2_score
from manifold_neural.helpers.datahandling import DataHandler
import matplotlib.pyplot as plt
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

def train_and_test_on_isomap_randcv(
        spks,
        bhv,
        regress,
        regressor,
        regressor_kwargs,
        reducer,
        reducer_kwargs, use_rand_search=False, manual_params=None, rat_id=None, savedir=None, num_windows=None):


    y = bhv[regress].values

    # Create your custom folds
    n_timesteps = spks.shape[0]

    custom_folds = create_folds(n_timesteps, num_folds=5, num_windows=num_windows)
    # Example, you can use your custom folds here
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('reducer', Isomap()),
        ('estimator', MultiOutputRegressor(regressor()))
    ])
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = open(f"{savedir}/random_search_{now}.log", "w")

    # Save the original stdout
    original_stdout = sys.stdout

    # Redirect stdout to the log file
    sys.stdout = log_file
    best_params = None
    if use_rand_search:


        param_grid = {
            'estimator__estimator__n_neighbors': [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 150, 200],
            'reducer__n_components': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            'estimator__estimator__metric': ['euclidean', 'cosine', 'minkowski'],
            'reducer__metric': ['euclidean', 'cosine', 'minkowski'],
            'reducer__n_neighbors': [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 150, 200],
        }

        zscore_cv = ZScoreCV(spks, custom_folds)

        # Initialize BayesSearchCV
        random_search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_grid,
            n_iter=500,
            cv=zscore_cv,
            verbose=3,
            n_jobs=10,
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
        fold_dataframe = pd.DataFrame()#
        fold_dataframe_shuffle = pd.DataFrame()
        for train_index, test_index in custom_folds:
            # Split the data into training and testing sets
            spks_train, spks_test = spks[train_index], spks[test_index]
            y_train, y_test = y[train_index], y[test_index]
            y_test_shuffle = copy.deepcopy(y_test)
            np.random.shuffle(y_test_shuffle)
            y_train_shuffle = copy.deepcopy(y_train)
            np.random.shuffle(y_train_shuffle)


            # Set the parameters
            # formatted_params = format_params(manual_params)
            pipeline.set_params(**manual_params)

            # Fit the pipeline on the training data
            pipeline.fit(spks_train, y_train)

            # Use the pipeline to predict on the test set
            y_pred = pipeline.predict(spks_test)
            y_pred_shuffle = pipeline.predict(spks_test)
            #get the individaul scores
            indiv_results_dataframe = pd.DataFrame()
            indiv_results_dataframe_shuffle = pd.DataFrame()

            for i in range(y_test.shape[1]):
                score_indiv = r2_score(y_test[:, i], y_pred[:, i])
                score_indiv_shuffle = r2_score(y_test_shuffle[:, i], y_pred_shuffle[:, i])
                indiv_results_dataframe = pd.concat(
                    [indiv_results_dataframe, pd.DataFrame([score_indiv], columns=[regress[i]])], axis=1)
                indiv_results_dataframe_shuffle = pd.concat(
                    [indiv_results_dataframe_shuffle, pd.DataFrame([score_indiv_shuffle], columns=[regress[i]])],
                    axis=1)

                print(f'R2 score for {regress[i]} is {score_indiv}')
            # break down the score into its components
            indiv_results_dataframe['fold'] = count
            indiv_results_dataframe_shuffle['fold'] = count

            fold_dataframe = pd.concat([fold_dataframe, indiv_results_dataframe], axis=0)
            fold_dataframe_shuffle = pd.concat([fold_dataframe_shuffle, indiv_results_dataframe_shuffle], axis=0)

            # Calculate the training and test scores
            train_score = pipeline.score(spks_train, y_train)
            test_score = pipeline.score(spks_test, y_test)
            train_scores.append(train_score)
            test_scores.append(test_score)

            # Extract transformed test data for plotting
            # Assuming 'reducer' is the name of the dimensionality reduction step in your pipeline
            X_test_transformed = pipeline.named_steps['reducer'].transform(
                pipeline.named_steps['scaler'].transform(spks_test))

            actual_angle = np.arctan2(y_test[:, 2], y_test[:, 3])
            actual_distance = np.sqrt(y_test[:, 0] ** 2 + y_test[:, 1] ** 2)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            sc = ax.scatter(X_test_transformed[:, 0], X_test_transformed[:, 1], c=actual_angle, cmap='viridis')
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.set_zlabel('Component 3')
            cbar = plt.colorbar(sc, ax=ax)
            ax.set_title(
                f'Isomap test embeddings color-coded by head angle rel. to goal for fold: {count} rat id: {rat_id}')
            plt.savefig(f'{savedir}/isomap_embeddings_fold_{count}.png', dpi=300, bbox_inches='tight')
            n_components = X_test_transformed.shape[1]

            # Iterate over each unique pair of components
            for i in range(n_components):
                for j in range(i + 1, n_components):
                    # Create a new figure and axis
                    fig, ax = plt.subplots()
                    # Scatter plot of component i vs component j
                    sc = ax.scatter(X_test_transformed[:, i], X_test_transformed[:, j], c=actual_angle, cmap='twilight')
                    # Set labels
                    ax.set_xlabel(f'isomap {i + 1}')
                    ax.set_ylabel(f'isomap {j + 1}')
                    # Add a color bar
                    plt.colorbar(sc, ax=ax)
                    plt.savefig(f'{savedir}/isomap_embeddings_fold_{count}_components_{i}_{j}.png', dpi=300,
                                bbox_inches='tight')
                    # plt.show()
                    plt.close('all')

                    #color code by xy position
                    #get distance from origin
                    fig, ax = plt.subplots()
                    # Scatter plot of component i vs component j
                    sc = ax.scatter(X_test_transformed[:, i], X_test_transformed[:, j], c=actual_distance, cmap='viridis')
                    # Set labels
                    ax.set_xlabel(f'isomap {i + 1}')
                    ax.set_ylabel(f'isomap {j + 1}')
                    # Add a color bar
                    plt.colorbar(sc, ax=ax)
                    plt.savefig(
                        f'{savedir}/isomap_embeddings_fold_{count}_colorcodedbydistancefromorigin_components_{i}_{j}.png',
                        dpi=300, bbox_inches='tight')
                    # plt.show()
                    plt.close('all')

            count += 1

            # Calculate the mean training and test scores
        mean_train_score = np.mean(train_scores)
        mean_test_score = np.mean(test_scores)
        print(f'Mean training score: {mean_train_score}')
        print(f'Mean test score: {mean_test_score}')
        fold_dataframe.to_csv(f'{savedir}/fold_results.csv', index=False)
        fold_dataframe_shuffle.to_csv(f'{savedir}/fold_results_shuffle.csv', index=False)

    return best_params, best_score, fold_dataframe, fold_dataframe_shuffle


def main():
    base_dir = 'C:/neural_data/'
    big_result_df = pd.DataFrame()
    big_result_df_shuffle = pd.DataFrame()
    for data_dir in [f'{base_dir}/rat_7/6-12-2019', f'{base_dir}/rat_10/23-11-2021',
                     f'{base_dir}/rat_8/15-10-2019', f'{base_dir}/rat_9/10-12-2021',
                     f'{base_dir}/rat_3/25-3-2019']:


        previous_results, score_dict, num_windows_dict = DataHandler.load_previous_results(
        'randsearch_sanitycheckallvarindepen_isomap_2024-07-')
        rat_id = data_dir.split('/')[-2]
        manual_params_rat = previous_results[rat_id]
        manual_params_rat = manual_params_rat.item()

        spike_dir = os.path.join(data_dir, 'physiology_data')
        dlc_dir = os.path.join(data_dir, 'positional_data')
        labels = np.load(f'{dlc_dir}/labels_250_raw.npy')
        col_list = np.load(f'{dlc_dir}/col_names_250_raw.npy')

        spike_data = np.load(f'{spike_dir}/inputs_10052024_250.npy')

        window_df = pd.read_csv(f'{base_dir}/mean_p_value_vs_window_size_across_rats_grid_250_windows_scale_to_angle_range_False_allo_True.csv')
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
        reducer = Isomap
        reducer_kwargs = {
            'n_components': 3,
            'metric': 'cosine',
            'n_jobs': -1,
        }

        regress = ['x', 'y', 'cos_hd', 'sin_hd']

        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        now_day = datetime.now().strftime("%Y-%m-%d")


        save_dir_path = Path(f'{data_dir}/plot_results/plot_isomap_{now_day}')
        save_dir = save_dir_path
        save_dir.mkdir(parents=True, exist_ok=True)

        best_params, mean_score, result_df, result_df_shuffle = train_and_test_on_isomap_randcv(
            X_for_umap,
            label_df,
            regress,
            regressor,
            regressor_kwargs,
            reducer,
            reducer_kwargs, num_windows = num_windows, savedir=save_dir, manual_params=manual_params_rat
        )
        result_df['rat_id'] = rat_id
        result_df_shuffle['rat_id'] = rat_id

        big_result_df = pd.concat([big_result_df, result_df], axis=0)
        big_result_df_shuffle = pd.concat([big_result_df_shuffle, result_df_shuffle], axis=0)
    big_result_df.to_csv(f'{base_dir}/big_result_df_isomap_250.csv', index=False)
    big_result_df_shuffle.to_csv(f'{base_dir}/big_result_df_shuffle_isomap_250.csv', index=False)





if __name__ == '__main__':
    #
    main()