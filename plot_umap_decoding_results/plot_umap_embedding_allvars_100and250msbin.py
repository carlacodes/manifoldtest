# from pathlib import Path
import copy
from datetime import datetime
from sklearn.model_selection import ParameterSampler
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
# from helpers.datahandling import DataHandler
from scipy.stats import randint
from sklearn.decomposition import PCA
import sympy as sp

from sklearn.neighbors import KNeighborsRegressor
from manifold_neural.helpers.datahandling import DataHandler
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
from helpers import tools

''' Modified from Jules Lebert's code
spks was a numpy arrray of size trial* timebins*neuron, and bhv is  a pandas dataframe where each row represents a trial, the trial is the index '''
import os
import scipy
import pickle as pkl
from sklearn.base import BaseEstimator
from ripser import ripser
from ripser import Rips
plt.rcParams.update(plt.rcParamsDefault)

# plt.rcParams['text.usetex'] = False
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


# Define the calculate_torsion function
def calculate_torsion(x, y, z, t):
    dx = sp.diff(x, t)
    dy = sp.diff(y, t)
    dz = sp.diff(z, t)
    ddx = sp.diff(dx, t)
    ddy = sp.diff(dy, t)
    ddz = sp.diff(dz, t)
    numerator = dx*ddy - ddx*dy + dy*ddz - ddy*dz + dz*ddx - ddz*dx
    denominator = (dx**2 + dy**2 + dz**2)**(3/2)
    torsion = numerator / denominator
    return torsion


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
        reducer_kwargs, logger, save_dir_path, use_rand_search=False, manual_params=None, rat_id=None, savedir=None, num_windows =  1000
):


    y = bhv[regress].values

    rat_dataframe = pd.DataFrame()
    rat_dataframe_shuffle = pd.DataFrame()

    random_search_results = []

    # Create your custom folds
    n_timesteps = spks.shape[0]

    custom_folds = create_folds(n_timesteps, num_folds=5, num_windows=num_windows)
    # Example, you can use your custom folds here
    pipeline = Pipeline([
        ('reducer', CustomUMAP()),
        ('estimator', MultiOutputRegressor(regressor()))
    ])

    pipeline_shuffle = Pipeline([
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
        # Define the parameter grid
        # param_grid = {
        #     'estimator__estimator__n_neighbors': (2, 70),  # range of values
        #     'reducer__n_components': (3, 10),
        #     'estimator__estimator__metric': ['euclidean', 'cosine', 'minkowski'],
        #     'reducer__n_neighbors': (2, 70),  # range of values
        #     'reducer__min_dist': (0.0001, 0.3),  # range of values
        #     'reducer__random_state': [42]
        # }
        param_grid = {
           'reducer__n_components': [3, 4, 5, 6, 7, 8, 9, 10],
            'reducer__min_dist': [0.0001, 0.001, 0.01, 0.1, 0.3],
            'reducer__n_neighbors': [10, 20, 30, 40, 50, 60, 70],
            'reducer__random_state': [42],
            'estimator__estimator__n_neighbors': [2, 5, 10, 20, 30, 40, 50, 60, 70],
           'estimator__estimator__metric': ['cosine', 'euclidean', 'minkowski'],}

        # Initialize BayesSearchCV
        random_search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_grid,
            n_iter=1000,
            cv=custom_folds,
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
        best_params = manual_params
        #add cosine metric to the reducer kwargs

        # Initialize lists to store the scores
        train_scores = []
        train_scores_shuffle = []
        test_scores = []
        test_scores_shuffle = []

        # Loop over the custom folds
        count = 0
        fold_dataframe = pd.DataFrame()
        fold_dataframe_shuffle = pd.DataFrame()

        spks_shuffle = copy.deepcopy(spks)
        np.random.shuffle(spks_shuffle)

        y_shuffle = copy.deepcopy(y)
        np.random.shuffle(y_shuffle)


        for train_index, test_index in custom_folds:
            # Split the data into training and testing sets
            spks_train, spks_test = spks[train_index], spks[test_index]
            y_train, y_test = y[train_index], y[test_index]

            spks_train_shuffle, spks_test_shuffle = spks_shuffle[train_index], spks_shuffle[test_index]
            y_train_shuffle, y_test_shuffle = y_shuffle[train_index], y_shuffle[test_index]

            # Set the parameters
            # formatted_params = format_params(manual_params)
            pipeline.set_params(**manual_params)
            pipeline_shuffle.set_params(**manual_params)

            # Fit the pipeline on the training data
            pipeline.fit(spks_train, y_train)
            pipeline_shuffle.fit(spks_train_shuffle, y_train_shuffle)
            fitted_reducer = pipeline.named_steps['reducer']
            fitted_reducer_shuffle = pipeline_shuffle.named_steps['reducer']
            X_test_reduced = fitted_reducer.transform(spks_test)
            X_test_reduced_shuffle = fitted_reducer_shuffle.transform(spks_test_shuffle)

            rips = Rips()
            diagrams = rips.fit_transform(X_test_reduced)
            rips.plot(diagrams, title='Rips Complex for fold: ' + str(count) + '  rat id :' + str(rat_id))
            # plt.title('Rips Diagrams for fold: ' + str(count), 'rat id:' + str(rat_id))
            plt.savefig(f'{savedir}/rips_diagrams_fold_' + str(count) + '.png', dpi=300, bbox_inches='tight')
            plt.show()

            # rips = Rips()
            # diagrams = rips.fit_transform(X_test_reduced)
            #
            # # Plot the H2 diagram
            # rips.plot(diagrams[2], title='Rips Complex H2 for fold: ' + str(count) + '  rat id :' + str(rat_id))
            #
            # # Save the figure
            # plt.savefig(f'{savedir}/rips_diagrams_H2_fold_' + str(count) + '.png', dpi=300, bbox_inches='tight')
            #
            # # Display the figure
            # plt.show()

            # Apply PCA to reduce the dimensionality to 3
            # pca = PCA(n_components=3)
            # X_test_reduced_3d = pca.fit_transform(X_test_reduced)

            # t = sp.symbols('t')
            # x = sp.interpolate(X_test_reduced_3d[:, 0], t)
            # y = sp.interpolate(X_test_reduced_3d[:, 1], t)
            # z = sp.interpolate(X_test_reduced_3d[:, 2], t)
            # Calculate the torsion
            # torsion = calculate_torsion(x, y, z, t)

            # radii = np.linspace(0.1, 1.0, 10)  # Change this to your desired radii
            # diagrams = []
            #
            # for radius in radii:
            #     # Subsample the data
            #     subsample = X_test_reduced[X_test_reduced[:, 0] ** 2 + X_test_reduced[:, 1] ** 2 < radius ** 2]
            #
            #     # Compute the persistence diagram for the subsample if it is not empty
            #     if subsample.size > 0:
            #         diagram = ripser(subsample)['dgms']
            #         diagrams.append(diagram)
            #     else:
            #         print(f"No points found for radius {radius}. Skipping this radius.")
            #
            # fig = plt.figure(figsize=(20, 20))
            #
            # # Plot each diagram in a separate subplot
            # for i, diagram in enumerate(diagrams):
            #     ax = fig.add_subplot(len(radii), 1, i + 1)
            #     Rips().plot(diagram, ax=ax)
            #
            # # Show the figure
            # plt.show()

            # Calculate the training score and append it to the list
            train_score = pipeline.score(spks_train, y_train)
            train_scores_shuffle.append(pipeline_shuffle.score(spks_train_shuffle, y_train_shuffle))
            train_scores.append(train_score)

            # Calculate the test score and append it to the list
            test_score = pipeline.score(spks_test, y_test)
            test_scores_shuffle.append(pipeline_shuffle.score(spks_test_shuffle, y_test_shuffle))

            y_pred = pipeline.predict(spks_test)
            y_pred_shuffle = pipeline_shuffle.predict(spks_test_shuffle)

            col_list = ['x', 'y',  'angle_sin_goal', 'angle_cos_goal']
            indiv_results_dataframe = pd.DataFrame(y_pred, columns=['x', 'y',
                                                                    'angle_sin_goal', 'angle_cos_goal'])
            indiv_results_dataframe_shuffle = pd.DataFrame(y_pred_shuffle, columns=['x', 'y',
                                                                    'angle_sin_goal', 'angle_cos_goal'])

            for i in range(y_test.shape[1]):
                score_indiv = r2_score(y_test[:, i], y_pred[:, i])
                score_indiv_shuffle = r2_score(y_test_shuffle[:, i], y_pred_shuffle[:, i])
                indiv_results_dataframe[col_list[i]] = score_indiv
                indiv_results_dataframe_shuffle[col_list[i]] = score_indiv_shuffle

                print(f'R2 score for {col_list[i]} is {score_indiv}')
            # break down the score into its components
            indiv_results_dataframe['fold'] = count
            indiv_results_dataframe_shuffle['fold'] = count

            fold_dataframe = pd.concat([fold_dataframe, indiv_results_dataframe], axis=0)
            fold_dataframe_shuffle = pd.concat([fold_dataframe_shuffle, indiv_results_dataframe_shuffle], axis=0)

            test_scores.append(test_score)
            actual_angle = np.arcsin(y_test[:, 0])

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            sc = ax.scatter(X_test_reduced[:, 0], X_test_reduced[:, 1], c=actual_angle, cmap='viridis')
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            ax.set_zlabel('UMAP 3')
            # add a color bar
            cbar = plt.colorbar(sc, ax=ax)
            ax.set_title('UMAP test embeddings color-coded by head angle rel. \n  to goal for fold: ' + str(
                count) + 'rat id:' + str(rat_id))
            plt.savefig(f'{savedir}/umap_embeddings_fold_' + str(count) + '.png', dpi=300, bbox_inches='tight')
            plt.show()
            count += 1

        # Calculate the mean training and test scores
        mean_train_score = np.mean(train_scores)
        mean_test_score = np.mean(test_scores)
        mean_train_score_shuffle = np.mean(train_scores_shuffle)
        mean_test_score_shuffle = np.mean(test_scores_shuffle)


        rat_dataframe = pd.concat([rat_dataframe, indiv_results_dataframe], axis=0)
        rat_dataframe['rat_id'] = rat_id
        rat_dataframe['mean_test_score'] = mean_test_score
        rat_dataframe['mean_train_score'] = mean_train_score

        # Print the mean scores
        print(f'Mean training score: {mean_train_score}')
        print(f'Mean test score: {mean_test_score}')

        rat_dataframe_shuffle = pd.concat([rat_dataframe_shuffle, indiv_results_dataframe_shuffle], axis=0)
        rat_dataframe_shuffle['rat_id'] = rat_id
        rat_dataframe_shuffle['mean_test_score'] = mean_test_score_shuffle
        rat_dataframe_shuffle['mean_train_score'] = mean_train_score_shuffle



        best_score = mean_test_score
        #append to a dataframe
    return best_params, best_score,rat_dataframe, rat_dataframe_shuffle


def run_umap_pipeline_across_rats():
    data_dir = 'C:/neural_data/rat_7/6-12-2019/'
    data_dir_list = ['C:/neural_data/rat_7/6-12-2019/','C:/neural_data/rat_10/23-11-2021/', 'C:/neural_data/rat_8/15-10-2019/', 'C:/neural_data/rat_9/10-12-2021/', 'C:/neural_data/rat_3/25-3-2019/']
    across_dir_dataframe = pd.DataFrame()
    across_dir_dataframe_shuffled = pd.DataFrame()
    bin_size = 250
    for data_dir in data_dir_list:
        spike_dir = os.path.join(data_dir, 'physiology_data')
        dlc_dir = os.path.join(data_dir, 'positional_data')
        labels = np.load(f'{dlc_dir}/labels_{bin_size}_scale_to_angle_range_False.npy')
        col_list = np.load(f'{dlc_dir}/col_names_{bin_size}_scale_to_angle_range_False.npy')
        spike_data = np.load(f'{spike_dir}/inputs_10052024_{bin_size}.npy')
        old_spike_data = np.load(f'{spike_dir}/inputs_overlap_False_window_size_{bin_size}.npy')
        #check if they are the same array
        # if np.allclose(spike_data, old_spike_data):
        #     print('The two arrays are the same')
        # else:
        #     print('The two arrays are not the same')


        # print out the first couple of rows of the lfp_data
        #randsearch_allvars_lfadssmooth_empiricalwindows_1000iter_independentvar_2024-05-24
        if bin_size == 100:
            previous_results, score_dict, num_windows_dict = DataHandler.load_previous_results('randsearch_independentvar_lfadssmooth_empiricalwindow_scaled_labels_True_binsize_100_')
        elif bin_size == 250:
            previous_results, score_dict, num_windows_dict = DataHandler.load_previous_results('randsearch_allvars_lfadssmooth_empiricalwindow_zscoredlabels_1000iter_independentvar_smoothaftersplit_v2_2024-06-20')
        rat_id = data_dir.split('/')[-3]
        manual_params = previous_results[rat_id]
        manual_params = manual_params.item()

        num_windows = num_windows_dict[rat_id]

        spike_data_copy = copy.deepcopy(spike_data)
        tolerance = 1e-10  # or any small number that suits your needs
        if np.any(np.abs(np.std(spike_data_copy, axis=0)) < tolerance):
            print('There are neurons with constant firing rates')
            # remove those neurons
            spike_data_copy = spike_data_copy[:, np.abs(np.std(spike_data_copy, axis=0)) >= tolerance]

        X_for_umap_smoothed, removed_indices = tools.apply_lfads_smoothing(spike_data_copy)
        X_for_umap = spike_data_copy
        # X_for_umap = scipy.stats.zscore(X_for_umap, axis=0)
        # X_for_umap_smoothed = scipy.stats.zscore(X_for_umap_smoothed, axis=0)
        # #plot the X_for_umap_smoothed and X_for_umap
        # fig, ax = plt.subplots()
        # ax.plot(X_for_umap_smoothed[:, 10], label='smoothed', alpha=0.5)
        # ax.plot(X_for_umap[removed_indices[-1]:, 10], label='unsmoothed', alpha =0.5)
        # ax.legend()
        # plt.show()


        labels_for_umap = labels
        #remove the indices
        # labels_for_umap = np.delete(labels_for_umap, removed_indices, axis=0)

        label_df = pd.DataFrame(labels_for_umap,
                                columns=col_list)
        label_df['time_index'] = np.arange(0, label_df.shape[0])

        #plot angle sin and angle cos to goal
        fig, ax = plt.subplots()
        ax.plot(label_df['sin_relative_direction'][:120], label = 'angle_sin_goal')
        ax.plot(label_df['cos_relative_direction'][:120], label = 'angle_cos_goal')
        ax.legend()
        plt.show()



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

        regress = ['x', 'y',  'cos_relative_direction', 'sin_relative_direction']  # changing to two target variables

        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        now_day = datetime.now().strftime("%Y-%m-%d")
        filename = f'params_all_trials_randsearch_{bin_size}bin_340windows_jake_fold_allvars_{now}.npy'
        filename_mean_score = f'mean_score_all_trials_randsearch_{bin_size}bin_340windows_jake_fold_{now}.npy'
        save_dir_path = Path(f'{data_dir}/randsearch_allvars_lfadssmooth_340windows_1000iter_independentvar_{now_day}')
        save_dir_path.mkdir(parents=True, exist_ok=True)
        # initalise a logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        # create a file handler
        handler = logging.FileHandler(save_dir_path / f'rand_search_cv_{now}.log')
        handler.setLevel(logging.INFO)
        # create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        # add the handlers to the logger
        logger.addHandler(handler)
        logger.info('Starting the training and testing of the lfp data with the spike data')
        #remove numpy array, just get mapping from manual_params

        best_params, mean_score, rat_dataframe, rat_dataframe_shuffled = train_and_test_on_umap_randcv(
            X_for_umap,
            label_df,
            regress,
            regressor,
            regressor_kwargs,
            reducer,
            reducer_kwargs, logger, save_dir_path, use_rand_search=False, manual_params=manual_params, savedir=save_dir_path, rat_id=rat_id, num_windows = num_windows
        )
        np.save(save_dir_path / filename, best_params)
        np.save(save_dir_path / filename_mean_score, mean_score)
        #append to larger dataframe
        across_dir_dataframe = pd.concat([across_dir_dataframe, rat_dataframe], axis=0)
        across_dir_dataframe_shuffled = pd.concat([across_dir_dataframe_shuffled, rat_dataframe_shuffled], axis=0)
     #save to csv
    across_dir_dataframe['mean_test_score_across_rats'] = across_dir_dataframe['mean_test_score'].mean()
    across_dir_dataframe.to_csv(f'{data_dir}/across_dir_dataframe_bin_size_splitsmooth_{bin_size}.csv')
    across_dir_dataframe_shuffled['mean_test_score_across_rats'] = across_dir_dataframe_shuffled['mean_test_score'].mean()
    across_dir_dataframe_shuffled.to_csv(f'{data_dir}/across_dir_dataframe_shuffled_bin_size_splitsmooth{bin_size}.csv')
    return across_dir_dataframe


if __name__ == '__main__':
    #
    run_umap_pipeline_across_rats()