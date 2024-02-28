#from pathlib import Path
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor
# from mpl_toolkits import mplot3d
import os
from tqdm import tqdm
from joblib import Parallel, delayed
# from extractlfpandspikedata import load_theta_data
from helpers.load_and_save_data import load_pickle, save_pickle
from helpers.datahandling import DataHandler
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, TimeSeriesSplit, permutation_test_score, GridSearchCV, \
    RandomizedSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.dummy import DummyClassifier
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
from umap import UMAP
from sklearn.model_selection import train_test_split
import umap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# mpl.use('Qt5Agg')  # or can use 'TkAgg', whatever you have/prefer
''' Modified from Jules Lebert's code
spks was a numpy arrray of size trial* timebins*neuron, and bhv is  a pandas dataframe where each row represents a trial, the trial is the index '''
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import ParameterGrid
import os
os.environ['JOBLIB_TEMP_FOLDER'] = 'C:/tmp'


def process_window_within_split(
        w,
        spks_train,
        spks_test,
        window_size,
        y_train,
        y_test,
        reducer_pipeline,
        regressor,
        regressor_kwargs,
):
    reg = regressor(**regressor_kwargs)
    window_train = spks_train[:, w:w + window_size, :].reshape(spks_train.shape[0], -1)
    window_test = spks_test[:, w:w + window_size, :].reshape(spks_test.shape[0], -1)
    # scaler = StandardScaler()
    # # scaler.fit(window_train)
    # window_train = scaler.transform(window_train)
    # window_test = scaler.transform(window_test)
    # print("Before any transformation:", window_train.shape)
    reducer_pipeline.fit(window_train, y=y_train)
    # Transform the reference and non-reference space
    window_ref_reduced = reducer_pipeline.transform(window_train)
    window_nref_reduced = reducer_pipeline.transform(window_test)

    # Fit the classifier on the reference space
    reg.fit(window_ref_reduced, y_train)

    # Predict on the testing data
    y_pred = reg.predict(window_nref_reduced)
    y_pred_train = reg.predict(window_ref_reduced)


    # Compute the mean squared error and R2 score
    mse_score_train = mean_squared_error(y_train, y_pred_train)
    r2_score_train = r2_score(y_train, y_pred_train)

    mse_score = mean_squared_error(y_test, y_pred)
    r2_score_val = r2_score(y_test, y_pred)

    results = {
        'mse_score_train': mse_score_train,
        'r2_score_train': r2_score_train,
        'r2_score_train': r2_score_train,
        'mse_score': mse_score,
        'r2_score': r2_score_val,
        'w': w,
    }

    return results



def train_and_test_on_reduced(
        spks,
        bhv,
        regress,
        regressor,
        regressor_kwargs,
        reducer,
        reducer_kwargs,
        window_size,
        n_jobs_parallel=1,
):
    # Define the grid of hyperparameters
    param_grid = {
        'regressor__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'regressor__C': [0.1, 1, 10],
        'reducer__n_components': [2, 3, 4],
        'reducer__n_neighbors': [10, 20, 30, 40, 50, 60, 70, 80],
        'reducer__min_dist': [0.1, 0.2, 0.3, 0.4, 0.5],
        'reducer__metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski'],
    }

    # Initialize the best hyperparameters and the largest difference
    best_params = None
    largest_diff = float('-inf')
    y = bhv[regress].values

    # Create a TimeSeriesSplit object for 5-fold cross-validation
    tscv = TimeSeriesSplit(n_splits=5)

    # Iterate over all combinations of hyperparameters
    for params in ParameterGrid(param_grid):
        # Update the kwargs with the current parameters
        regressor_kwargs.update({k.replace('regressor__', ''): v for k, v in params.items() if k.startswith('regressor__')})
        reducer_kwargs.update({k.replace('reducer__', ''): v for k, v in params.items() if k.startswith('reducer__')})

        # Initialize lists to store results_cv and permutation_results for each fold
        results_cv_list = []
        permutation_results_list = []
        reducer_pipeline = Pipeline([
            ('reducer', reducer(**reducer_kwargs)),
        ])



        # Perform 5-fold cross-validation
        for train_index, test_index in tscv.split(spks):
            # Split the data into training and testing sets
            X_train, X_test = spks[train_index], spks[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Train the model and compute results_cv


            results_cv = Parallel(n_jobs=n_jobs_parallel, verbose=1)(
                delayed(process_window_within_split)(w, X_train, X_test, window_size, y_train, y_test, reducer_pipeline,
                                                     regressor, regressor_kwargs) for w in
                tqdm(range(spks.shape[1] - window_size)))
            results_cv_list.append(results_cv)

            # Compute permutation_results
            y_train_perm = np.random.permutation(y_train)
            # results_perm = process_window_within_split(
            #     w, X_train, X_test, window_size, y_train_perm, y_test, reducer, regressor, regressor_kwargs
            # )
            results_perm = Parallel(n_jobs=n_jobs_parallel, verbose=1)(
                delayed(process_window_within_split)(w, X_train, X_test, window_size, y_train_perm, y_test, reducer_pipeline,
                                                     regressor, regressor_kwargs) for w in
                tqdm(range(spks.shape[1] - window_size)))
            permutation_results_list.append(results_perm)

        # Calculate the difference between the mean of results_cv and permutation_results
        # diff = np.mean(results_cv_list) - np.mean(permutation_results_list)
        # diff = np.mean([res['mse_score'] for res in results_cv_list]) - np.mean([res['mse_score'] for res in permutation_results_list])
        diff = np.mean([res['mse_score'] for sublist in results_cv_list for res in sublist]) - np.mean([res['mse_score'] for sublist in permutation_results_list for res in sublist])



        # If this difference is larger than the current largest difference, update the best hyperparameters and the largest difference
        if diff > largest_diff:
            largest_diff = diff
            best_params = params

    # After the loop, the best hyperparameters are those that yield the largest difference
    return best_params, largest_diff


def main():
    data_dir = 'C:/neural_data/rat_7/6-12-2019/'
    spike_dir = os.path.join(data_dir, 'physiology_data')
    # spike_trains = load_pickle('spike_trains', spike_dir)
    dlc_dir = os.path.join(data_dir, 'positional_data')

    #load labels
    labels = np.load(f'{dlc_dir}/labels.npy')
    spike_data = np.load(f'{spike_dir}/inputs.npy')
    #find the times where the head angle is stationary
    angle_labels = labels[:, 2]
    stationary_indices = np.where(np.diff(angle_labels) == 0)[0]
    #remove the stationary indices
    labels = np.delete(labels, stationary_indices, axis=0)
    spike_data = np.delete(spike_data, stationary_indices, axis=0)


    param_grid_upper = {
        'bins_before': [6, 7, 8, 10, 20, 30, 50, 100],
        'bin_width': [0.5],
        'window_for_decoding': [0.5, 1, 2, 3, 4, 5, 6],

    }
    largest_diff = float('-inf')
    param_results = {}
    intermediate_results = pd.DataFrame(columns=['difference', 'best_params', 'upper_params'])
    for params in ParameterGrid(param_grid_upper):
        bins_before = params['bins_before']  # How many bins of neural data prior to the output are used for decoding
        bins_current = 1  # Whether to use concurrent time bin of neural data
        bins_after = bins_before  # How many bins of neural data after the output are used for decoding
        X = DataHandler.get_spikes_with_history(spike_data, bins_before, bins_after, bins_current)
        #remove the first six and last six bins
        X_for_umap = X[bins_before:-bins_before]
        labels_for_umap = labels[bins_before:-bins_before]
        labels_for_umap = labels_for_umap[:, 0:3]
        label_df = pd.DataFrame(labels_for_umap, columns=['x', 'y', 'angle'])
        label_df['time_index'] = np.arange(0, label_df.shape[0])
        bin_width = params['bin_width']
        window_for_decoding = params['window_for_decoding']  # in s
        window_size = int(window_for_decoding / bin_width)  # in bins

        regressor = SVR
        regressor_kwargs = {'kernel': 'linear', 'C': 1}


        reducer = UMAP

        reducer_kwargs = {
            'n_components': 3,
            'n_neighbors': 70,
            'min_dist': 0.3,
            'metric': 'euclidean',
            'n_jobs': 10,
        }

        # space_ref = ['No Noise', 'Noise']
        #temporarily remove the space_ref variable, I don't want to incorporate separate data yet
        regress = 'angle'  # Assuming 'head_angle' is the column in your DataFrame for regression

        # Use KFold for regression
        # kf = KFold(n_splits=5, shuffle=True)

        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        now_day = datetime.now().strftime("%Y-%m-%d")
        filename = f'params_{now}.npy'
        filename_intermediate_params = f'intermediate_params_{now_day}.npy'

        if window_size >= X_for_umap.shape[1]:
            print(f'Window size of {window_size} is too large for the number of time bins of {X_for_umap.shape[1]} in the neural data')
            continue


        best_params, diff_result = train_and_test_on_reduced(
            X_for_umap,
            label_df,
            regress,
            regressor,
            regressor_kwargs,
            reducer,
            reducer_kwargs,
            window_size,
            n_jobs_parallel=10,
        )
        #save at intermediate stage of grid search
        intermediate_results = intermediate_results.append({'difference': diff_result, 'best_params': best_params, 'upper_params': params}, ignore_index=True)
        np.save(data_dir / filename_intermediate_params, intermediate_results)

        if diff_result > largest_diff:
            largest_diff = diff_result
            best_params_final = best_params
    param_results['difference'] = diff_result
    param_results['best_params'] = best_params_final
    param_results['upper_params'] = params
    #save to data_dir

    np.save(data_dir / filename, param_results)






if __name__ == '__main__':
    #
    main()