#from pathlib import Path
import copy
from datetime import datetime
from sklearn.model_selection import ParameterSampler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.pipeline import Pipeline
from scipy.stats import randint
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
from umap import UMAP
from sklearn.model_selection import train_test_split
import umap
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
import matplotlib.pyplot as plt
os.environ['JOBLIB_TEMP_FOLDER'] = 'C:/tmp'
#TODO: 1. change hyperparameters to normalise y = True and kernel = (constant kernel * RBF) + white kernel
# 2. change the regressor to GaussianProcessRegressor
#3. should the umap X_training data be 2d rather than 3d? Also need to z-score the X input data
#4. in the 2021 sci advances paper they used 2 fold cross validation
#5. for the isomap they used n_neighbours = 20 #
#6. they used the gaussian-filtered (omega = 2-time bins) square root of instantenous firing rates for the isomap decomposition
#7. bin duration = 512 ms, so about the same as what I have
#8. target position was smoothed using a gaussian filter
def process_data_within_split(
        spks_train,
        spks_test,
        y_train,
        y_test,
        reducer_pipeline,
        regressor,
        regressor_kwargs,
):
    base_reg = regressor(**regressor_kwargs)
    reg = MultiOutputRegressor(base_reg)

    reducer_pipeline.fit(spks_train)
    spks_train_reduced = reducer_pipeline.transform(spks_train)
    spks_test_reduced = reducer_pipeline.transform(spks_test)

    reg.fit(spks_train_reduced, y_train)

    y_pred = reg.predict(spks_test_reduced)
    y_pred_train = reg.predict(spks_train_reduced)

    mse_score_train = mean_squared_error(y_train, y_pred_train)
    r2_score_train = r2_score(y_train, y_pred_train)

    mse_score = mean_squared_error(y_test, y_pred)
    r2_score_val = r2_score(y_test, y_pred)

    results = {
        'mse_score_train': mse_score_train,
        'r2_score_train': r2_score_train,
        'mse_score': mse_score,
        'r2_score': r2_score_val,
    }

    return results

def create_folds(n_timesteps, num_folds=5, num_windows=4):
    n_windows_total = num_folds * num_windows
    window_size = n_timesteps // n_windows_total
    window_start_ind = np.arange(0, n_timesteps, window_size)

    folds = []

    for i in range(num_folds):
        test_windows = np.arange(i, n_windows_total, num_folds)
        test_ind = []
        for j in test_windows:
            test_ind.extend(np.arange(window_start_ind[j], window_start_ind[j] + window_size))
        train_ind = list(set(range(n_timesteps)) - set(test_ind))

        folds.append((train_ind, test_ind))

    return folds

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
    base_reg = regressor(**regressor_kwargs)
    reg = MultiOutputRegressor(base_reg)
    # window_train = spks_train[:, w:w + window_size, :].reshape(spks_train.shape[0], -1)
    window_train = spks_train[w:w + window_size, :]
    window_train_y = y_train[w:w + window_size, :]

    #if window_train is less than window_size, then skip
    if window_train.shape[0] < window_size:
        print(f'Window train shape is {window_train.shape[0]} and window size is {window_size}')
        return
    # window_test = spks_test[:, w:w + window_size, :].reshape(spks_test.shape[0], -1)
    window_test = spks_test[w:w + window_size, :]
    window_test_y = y_test[w:w + window_size, :]
    # scaler = StandardScaler()
    # # scaler.fit(window_train)
    # window_train = scaler.transform(window_train)
    # window_test = scaler.transform(window_test)
    # print("Before any transformation:", window_train.shape)
    #make into coordinates for umap
    sin_values = window_train_y[:, 0]
    cos_values = window_train_y[:, 1]

    combined_values = np.array([(sin, cos) for sin, cos in zip(sin_values, cos_values)])
    coord_list = []

    for i in range(len(combined_values)):
        coords = combined_values[i]

        # Map values to the range [0, 2] (assuming values are between -1 and 1)
        mapped_coords = [(val + 1) / 2 for val in coords]

        # Convert mapped coordinates to a single float using a unique multiplier
        unique_float_representation = sum(val * (10 ** (i + 1)) for i, val in enumerate(mapped_coords))
        coord_list.append(unique_float_representation)
    coord_list = np.array(coord_list).reshape(-1, 1)

    #combine coord_list into one number per row

    reducer_pipeline.fit(window_train, y = coord_list)
    # Transform the reference and non-reference space
    window_ref_reduced = reducer_pipeline.transform(window_train)
    window_nref_reduced = reducer_pipeline.transform(window_test)

    # Fit the classifier on the reference space
    reg.fit(window_ref_reduced, window_train_y)

    # Predict on the testing data
    y_pred = reg.predict(window_nref_reduced)
    y_pred_train = reg.predict(window_ref_reduced)


    # Compute the mean squared error and R2 score
    mse_score_train = mean_squared_error(window_train_y, y_pred_train)
    r2_score_train = r2_score(window_train_y, y_pred_train)

    mse_score = mean_squared_error(window_test_y, y_pred)
    r2_score_val = r2_score(window_test_y, y_pred)

    results = {
        'mse_score_train': mse_score_train,
        'r2_score_train': r2_score_train,
        'r2_score_train': r2_score_train,
        'mse_score': mse_score,
        'r2_score': r2_score_val,
        'w': w,
    }

    return results


#
def train_and_test_on_reduced(
        spks,
        bhv,
        regress,
        regressor,
        regressor_kwargs,
        reducer,
        reducer_kwargs,
        window_size,
        n_jobs_parallel=5,
):
    # Define the grid of hyperparameters
    param_grid = {
        # 'regressor__kernel': ['linear'],
        # 'regressor__C': [0.1, 1, 10],
        'regressor__n_neighbors': [70],
        # 'regressor__kernel': [ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(noise_level_bounds=(1e-07, 1.0))],
        'reducer__n_components': [3],
        'reducer__n_neighbors': [20],
        # 'regressor__n_restarts_optimizer': [1],
        # 'reducer__min_dist': [0.01, 0.1, 0.2, 0.3],
        # 'reducer__metric': ['euclidean'],
    }

    # Initialize the best hyperparameters and the largest difference
    best_params = None
    largest_diff = float('-inf')
    y = bhv[regress].values

    # Create a TimeSeriesSplit object for 5-fold cross-validation
    tscv = TimeSeriesSplit(n_splits=2)


    #TODO:check if the n_splits is too big and the sample size is too small?1??!?

    # Iterate over all combinations of hyperparameters
    # for params in ParameterGrid(param_grid):
    n_iter = 1
    for params in ParameterSampler(param_grid, n_iter=n_iter):
        # Update the kwargs with the current parameters
        regressor_kwargs.update({k.replace('regressor__', ''): v for k, v in params.items() if k.startswith('regressor__')})
        reducer_kwargs.update({k.replace('reducer__', ''): v for k, v in params.items() if k.startswith('reducer__')})

        # Initialize lists to store results_cv and permutation_results for each fold
        results_cv_list = []
        permutation_results_list = []
        reducer_pipeline = Pipeline([
            ('reducer', reducer(**reducer_kwargs)),
        ])

        n_timesteps = spks.shape[0]
        folds = create_folds(n_timesteps, num_folds=20, num_windows=19)
        #double check there is no data contamination
        # for train_index, test_index in folds:
        #     if np.intersect1d(train_index, test_index).size > 0:
        #         print('Data contamination')

        # Perform 5-fold cross-validation
        for train_index, test_index in folds:
            # Split the data into training and testing sets
            X_train, X_test = spks[train_index], spks[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Train the model and compute results_cv


            results_cv = process_data_within_split(X_train, X_test, y_train, y_test, reducer_pipeline, regressor,
                                                   regressor_kwargs)

            results_cv_list.append(results_cv)

            # Compute permutation_results
            y_train_perm = copy.deepcopy(y_train)
            X_train_perm = copy.deepcopy(X_train)
            # for i in range(100):
            #     row_indices = np.arange(X_train_perm.shape[0])
            #     np.random.shuffle(row_indices)
            #     X_train_perm = X_train_perm[row_indices]

            #shuffle along the second axis
            # X_train_perm = X_train_perm[:, np.random.permutation(X_train_perm.shape[1]), :]
            y_train_perm = y_train_perm[np.random.permutation(y_train_perm.shape[0])]
            #randomly permute the y_train_perm
            #check if y_train_perm is equal to y_train
            if np.array_equal(y_train_perm, y_train):
                print('y_train_perm is equal to y_train')

            results_perm = process_data_within_split(X_train_perm, X_test, y_train_perm, y_test, reducer_pipeline, regressor,
                                                   regressor_kwargs)

            permutation_results_list.append(results_perm)

        # Calculate the difference between the mean of results_cv and permutation_results
        diff = np.mean([res['mse_score'] for sublist in results_cv_list for res in sublist]) - np.mean([res['mse_score'] for sublist in permutation_results_list for res in sublist])
        print(f'parameters are:{params} and the difference is {diff}')



        # If this difference is larger than the current largest difference, update the best hyperparameters and the largest difference
        if diff > largest_diff:
            largest_diff = diff
            best_params = params

    # After the loop, the best hyperparameters are those that yield the largest difference
    return best_params, largest_diff





def main():
    data_dir = 'C:/neural_data/rat_7/6-12-2019/'
    spike_dir = os.path.join(data_dir, 'physiology_data')
    dlc_dir = os.path.join(data_dir, 'positional_data')

    #load labels
    # labels = np.load(f'{dlc_dir}/labels_0403_with_dist2goal_scale_data_False_zscore_data_False.npy')

    spike_data = pkl.load(open(f'{spike_dir}/spike_array_list_overlap_False.pickle', 'rb'))
    labels = pkl.load(open(f'{dlc_dir}/rearranged_dlc_overlap_False.pickle', 'rb'))
    #extract the 10th trial
    hd_sin_trial = labels['hd_sin'][9]
    hd_cos_trial = labels['hd_cos'][9]
    #stack them
    hd_trial = np.column_stack((hd_sin_trial, hd_cos_trial))
    spike_data_trial = spike_data[9]
    #transpose the spike data
    spike_data_trial = np.transpose(spike_data_trial, (1, 0))

    # #find the times where the head angle is stationary
    # angle_labels = labels[:, 2]
    # stationary_indices = np.where(np.diff(angle_labels) == 0)[0]
    # #remove the stationary indices
    # labels = np.delete(labels, stationary_indices, axis=0)
    # spike_data = np.delete(spike_data, stationary_indices, axis=0)

    data_dir_path = Path(data_dir)

    param_grid_upper = {
        'bin_width': [0.5],
        'window_for_decoding': [100],
    }
    largest_diff = float('-inf')
    param_results = {}
    n_iter = 1
    for params in ParameterSampler(param_grid_upper, n_iter=n_iter):

        #check for neurons with constant firing rates
        tolerance = 1e-10  # or any small number that suits your needs
        if np.any(np.abs(np.std(spike_data_trial, axis=0)) < tolerance):
            print('There are neurons with constant firing rates')
            # remove those neurons
            spike_data_trial = spike_data_trial[:, np.abs(np.std(spike_data_trial, axis=0)) >= tolerance]
        #THEN DO THE Z SCORE
        X_for_umap = scipy.stats.zscore(spike_data_trial, axis=0)

        if np.isnan(X_for_umap).any():
            print('There are nans in the data')



        X_for_umap = scipy.ndimage.gaussian_filter(X_for_umap, 2, axes=0)

        #as a check, plot the firing rates for a single neuron before and after smoothing
        # fig, ax = plt.subplots(1, 2)
        # ax[0].plot(X_for_umap[:, 0])
        # ax[0].set_title('Before smoothing')
        # ax[1].plot(X_for_umap_smooth[ :, 0])
        # ax[1].set_title('After smoothing')
        # plt.show()

        labels_for_umap = hd_trial
        #apply the same gaussian smoothing to the labels
        labels_for_umap = scipy.ndimage.gaussian_filter(labels_for_umap, 2, axes=0)


        label_df = pd.DataFrame(labels_for_umap, columns=['angle_sin', 'angle_cos',])
        # label_df['time_index'] = np.arange(0, label_df.shape[0])
        bin_width = params['bin_width']
        window_for_decoding = params['window_for_decoding']  # in s
        window_size = int(window_for_decoding / bin_width)  # in bins

        regressor = KNeighborsRegressor

        # regressor_kwargs = {'kernel': 'linear', 'C': 1}
        regressor_kwargs = {'n_neighbors': 70}

        reducer = UMAP

        reducer_kwargs = {
            'n_components': 3,
            # 'n_neighbors': 70,
            # 'min_dist': 0.3,
            'metric': 'euclidean',
            'n_jobs': 1,
        }

        # space_ref = ['No Noise', 'Noise']
        #temporarily remove the space_ref variable, I don't want to incorporate separate data yet
        regress = ['angle_sin', 'angle_cos']# changing to two target variables


        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        now_day = datetime.now().strftime("%Y-%m-%d")
        filename = f'params_trial9_sinandcos_{now}.npy'
        filename_intermediate_params = f'intermediate_params_trial9_sin_and_cos_v2_{now_day}.npy'

        if window_size >= X_for_umap.shape[0]:
            print(f'Window size of {window_size} is too large for the number of time bins of {X_for_umap.shape[0]} in the neural data')
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
            n_jobs_parallel=1,
        )
        #save at intermediate stage of grid search
        # intermediate_results = intermediate_results.append({'difference': diff_result, 'best_params': best_params, 'upper_params': params}, ignore_index=True)
        intermediate_results = pd.DataFrame(
            {'difference': [diff_result], 'best_params': [best_params], 'upper_params': [params]})
        np.save(data_dir_path / filename_intermediate_params, intermediate_results)


        if diff_result > largest_diff:
            largest_diff = diff_result
            best_params_final = best_params
    param_results['difference'] = diff_result
    param_results['best_params'] = best_params_final
    param_results['upper_params'] = params
    #save to data_dir

    np.save(data_dir_path/ filename, param_results)






if __name__ == '__main__':
    #
    main()