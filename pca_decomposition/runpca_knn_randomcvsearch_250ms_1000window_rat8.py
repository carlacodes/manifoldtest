# from pathlib import Path
import copy
from datetime import datetime
from sklearn.model_selection import ParameterSampler
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
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


''' Modified from Jules Lebert's code
spks was a numpy arrray of size trial* timebins*neuron, and bhv is  a pandas dataframe where each row represents a trial, the trial is the index '''
import os
import scipy
import pickle as pkl
os.environ['JOBLIB_TEMP_FOLDER'] = 'C:/tmp'

# TODO: 1. change hyperparameters to normalise y = True and kernel = (constant kernel * RBF) + white kernel
# 2. change the regressor to GaussianProcessRegressor
# 3. should the umap X_training data be 2d rather than 3d? Also need to z-score the X input data
# 4. in the 2021 sci advances paper they used 2 fold cross validation
# 5. for the isomap they used n_neighbours = 20 #
# 6. they used the gaussian-filtered (omega = 2-time bins) square root of instantenous firing rates for the isomap decomposition
# 7. bin duration = 512 ms, so about the same as what I have
# 8. target position was smoothed using a gaussian filter



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
    #make sure they are all float32
    mse_score_train = np.float32(mse_score_train)
    r2_score_train = np.float32(r2_score_train)
    mse_score = np.float32(mse_score)
    r2_score_val = np.float32(r2_score_val)


    results = {
        'mse_score_train': mse_score_train,
        'r2_score_train': r2_score_train,
        'mse_score': mse_score,
        'r2_score': r2_score_val,
    }

    return results





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
    # window_test = spks_test[:, w:w + window_size, :].reshape(spks_test.shape[0], -1)
    window_test = spks_test[w:w + window_size, :]
    window_test_y = y_test[w:w + window_size, :]
    # scaler = StandardScaler()
    # # scaler.fit(window_train)
    # window_train = scaler.transform(window_train)
    # window_test = scaler.transform(window_test)
    # print("Before any transformation:", window_train.shape)
    # make into coordinates for umap
    sin_values = window_train_y[:, 0]
    cos_values = window_train_y[:, 1]

    if window_train.shape[0] < window_size:
        print(f'Window train shape is {window_train.shape[0]} and window size is {window_size}')
        return

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

    # combine coord_list into one number per row

    reducer_pipeline.fit(window_train, y=coord_list)
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
        #convert test_ind to int
        test_ind = [int(i) for i in test_ind]

        folds.append((train_ind, test_ind))
        #print the ratio
        ratio = len(train_ind) / len(test_ind)
        print(f'Ratio of train to test indices is {ratio}')

    ############ PLOT FOLDS ##################

    # n_folds = len(folds)
    # # create figure with 10 subplots arranged in 2 rows
    # fig = plt.figure(figsize=(20, 10), dpi=100)

    # for f in range(2):
    #     ax = fig.add_subplot(2, 1, f + 1)
    #     train_index = folds[f][0]
    #     test_index = folds[f][1]

    #     # plot train index as lines from 0 to 1
    #     ax.vlines(train_index, 0, 1, colors='b', linewidth=0.1)
    #     ax.vlines(test_index, 1, 2, colors='r', linewidth=0.1)

    #     ax.set_title(f'Fold {f + 1}')
    #     pass

    return folds



def train_and_test_on_umap_randcv(
        spks,
        bhv,
        regress,
        regressor,
        regressor_kwargs,
        reducer,
        reducer_kwargs,
):
    param_grid = {
        'estimator__n_neighbors': [2, 5, 10, 30, 40, 50, 60, 70],
        'reducer__n_components': [3, 4, 5, 6, 7, 8, 9],
        'estimator__metric': ['euclidean', 'cosine', 'minkowski'],
    }

    y = bhv[regress].values

    random_search_results = []

    # Create your custom folds
    n_timesteps = spks.shape[0]
    custom_folds = create_folds(n_timesteps, num_folds=10, num_windows=1000)
    # Example, you can use your custom folds here

    for _ in range(100):  # 100 iterations for RandomizedSearchCV
        params = {key: np.random.choice(values) for key, values in param_grid.items()}
        regressor_kwargs.update(
            {k.replace('estimator__', ''): v for k, v in params.items() if k.startswith('estimator__')})
        reducer_kwargs.update({k.replace('reducer__', ''): v for k, v in params.items() if k.startswith('reducer__')})

        # Initialize the regressor with current parameters
        current_regressor = MultiOutputRegressor(regressor(**regressor_kwargs))

        # Initialize the reducer with current parameters
        current_reducer = reducer(**reducer_kwargs)

        scores = []
        for train_index, test_index in custom_folds:
            X_train, X_test = spks[train_index], spks[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Apply dimensionality reduction
            X_train_reduced = current_reducer.fit_transform(X_train)
            X_test_reduced = current_reducer.transform(X_test)

            # Fit the regressor
            current_regressor.fit(X_train_reduced, y_train)

            # Evaluate the regressor
            score = current_regressor.score(X_test_reduced, y_test)
            scores.append(score)

        # Calculate mean score for the current parameter combination
        mean_score = np.mean(scores)

        random_search_results.append((params, mean_score))

    # Select the best parameters based on mean score
    best_params, _ = max(random_search_results, key=lambda x: x[1])
    _, mean_score_max = max(random_search_results, key=lambda x: x[1])


    return best_params, mean_score_max

def main():
    data_dir = '/ceph/scratch/carlag/honeycomb_neural_data/rat_8/15-10-2019/'
    spike_dir = os.path.join(data_dir, 'physiology_data')
    dlc_dir = os.path.join(data_dir, 'positional_data')
    labels = np.load(f'{dlc_dir}/labels_1203_with_dist2goal_scale_data_False_zscore_data_False_overlap_False_window_size_250.npy')
    spike_data = np.load(f'{spike_dir}/inputs_overlap_False_window_size_250.npy')

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
    labels_for_umap = scipy.ndimage.gaussian_filter(labels_for_umap, 2, axes=0)

    label_df = pd.DataFrame(labels_for_umap,
                            columns=['x', 'y', 'dist2goal', 'angle_sin', 'angle_cos', 'dlc_angle_zscore'])
    label_df['time_index'] = np.arange(0, label_df.shape[0])

    regressor = KNeighborsRegressor
    regressor_kwargs = {'n_neighbors': 70, 'metric': 'euclidean'}

    reducer = PCA

    reducer_kwargs = {
        'n_components': 3,
    }

    regress = ['angle_sin', 'angle_cos']  # changing to two target variables

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    now_day = datetime.now().strftime("%Y-%m-%d")
    filename = f'params_all_trials_randomizedsearchcv_250bin_1000windows_jake_fold_sinandcos_{now}.npy'
    filename_mean_score = f'mean_score_all_trials_randomizedsearchcv_250bin_1000windows_jake_fold_sinandcos_{now_day}.npy'
    save_dir_path = data_dir_path / 'pca_decomposition'
    save_dir_path.mkdir(parents=True, exist_ok=True)

    best_params, mean_score = train_and_test_on_umap_randcv(
        X_for_umap,
        label_df,
        regress,
        regressor,
        regressor_kwargs,
        reducer,
        reducer_kwargs,
    )
    np.save(save_dir_path / filename, best_params)
    np.save(save_dir_path / filename_mean_score, mean_score)


if __name__ == '__main__':
    #
    main()