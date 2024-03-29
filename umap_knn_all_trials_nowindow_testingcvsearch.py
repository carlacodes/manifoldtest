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
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from umap import UMAP
import seaborn as sns
from scipy.stats import ttest_ind
from numpy import mean, std, var, sqrt


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
def passthrough_func(X):
    return X
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

    n_folds = len(folds)
    # create figure with 10 subplots arranged in 2 rows
    fig = plt.figure(figsize=(20, 10), dpi=100)

    for f in range(2):
        ax = fig.add_subplot(2, 1, f + 1)
        train_index = folds[f][0]
        test_index = folds[f][1]

        # plot train index as lines from 0 to 1
        ax.vlines(train_index, 0, 1, colors='b', linewidth=0.1)
        ax.vlines(test_index, 1, 2, colors='r', linewidth=0.1)

        ax.set_title(f'Fold {f + 1}')
        pass

    return folds
def create_folds_do_not_use(n_timesteps, num_folds=5, num_windows=3):
    n_windows_total = num_folds * num_windows
    window_size = n_timesteps // n_windows_total
    window_start_ind = np.arange(0, n_timesteps, window_size)

    folds = []

    for i in range(num_folds):
        # Uniformly select test windows from the total windows
        step_size = n_windows_total // num_windows
        step_size_train = n_windows_total // (num_windows - 1)
        test_windows = np.arange(i, n_windows_total, step_size)
        test_ind = []
        #make sure the train and test indices are equivalent in length

        for j in test_windows:
            # Select every nth index for testing, where n is the step size
            test_ind.extend(np.arange(window_start_ind[j], window_start_ind[j] + window_size, step_size))
        # for j2 in range(n_windows_total):
        #     if j2 not in test_windows:
        #         #ensure the
        #         train_ind = np.arange(window_start_ind[j2], window_start_ind[j2] + window_size)
        train_ind = list(set(range(n_timesteps)) - set(test_ind))


        folds.append((train_ind, test_ind))
        ratio =  len(test_ind) / len(train_ind)
        print(f'Ratio of train to test indices is {ratio}')


    # As a sanity check, plot the distribution of the test indices
    fig, ax = plt.subplots()
    ax.hist(train_ind, label='train')
    ax.hist(test_ind, label='test')
    ax.legend()
    plt.show()

    return folds
def create_folds_old(n_timesteps, num_folds=10, num_windows=200):
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
        #adjust the train indices so it doesnt oversample from the end of the distribution
        train_ind = train_ind[0:len(test_ind)]
        #add the leftover indices to the test indices
        test_ind.extend(list(set(range(n_timesteps)) - set(train_ind)))

        folds.append((train_ind, test_ind))
        #check the ratio of train_ind to test_ind
        ratio = len(train_ind) / len(test_ind)
    #as a sanity check, plot the distribution of the test indices
    return folds

def create_sliding_window_folds(n_timesteps, num_folds=10):
    window_size = n_timesteps // num_folds
    folds = []

    for i in range(num_folds):
        start = i * window_size
        end = start + window_size

        if end > n_timesteps:
            break

        train_ind = list(range(start))
        test_ind = list(range(start, end))

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

def cohend(d1, d2):
    # calculate the size of samples
    n1, n2 = len(d1), len(d2)
    # calculate the variance of the samples
    s1, s2 = var(d1, ddof=1), var(d2, ddof=1)
    # calculate the pooled standard deviation
    s = sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    # calculate the means of the samples
    u1, u2 = mean(d1), mean(d2)
    # calculate the effect size
    return (u1 - u2) / s

def train_and_test_on_umap_randcv(
        spks,
        bhv,
        regress,
        regressor,
        regressor_kwargs,
        reducer,
        reducer_kwargs,
):
    # param_grid = {
    #     'estimator__n_neighbors': [2, 5, 10, 30, 40, 50, 60, 70],
    #     'reducer__n_components': [3, 4, 5, 6, 7, 8, 9],
    #     'estimator__metric': ['euclidean', 'cosine', 'minkowski'],
    #     'reducer__n_neighbors': [20, 30, 40, 50, 60, 70],
    #     'reducer__min_dist': [0.001, 0.01, 0.1, 0.3],
    # }

    param_grid = {'estimator__n_neighbors': [70], 'reducer__n_components': [5], 'estimator__metric': ['cosine'],
     'reducer__n_neighbors': [60], 'reducer__min_dist': [0.01]}
    # {'estimator__n_neighbors': 70, 'reducer__n_components': 5, 'estimator__metric': 'cosine',
    #  'reducer__n_neighbors': 60, 'reducer__min_dist': 0.01}

    y = bhv[regress].values

    random_search_results = []

    # Create your custom folds
    n_timesteps = spks.shape[0]
    custom_folds = create_folds(n_timesteps, num_folds=10, num_windows=200)

    # custom_folds = create_sliding_window_folds(n_timesteps, num_folds=10)
    # Example, you can use your custom folds here
    count = 0
    for train_index, test_index in custom_folds:
        if len(set(train_index).intersection(set(test_index))) > 0:
            print('There is overlap between the train and test indices')
        # plot the head angle for the train and test indices
        # fig, ax = plt.subplots()
        # ax.plot(bhv['angle_cos'].values[train_index], label = 'train')
        # ax.plot(bhv['angle_cos'].values[test_index], label = 'test')
        # ax.set_title(f'Head angle cosine values for train and test indices, fold number: {count}')
        # plt.legend()
        # plt.show()
        # fig, ax = plt.subplots()
        # ax.plot(bhv['angle_sin'].values[train_index], label = 'train')
        # ax.plot(bhv['angle_sin'].values[test_index], label = 'test')
        # ax.set_title(f'Head angle sine values for train and test indices, fold number: {count}')
        # plt.legend()
        # plt.show()
        # #plot just the test data
        # fig, ax = plt.subplots()
        # ax.plot(bhv['angle_cos'].values[test_index], label = 'test')
        # ax.set_title(f'Head angle cosine values for test indices, fold number: {count}')
        # plt.legend()
        # plt.show()
        # fig, ax = plt.subplots()
        # ax.plot(bhv['angle_sin'].values[test_index], label = 'test')
        # ax.set_title(f'Head angle sine values for test indices, fold number: {count}')
        # plt.legend()
        # plt.show()

        #plot how the indexes are distributed
        fig, ax = plt.subplots()
        sns.distplot(train_index, label = 'train', ax = ax)
        sns.distplot(test_index, label = 'test', ax = ax)
        #run a t-test to see if the distributions are different

        t_stat, p_val = ttest_ind(train_index, test_index)
        p_val_rounded = round(p_val, 3)
        print(f'The t-statistic is {t_stat} and the p-value is {p_val}')
        #effect size calculation
        from numpy import mean, std, var
        from math import sqrt
        # function to calculate Cohen's d for independent samples

        # calculate cohen's d
        d = cohend(train_index, test_index)

        ax.set_title(f'Distribution of train and test indices, fold number: {count}, \n'
                     f' p-value: {p_val_rounded}, effect size: {d}')
        plt.legend()
        plt.show()

        #plot the distribution of the actual test and the train data

        #calculate the t statistic and p value for the head angle cosine values
        t_stat, p_val = ttest_ind(bhv['angle_cos'].values[train_index], bhv['angle_cos'].values[test_index])
        p_val_rounded = round(p_val, 3)

        fig, ax = plt.subplots()
        sns.distplot(bhv['angle_cos'].values[train_index], label = 'train', ax = ax)
        sns.distplot(bhv['angle_cos'].values[test_index], label = 'test', ax = ax)
        ax.set_title(f'Distribution of head angle cosine values for train and test indices, fold number: {count}, \n p-value: {p_val_rounded}')
        plt.legend()
        plt.show()

        t_stat, p_val = ttest_ind(bhv['angle_sin'].values[train_index], bhv['angle_sin'].values[test_index])
        p_val_rounded_sin = round(p_val, 3)

        fig, ax = plt.subplots(
        )
        sns.distplot(bhv['angle_sin'].values[train_index], label = 'train', ax = ax)
        sns.distplot(bhv['angle_sin'].values[test_index], label = 'test', ax = ax)
        ax.set_title(f'Distribution of head angle sine values for train and test indices, fold number: {count}, \n p-value: {p_val_rounded_sin}')
        plt.legend()
        plt.show()


        #effect size calculation
        d = cohend(bhv['angle_cos'].values[train_index], bhv['angle_cos'].values[test_index])
        print(f'The t-statistic is {t_stat} and the p-value is {p_val}')
        print(f'The effect size is {d}')

        count += 1

    for _ in range(1):  # 100 iterations for RandomizedSearchCV
        params = {key: np.random.choice(values) for key, values in param_grid.items()}

        # Initialize the regressor with current parameters
        current_regressor = MultiOutputRegressor(regressor(**regressor_kwargs))

        # Initialize the reducer with current parameters
        current_reducer = reducer(**reducer_kwargs)

        scores = []
        scores_train = []
        for train_index, test_index in custom_folds:
            X_train, X_test = spks[train_index], spks[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Apply dimensionality reduction
            X_train_reduced = current_reducer.fit_transform(X_train)
            X_test_reduced = current_reducer.transform(X_test)

            # Fit the regressor
            current_regressor.fit(X_train_reduced, y_train)

            # Evaluate the regressor: using the default for regressors which is r2
            score = current_regressor.score(X_test_reduced, y_test)
            score_train = current_regressor.score(X_train_reduced, y_train)
            scores.append(score)
            scores_train.append(score_train)

        # Calculate mean score for the current parameter combination
        mean_score = np.mean(scores)
        mean_score_train = np.mean(scores_train)

        random_search_results.append((params, mean_score))

    # Select the best parameters based on mean score
    best_params, _ = max(random_search_results, key=lambda x: x[1])

    return best_params, mean_score


def load_previous_results(data_dir):
    # previous_results = np.load(f'{data_dir}/results_cv_2024-03-21_12-31-37.npy', allow_pickle=True)
    previous_best_params = np.load(f'{data_dir}/cluster_results/params_all_trials_randomizedsearchcv_200windows_jake_fold_sinandcos_2024-03-26_16-09-16.npy', allow_pickle=True)
    # previous_perm_results = np.load(f'{data_dir}/perm_results_list_2024-03-21_12-31-37.npy', allow_pickle=True)
    print(previous_best_params)
    return previous_best_params

def main():
    data_dir = 'C:/neural_data/rat_7/6-12-2019/'
    prev_best_params = load_previous_results(data_dir)
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
    regressor_kwargs = {'n_neighbors': 70}

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
    filename = f'params_all_trials_randomizedsearchcv_jake_fold_sinandcos_{now}.npy'
    filename_score = f'results_all_trials_randomizedsearchcv_{now}.npy'


    best_params, mean_score = train_and_test_on_umap_randcv(
        X_for_umap,
        label_df,
        regress,
        regressor,
        regressor_kwargs,
        reducer,
        reducer_kwargs,
    )
    y = label_df[regress].values

    np.save(data_dir_path / filename, best_params)
    np.save(data_dir_path / filename_score, mean_score)


if __name__ == '__main__':
    #
    main()