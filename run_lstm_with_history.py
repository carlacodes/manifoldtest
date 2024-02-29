import copy
from pathlib import Path
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
from sklearn.metrics import mean_squared_error, r2_score
from umap import UMAP
from sklearn.model_selection import train_test_split
import umap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# mpl.use('Qt5Agg')  # or can use 'TkAgg', whatever you have/prefer
''' Modified from Jules Lebert's code
spks is a numpy arrray of size trial* timebins*neuron, and bhv is  a pandas dataframe where each row represents a trial, the trial is the index '''
from sklearn.decomposition import PCA


def process_window(
        w,
        spks,
        window_size,
        y,
        reducer_pipeline,
        regressor,
        regressor_kwargs,
):
    reg = regressor(**regressor_kwargs)

    window = spks[:, w:w + window_size, :].reshape(spks.shape[0], -1)

    # Split the data into training and testing sets

    window_train, window_test, y_train, y_test = train_test_split(window, y, test_size=0.2, shuffle=False)
    # y_train = np.ravel(y_train)
    # y_test = np.ravel(y_test)
    # Fit the reducer on the training data
    scaler = StandardScaler()
    scaler.fit(window_train)
    window_train = scaler.transform(window_train)
    window_test = scaler.transform(window_test)
    # print("Before any transformation:", window_train.shape)
    reducer_pipeline.fit(window_train, y=y_train)
    # print("After pipeline transformation:", window_train.shape)

    # Transform the training and testing data
    window_train_reduced = reducer_pipeline.transform(window_train)
    window_test_reduced = reducer_pipeline.transform(window_test)

    # Fit the regressor on the training data
    reg.fit(window_train_reduced, y_train)

    # Predict on the testing data
    y_pred = reg.predict(window_test_reduced)

    # Compute the mean squared error and R2 score
    mse_score = mean_squared_error(y_test, y_pred)
    r2_score_val = r2_score(y_test, y_pred)

    results = {
        'mse_score': mse_score,
        'r2_score': r2_score_val,
        'w': w,
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

def train_ref_classify_rest(
        spks,
        bhv,
        regress,
        regressor,
        regressor_kwargs,
        reducer,
        reducer_kwargs,
        window_size,
        n_permutations=100,
        n_jobs_parallel=1,
):
    """
    Analyzes spike data using dimensionality reduction and regression.

    Parameters:
    - spks: The spike data.
    - bhv: Behavioral data containing masks and labels.
    - regress: Column name in bhv to use for regression labels.
    - regressor: Regressor to use.
    - regressor_kwargs: Keyword arguments for the regressor.
    - reducer: Dimensionality reduction method to use.
    - reducer_kwargs: Keyword arguments for the reducer.
    - window_size: Size of the window to use for analysis.

    Returns:
    - Dictionary containing the mean squared error and R2 scores.
    """
    # Z-score with respect to reference space
    spks_mean = np.nanmean(spks, axis=0)
    spks_std = np.nanstd(spks, axis=0)
    spks_std[spks_std == 0] = np.finfo(float).eps
    spks = (spks - spks_mean) / spks_std
    scaler = StandardScaler()
    spks_scaled = scaler.fit_transform(spks.reshape(spks.shape[0], -1))

    reducer_pipeline = Pipeline([
        # ('scaler', StandardScaler()),
        ('reducer', reducer(**reducer_kwargs)),
    ])

    y = bhv[regress].values

    # results_cv = Parallel(n_jobs=n_jobs_parallel, verbose=1, prefer="threads")(
    #     delayed(process_window)(w, spks, window_size, y, reducer_pipeline, regressor,
    #                             regressor_kwargs) for w in tqdm(range(spks.shape[1] - window_size)))
    results_perm = []
    if n_permutations > 0:
        for n in tqdm(range(n_permutations)):
            y_perm = np.random.permutation(y)
            # shift = np.random.randint(spks.shape[1])
            # spks_perm = np.roll(spks, shift, axis=1)
            spks_perm = np.random.permutation(spks)
            # fig, ax = plt.subplots()
            # #plot y_perm versus y
            # x_axis = np.arange(0, 20)
            # plt.plot(x_axis, y_perm[:20], label='y_perm', color='red')
            # plt.plot(x_axis, y[:20], label='y', color='blue')
            # plt.show()
            # # offset = 2 * np.pi * np.random.random()
            # # y_perm = np.random.permutation((y + offset) % (2 * np.pi))


            # Initialize the Support Vector Regressor (SVR)
            reg = SVR(kernel='rbf', C=1)
            results_perm_n = []
            for w in tqdm(range(spks.shape[1] - window_size)):
                window = spks_perm[:, w:w + window_size, :].reshape(spks.shape[0], -1)
                window_train, window_test, y_train, y_test = train_test_split(window, y_perm, test_size=0.2, shuffle=False)

                # Fit the regressor on the reference space
                reg.fit(window_train, y_train)

                # Predict on the non-reference space
                y_pred = reg.predict(window_test)

                # Compute the mean squared error and R2 score
                mse_score = mean_squared_error(y_test, y_pred)
                r2_score_val = r2_score(y_test, y_pred)

                results = {
                    'mse_score': mse_score,
                    'r2_score': r2_score_val,
                    'w': w,
                }

                results_perm_n.append(results)
            results_perm.append(results_perm_n)

    results = {
        # 'cv': results_cv,
        'perm': results_perm,
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
        n_permutations=100,
        n_jobs_parallel=1,
):


    reducer_pipeline = Pipeline([
        # ('scaler', StandardScaler()),
        ('reducer', reducer(**reducer_kwargs)),
    ])
    y = bhv[regress].values


    cv_results = []
    skf = TimeSeriesSplit(n_splits=5)

    for i, (train_idx, test_idx) in enumerate(skf.split(spks, y)):
        print(f'Fold {i} / {skf.get_n_splits()}')
        X_train = spks[train_idx, :, :]
        X_test = spks[test_idx, :, :]
        y_train = y[train_idx]
        y_test = y[test_idx]

        # Z-score to train data
        spks_mean = np.nanmean(X_train, axis=0)

        spks_std = np.nanstd(X_train, axis=0)
        spks_std[spks_std == 0] = np.finfo(float).eps



        X_train = (X_train - spks_mean) / spks_std
        X_test = (X_test - spks_mean) / spks_std

        # results = Parallel(n_jobs=n_jobs_parallel, verbose=1)(
        #     delayed(process_window_within_split)(w, X_train, X_test, window_size, y_train, y_test, reducer_pipeline, regressor,
        #                                          regressor_kwargs) for w in tqdm(range(spks.shape[1] - window_size)))
        #possible redundnacy in Jules' original code as he normalises the data and then ALSO applies a Scaler, seems a bit redundant
        results = Parallel(n_jobs=n_jobs_parallel, backend='loky', verbose=1)(
            delayed(process_window_within_split)(
                w, X_train, X_test, window_size, y_train, y_test, reducer_pipeline, regressor, regressor_kwargs
            ) for w in tqdm(range(spks.shape[1] - window_size))
        )

        cv_results.append(results)

    results_perm = []
    if n_permutations > 0:
        for n in range(n_permutations):
            print(f'Permutation {n} / {n_permutations}')
            y_perm = copy.deepcopy(y)
            spks_perm = copy.deepcopy(spks)
            for i, (train_idx, test_idx) in enumerate(skf.split(spks_perm, y_perm)):
                print(f'Fold {i} / {skf.get_n_splits()}')
                X_train = spks_perm[train_idx, :, :]
                X_test = spks_perm[test_idx, :, :]
                y_train = y_perm[train_idx]
                y_train = np.random.permutation(y_train)
                y_test = y_perm[test_idx]

                # Z-score to train data
                spks_mean = np.nanmean(X_train, axis=0)
                spks_std = np.nanstd(X_train, axis=0)
                spks_std[spks_std == 0] = np.finfo(float).eps


                X_train = (X_train - spks_mean) / spks_std
                X_test = (X_test - spks_mean) / spks_std

                results = Parallel(n_jobs=n_jobs_parallel, verbose=1)(
                    delayed(process_window_within_split)(w, X_train, X_test, window_size, y_train, y_test, reducer_pipeline,
                                            regressor, regressor_kwargs) for w in
                    tqdm(range(spks.shape[1] - window_size)))

                results_perm.append(results)

    results = {
        'cv': cv_results,
        'perm': results_perm,
    }
    return results


def run_umap_with_history(data_dir):

    spike_dir = os.path.join(data_dir, 'physiology_data')
    # spike_trains = load_pickle('spike_trains', spike_dir)
    dlc_dir = os.path.join(data_dir, 'positional_data')

    # load labels
    labels = np.load(f'{dlc_dir}/labels_2902.npy')
    spike_data = np.load(f'{spike_dir}/inputs.npy')

    bins_before = 6  # How many bins of neural data prior to the output are used for decoding
    bins_current = 1  # Whether to use concurrent time bin of neural data
    bins_after = 6  # How many bins of neural data after the output are used for decoding
    X = DataHandler.get_spikes_with_history(spike_data, bins_before, bins_after, bins_current)
    # remove the first six and last six bins
    X_for_umap = X[6:-6]
    labels_for_umap = labels[6:-6]
    labels_for_umap = labels_for_umap[:, 0:5]
    # labels_for_umap = labels[:, 0:3]
    label_df = pd.DataFrame(labels_for_umap, columns=['x', 'y', 'angle_sin', 'angle_cos', 'dlc_angle_norm'])
    label_df['time_index'] = np.arange(0, label_df.shape[0])
    # unsupervised_umap(X_for_umap, label_df, remove_low_variance_neurons=False, n_components=3)

    bin_width = 0.5
    window_for_decoding = 6  # in s
    window_size = int(window_for_decoding / bin_width)  # in bins

    n_runs = 1

    regressor = SVR
    regressor_kwargs = {'kernel': 'linear', 'C': 1}

    reducer = UMAP

    reducer_kwargs = {
        'n_components': 3,
        'n_neighbors': 70,
        'min_dist': 0.3,
        'metric': 'euclidean',
        'n_jobs': 1,
    }

    # space_ref = ['No Noise', 'Noise']
    # temporarily remove the space_ref variable, I don't want to incorporate separate data yet
    regress = 'dlc_angle_norm'  # Assuming 'head_angle' is the column in your DataFrame for regression

    # Use KFold for regression
    # kf = KFold(n_splits=5, shuffle=True)

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f'results_{now}.npy'
    results_between = {}
    results_within = {}
    results_w_perm_reduced = {}
    n_permutations = 5
    for run in range(n_runs):
        results_between[run] = {}
        results_within[run] = {}
        # for space in space_ref:

        # results_between[run] = train_ref_classify_rest(
        #     X_for_umap,
        #     label_df,
        #     regress,
        #     regressor,
        #     regressor_kwargs,
        #     reducer,
        #     reducer_kwargs,
        #     window_size,
        #     n_permutations=n_permutations,
        # )
        results_w_perm_reduced[run] = train_and_test_on_reduced(
            X_for_umap,
            label_df,
            regress,
            regressor,
            regressor_kwargs,
            reducer,
            reducer_kwargs,
            window_size,
            n_permutations=n_permutations, n_jobs_parallel=5
        )

        # Save results
    results = {'between': results_between, 'within': results_w_perm_reduced}
    save_path = Path('C:/neural_data/rat_7/6-12-2019')
    save_path.mkdir(exist_ok=True)
    np.save(save_path / filename, results)
def main():
    dir = 'C:/neural_data/rat_7/6-12-2019/'
    run_umap_with_history(dir)
    return



if __name__ == '__main__':
    main()