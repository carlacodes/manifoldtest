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
import os
import scipy
import pickle as pkl
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ['JOBLIB_TEMP_FOLDER'] = 'C:/tmp'

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
        'mse_score_train': np.float32(mse_score_train),
        'r2_score_train': np.float32(r2_score_train),
        'mse_score': np.float32(mse_score),
        'r2_score': np.float32(r2_score_val),
    }

    return results

def create_folds(n_timesteps, num_folds=5, num_windows=10):
    n_windows_total = num_folds * num_windows
    window_size = n_timesteps / n_windows_total
    window_start_ind = np.round(np.arange(0, n_windows_total) * window_size)

    folds = []
    for i in range(num_folds):
        test_windows = np.arange(i, n_windows_total, num_folds)
        test_ind = []
        for j in test_windows:
            test_ind.extend(np.arange(window_start_ind[j], window_start_ind[j] + np.round(window_size)))

        train_ind = list(set(range(n_timesteps)) - set(test_ind))
        test_ind = [int(i) for i in test_ind]

        folds.append((train_ind, test_ind))
        ratio = len(train_ind) / len(test_ind)
        print(f'Ratio of train to test indices is {ratio}')

    return folds

def evaluate_params(spks, y, custom_folds, regressor, regressor_kwargs, reducer, reducer_kwargs, params):
    regressor_kwargs.update({k.replace('estimator__', ''): v for k, v in params.items() if k.startswith('estimator__')})
    reducer_kwargs.update({k.replace('reducer__', ''): v for k, v in params.items() if k.startswith('reducer__')})

    current_regressor = MultiOutputRegressor(regressor(**regressor_kwargs))
    current_reducer = reducer(**reducer_kwargs)

    scores = []
    for train_index, test_index in custom_folds:
        X_train, X_test = spks[train_index], spks[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train = scipy.stats.zscore(X_train, axis=0)
        X_test = scipy.stats.zscore(X_test, axis=0)

        X_train_reduced = current_reducer.fit_transform(X_train)
        X_test_reduced = current_reducer.transform(X_test)

        current_regressor.fit(X_train_reduced, y_train)
        score = current_regressor.score(X_test_reduced, y_test)
        scores.append(score)

    mean_score = np.mean(scores)
    return params, mean_score

def train_and_test_on_umap_randcv(
        spks,
        bhv,
        regress,
        regressor,
        regressor_kwargs,
        reducer,
        reducer_kwargs, num_windows = 1000
):
    param_grid = {
        'estimator__n_neighbors': [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 150, 200],
        'reducer__n_components': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        'reducer__random_state': [42],
        'estimator__metric': ['euclidean', 'cosine', 'minkowski'],
        'reducer__n_neighbors': [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 150, 200],
        'reducer__min_dist': [0.0001, 0.001, 0.01, 0.1, 0.3],
        'reducer__random_state': [42]
    }

    y = bhv[regress].values

    random_search_results = []

    n_timesteps = spks.shape[0]
    custom_folds = create_folds(n_timesteps, num_folds=10, num_windows=num_windows)

    with ProcessPoolExecutor() as executor:
        futures = []
        for _ in range(500):
            params = {key: np.random.choice(values) for key, values in param_grid.items()}
            if 'estimator__metric' in params:
                params['estimator__metric'] = str(params['estimator__metric'])
            futures.append(executor.submit(evaluate_params, spks, y, custom_folds, regressor, regressor_kwargs, reducer, reducer_kwargs, params))

        for future in as_completed(futures):
            params, mean_score = future.result()
            random_search_results.append((params, mean_score))

    best_params, mean_score_max = max(random_search_results, key=lambda x: x[1])
    return best_params, mean_score_max

def main():
    data_dir = '/ceph/scratch/carlag/honeycomb_neural_data/rat_7/6-12-2019/'
    spike_dir = os.path.join(data_dir, 'physiology_data')
    dlc_dir = os.path.join(data_dir, 'positional_data')
    labels = np.load(f'{dlc_dir}/labels_250_raw.npy')
    col_list = np.load(f'{dlc_dir}/col_names_250_raw.npy')

    spike_data = np.load(f'{spike_dir}/inputs_10052024_250.npy')

    window_df = pd.read_csv(f'/ceph/scratch/carlag/honeycomb_neural_data/mean_p_value_vs_window_size_across_rats_grid_250_windows_scale_to_angle_range_False_allo_True.csv')
        #find the rat_id
    rat_id = data_dir.split('/')[-3]
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
    filename = f'params_all_trials_randomizedsearchcv_250bin_1000windows_jake_fold_sinandcos_{now}.npy'
    filename_mean_score = f'mean_score_all_trials_randomizedsearchcv_250bin_1000windows_jake_fold_sinandcos_{now_day}.npy'
    save_dir_path = Path(f'{data_dir}/randsearch_sanitycheckallvarindepen_parallel_{now_day}')
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
    main()
