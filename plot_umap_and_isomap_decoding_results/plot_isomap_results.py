import copy
from datetime import datetime
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from pathlib import Path
from sklearn.metrics import r2_score
from manifold_neural.helpers.datahandling import DataHandler
import pandas as pd
from sklearn.pipeline import Pipeline
import os
from sklearn.model_selection import BaseCrossValidator
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.manifold import Isomap


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

class IsomapTrainer:
    def __init__(self, spks, bhv, regress, regressor, savedir, num_windows, manual_params=None, just_folds=True, null_distribution=False):
        self.spks = spks
        self.bhv = bhv
        self.regress = regress
        self.regressor = regressor
        self.savedir = savedir
        self.num_windows = num_windows
        self.manual_params = manual_params
        self.just_folds = just_folds
        self.null_distribution = null_distribution
        self.custom_folds = self.create_folds(spks.shape[0], num_folds=5, num_windows=num_windows)
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('reducer', Isomap(n_jobs=-1)),
            ('estimator', MultiOutputRegressor(regressor(n_jobs=-1)))
        ])

    def create_folds(self, n_timesteps, num_folds=5, num_windows=10):
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
        return folds

    def train_and_test(self):
        y = self.bhv[self.regress].values
        if self.just_folds:
            self.pipeline.set_params(**self.manual_params)
            if self.null_distribution:
                spks_copy = copy.deepcopy(self.spks)
                while np.array_equal(spks_copy, self.spks):
                    np.random.shuffle(spks_copy)
                assert not np.array_equal(spks_copy, self.spks)
                self.pipeline.fit(spks_copy, y)
                custom_folds_df = pd.DataFrame(self.custom_folds, columns=['train', 'test'])
                custom_folds_df.to_csv(f'{self.savedir}/custom_folds.csv', index=False)
                full_set_transformed = self.pipeline.named_steps['reducer'].transform(
                    self.pipeline.named_steps['scaler'].transform(spks_copy))
                np.save(f'{self.savedir}/full_set_transformed_null.npy', full_set_transformed)
            else:
                self.pipeline.fit(self.spks, y)
                custom_folds_df = pd.DataFrame(self.custom_folds, columns=['train', 'test'])
                custom_folds_df.to_csv(f'{self.savedir}/custom_folds.csv', index=False)
                full_set_transformed = self.pipeline.named_steps['reducer'].transform(
                    self.pipeline.named_steps['scaler'].transform(self.spks))
                np.save(f'{self.savedir}/full_set_transformed.npy', full_set_transformed)
            return None, None, None, None

        train_scores = []
        test_scores = []
        fold_dataframe = pd.DataFrame()
        fold_dataframe_train = pd.DataFrame()
        fold_dataframe_shuffle = pd.DataFrame()
        fold_dataframe_shuffle_train = pd.DataFrame()

        for count, (train_index, test_index) in enumerate(self.custom_folds):
            spks_train, spks_test = self.spks[train_index], self.spks[test_index]
            y_train, y_test = y[train_index], y[test_index]
            y_test_shuffle = copy.deepcopy(y_test)
            np.random.shuffle(y_test_shuffle)
            y_train_shuffle = copy.deepcopy(y_train)
            np.random.shuffle(y_train_shuffle)

            self.pipeline.set_params(**self.manual_params)
            pipeline_shuffle = copy.deepcopy(self.pipeline)
            pipeline_shuffle.set_params(**self.manual_params)

            self.pipeline.fit(spks_train, y_train)
            pipeline_shuffle.fit(spks_train, y_train_shuffle)

            y_pred = self.pipeline.predict(spks_test)
            y_pred_shuffle = pipeline_shuffle.predict(spks_test)
            y_pred_train = self.pipeline.predict(spks_train)
            y_pred_train_shuffle = pipeline_shuffle.predict(spks_train)

            indiv_results_dataframe = pd.DataFrame()
            indiv_results_dataframe_train = pd.DataFrame()
            indiv_results_dataframe_shuffle = pd.DataFrame()
            indiv_results_dataframe_train_shuffle = pd.DataFrame()

            for i in range(y_test.shape[1]):
                score_indiv = r2_score(y_test[:, i], y_pred[:, i])
                score_indiv_shuffle = r2_score(y_test_shuffle[:, i], y_pred_shuffle[:, i])
                indiv_results_dataframe = pd.concat(
                    [indiv_results_dataframe, pd.DataFrame([score_indiv], columns=[self.regress[i]])], axis=1)
                indiv_results_dataframe_shuffle = pd.concat(
                    [indiv_results_dataframe_shuffle, pd.DataFrame([score_indiv_shuffle], columns=[self.regress[i]])],
                    axis=1)

            for j in range(y_train.shape[1]):
                score_indiv = r2_score(y_train[:, j], y_pred_train[:, j])
                score_indiv_shuffle = r2_score(y_train_shuffle[:, j], y_pred_train_shuffle[:, j])
                indiv_results_dataframe_train = pd.concat(
                    [indiv_results_dataframe_train, pd.DataFrame([score_indiv], columns=[self.regress[j]])], axis=1)
                indiv_results_dataframe_train_shuffle = pd.concat(
                    [indiv_results_dataframe_train_shuffle, pd.DataFrame([score_indiv_shuffle], columns=[self.regress[j]])],
                    axis=1)

            indiv_results_dataframe['fold'] = count
            indiv_results_dataframe_shuffle['fold'] = count
            indiv_results_dataframe_train['fold'] = count
            indiv_results_dataframe_train_shuffle['fold'] = count

            fold_dataframe = pd.concat([fold_dataframe, indiv_results_dataframe], axis=0)
            fold_dataframe_shuffle = pd.concat([fold_dataframe_shuffle, indiv_results_dataframe_shuffle], axis=0)
            fold_dataframe_train = pd.concat([fold_dataframe_train, indiv_results_dataframe_train], axis=0)
            fold_dataframe_shuffle_train = pd.concat([fold_dataframe_shuffle_train, indiv_results_dataframe_train_shuffle], axis=0)

            train_score = self.pipeline.score(spks_train, y_train)
            test_score = self.pipeline.score(spks_test, y_test)
            train_scores.append(train_score)
            test_scores.append(test_score)

        mean_train_score = np.mean(train_scores)
        mean_test_score = np.mean(test_scores)
        fold_dataframe.to_csv(f'{self.savedir}/fold_results.csv', index=False)
        fold_dataframe_train.to_csv(f'{self.savedir}/fold_results_train.csv', index=False)
        fold_dataframe_shuffle.to_csv(f'{self.savedir}/fold_results_shuffle.csv', index=False)
        fold_dataframe_shuffle_train.to_csv(f'{self.savedir}/fold_results_train_shuffle.csv', index=False)

        return None, None, fold_dataframe, fold_dataframe_shuffle, fold_dataframe_train, fold_dataframe_shuffle_train

    def save_results(self, big_result_df, big_result_df_train, big_result_df_shuffle, big_result_df_shuffle_train, big_componentresult_df):
        big_result_df.to_csv(f'{self.savedir}/big_result_df_isomap_250.csv', index=False)
        big_result_df_train.to_csv(f'{self.savedir}/big_result_df_train_isomap_250.csv', index=False)
        big_result_df_shuffle.to_csv(f'{self.savedir}/big_result_df_shuffle_isomap_250.csv', index=False)
        big_result_df_shuffle_train.to_csv(f'{self.savedir}/big_result_df_train_shuffle_isomap_250.csv', index=False)
        big_componentresult_df.to_csv(f'{self.savedir}/big_component_invesitgation_result_df_isomap_250.csv', index=False)

def main():
    base_dir = 'C:/neural_data/'
    big_result_df = pd.DataFrame()
    big_result_df_train = pd.DataFrame()
    big_result_df_shuffle = pd.DataFrame()
    big_result_df_shuffle_train = pd.DataFrame()
    big_componentresult_df = pd.DataFrame()
    just_folds = False
    null_distribution = False
    component_investigation = True

    for data_dir in [f'{base_dir}/rat_8/15-10-2019', f'{base_dir}/rat_9/10-12-2021', f'{base_dir}/rat_3/25-3-2019', f'{base_dir}/rat_7/6-12-2019', f'{base_dir}/rat_10/23-11-2021']:
        print(f'Processing {data_dir}')
        previous_results, score_dict, num_windows_dict = DataHandler.load_previous_results('randsearch_sanitycheckallvarindepen_isomap_2024-07-')
        rat_id = data_dir.split('/')[-2]
        manual_params_rat = previous_results[rat_id].item()
        spike_dir = os.path.join(data_dir, 'physiology_data')
        dlc_dir = os.path.join(data_dir, 'positional_data')
        labels = np.load(f'{dlc_dir}/labels_250_raw.npy')
        col_list = np.load(f'{dlc_dir}/col_names_250_raw.npy')
        spike_data = np.load(f'{spike_dir}/inputs_10052024_250.npy')
        window_df = pd.read_csv(f'{base_dir}/mean_p_value_vs_window_size_across_rats_grid_250_windows_scale_to_angle_range_False_allo_True.csv')
        rat_id = data_dir.split('/')[-2]
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
        regress = ['x', 'y', 'cos_hd', 'sin_hd']
        now_day = datetime.now().strftime("%Y-%m-%d")
        save_dir_path = Path(f'{data_dir}/plot_results/plot_test_isomap_{now_day}')
        save_dir = save_dir_path
        save_dir.mkdir(parents=True, exist_ok=True)

        trainer = IsomapTrainer(
            spks=spike_data_copy,
            bhv=label_df,
            regress=regress,
            regressor=regressor,
            savedir=save_dir,
            num_windows=num_windows,
            manual_params=manual_params_rat,
            just_folds=just_folds,
            null_distribution=null_distribution
        )

        if just_folds:
            trainer.train_and_test()
        elif component_investigation:
            results, results_train, results_shuffle, results_shuffle_train = trainer.train_and_test()
            big_result_df = pd.concat([big_result_df, results], axis=0)
            big_result_df_train = pd.concat([big_result_df_train, results_train], axis=0)
            big_result_df_shuffle = pd.concat([big_result_df_shuffle, results_shuffle], axis=0)
            big_result_df_shuffle_train = pd.concat([big_result_df_shuffle_train, results_shuffle_train], axis=0)
        else:
            trainer.train_and_test()

    if not just_folds:
        trainer.save_results(big_result_df, big_result_df_train, big_result_df_shuffle, big_result_df_shuffle_train, big_componentresult_df)

if __name__ == '__main__':
    main()
