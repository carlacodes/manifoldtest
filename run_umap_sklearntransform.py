# from pathlib import Path
import copy
from datetime import datetime
from sklearn.model_selection import ParameterSampler
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt
# from helpers.datahandling import DataHandler
from scipy.stats import randint
from sklearn.neighbors import KNeighborsRegressor
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
from umap import UMAP
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import logging
import sys
from helpers import tools

''' Modified from Jules Lebert's code
spks was a numpy arrray of size trial* timebins*neuron, and bhv is  a pandas dataframe where each row represents a trial, the trial is the index '''
import os
import scipy
import pickle as pkl
from sklearn.base import BaseEstimator, TransformerMixin
import scipy.stats
import numpy as np

from sklearn import pipeline
from sklearn.base import clone
from sklearn.utils import _print_elapsed_time
from sklearn.utils.validation import check_memory


class Pipeline(pipeline.Pipeline):

    # def _fit(self, X, y=None, **fit_params_steps):
    #     self.steps = list(self.steps)
    #     self._validate_steps()
    #     memory = check_memory(self.memory)
    #
    #     fit_transform_one_cached = memory.cache(pipeline._fit_transform_one)
    #
    #     for (step_idx, name, transformer) in self._iter(
    #             with_final=False, filter_passthrough=False
    #     ):
    #
    #         if transformer is None or transformer == "passthrough":
    #             with _print_elapsed_time("Pipeline", self._log_message(step_idx)):
    #                 continue
    #
    #         try:
    #             # joblib >= 0.12
    #             mem = memory.location
    #         except AttributeError:
    #             mem = memory.cachedir
    #         finally:
    #             cloned_transformer = clone(transformer) if mem else transformer
    #
    #         X, fitted_transformer = fit_transform_one_cached(
    #             cloned_transformer,
    #             X,
    #             y,
    #             None,
    #             message_clsname="Pipeline",
    #             message=self._log_message(step_idx),
    #             **fit_params_steps[name],
    #         )
    #
    #         if isinstance(X, tuple):  ###### unpack X if is tuple: X = (X,y)
    #             X, y = X
    #
    #         self.steps[step_idx] = (name, fitted_transformer)
    #
    #     return X, y


    def _fit(self, X, y=None, routed_params=None):
        # shallow copy of steps - this should really be steps_
        self.steps = list(self.steps)
        self._validate_steps()
        # Setup the memory
        memory = check_memory(self.memory)

        fit_transform_one_cached = memory.cache(pipeline._fit_transform_one)

        for step_idx, name, transformer in self._iter(
            with_final=False, filter_passthrough=False
        ):
            if transformer is None or transformer == "passthrough":
                with _print_elapsed_time("Pipeline", self._log_message(step_idx)):
                    continue

            if hasattr(memory, "location") and memory.location is None:
                # we do not clone when caching is disabled to
                # preserve backward compatibility
                cloned_transformer = transformer
            else:
                cloned_transformer = clone(transformer)
            # Fit or load from cache the current transformer
            X, fitted_transformer = fit_transform_one_cached(
                cloned_transformer,
                X,
                y,
                None,
                message_clsname="Pipeline",
                message=self._log_message(step_idx),
                params=routed_params[name],
            )
            # Replace the transformer of the step with the fitted
            # transformer. This is necessary when loading the transformer
            # from the cache.
            if isinstance(X, tuple):
                X, y = X
            self.steps[step_idx] = (name, fitted_transformer)


        return X, y
    def score(self, X, y=None, sample_weight=None, **params):
        """Transform the data, and apply `score` with the final estimator.

        Call `transform` of each transformer in the pipeline. The transformed
        data are finally passed to the final estimator that calls
        `score` method. Only valid if the final estimator implements `score`.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.

        y : iterable, default=None
            Targets used for scoring. Must fulfill label requirements for all
            steps of the pipeline.

        sample_weight : array-like, default=None
            If not None, this argument is passed as ``sample_weight`` keyword
            argument to the ``score`` method of the final estimator.

        **params : dict of str -> object
            Parameters requested and accepted by steps. Each step must have
            requested certain metadata for these parameters to be forwarded to
            them.

            .. versionadded:: 1.4
                Only available if `enable_metadata_routing=True`. See
                :ref:`Metadata Routing User Guide <metadata_routing>` for more
                details.

        Returns
        -------
        score : float
            Result of calling `score` on the final estimator.
        """
        Xt = X
        if not pipeline._routing_enabled():
            for _, name, transform in self._iter(with_final=False):
                Xt = transform.transform(Xt)
            score_params = {}
            if sample_weight is not None:
                score_params["sample_weight"] = sample_weight
            if len(Xt) < len(y):
                #get the difference
                diff = len(y) - len(Xt)
                y = y[diff:]
            return self.steps[-1][1].score(Xt, y, **score_params)

        # metadata routing is enabled.
        routed_params = pipeline.process_routing(
            self, "score", sample_weight=sample_weight, **params
        )

        Xt = X
        for _, name, transform in self._iter(with_final=False):
            Xt = transform.transform(Xt, **routed_params[name].transform)
        return self.steps[-1][1].score(Xt, y, **routed_params[self.steps[-1][0]].score)

    # def fit(self, X, y=None, **fit_params):
    #     fit_params_steps = self._check_fit_params(**fit_params)
    #     Xt = self._fit(X, y, **fit_params_steps)
    #
    #     if isinstance(Xt, tuple):  ###### unpack X if is tuple: X = (X,y)
    #         Xt, y = Xt
    #
    #     with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
    #         if self._final_estimator != "passthrough":
    #             fit_params_last_step = fit_params_steps[self.steps[-1][0]]
    #             self._final_estimator.fit(Xt, y, **fit_params_last_step)
    #
    #     return self

    def fit(self, X, y=None, **params):
        print('y before fit:', y)
        routed_params = self._check_method_params(method="fit", props=params)
        Xt = self._fit(X, y, routed_params)
        if isinstance(Xt, tuple):  ###### unpack X if is tuple: X = (X,y)
            Xt, y = Xt
        print('y after fit:', y)
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if self._final_estimator != "passthrough":
                last_step_params = routed_params[self.steps[-1][0]]
                self._final_estimator.fit(Xt, y, **last_step_params["fit"])

        return self


class LFADSSmoother(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        print(f"LFADSSmoother: X before transformation: {X}")
        print(f"LFADSSmoother: y before transformation: {y}")
        print('LFADssmoother: shape of X: ', X.shape)
        print('LFADssmoother: shape of y: ', y.shape)

        X, self.removed_indices = tools.apply_lfads_smoothing(X)
        X = scipy.stats.zscore(X, axis=0)



        if y is not None:
            y = np.delete(y, self.removed_indices, axis=0)

        print('LFADssmoother: shape of X after transform: ', X.shape)
        print('LFADssmoother: shape of y after transform: ', y.shape)
        print(f"LFADSSmoother: X after transformation: {X}")
        print(f"LFADSSmoother: y after transformation: {y}")
        return X, y

    def transform(self, X):
        # print(f"LFADSSmoother: X before transformation: {X}")
        X, self.removed_indices = tools.apply_lfads_smoothing(X)
        X = scipy.stats.zscore(X, axis=0)
        # print(f"LFADSSmoother: X after transformation: {X}")
        return X

# class IndexRemover(BaseEstimator, TransformerMixin):
#     def __init__(self, smoother):
#         self.smoother = smoother
#
#     def fit(self, X, y=None):
#         return self
#
#     def fit_transform(self, X, y=None):
#         X, y = self.transform(X, y)
#         return X, y
#
#     def transform(self, X, y=None):
#         if y is not None:
#             y = self.remove_indices(y)
#         assert X.size > 0, "The input numpy array is empty."
#         print('The shape of the data after removing the indices is:', X.shape)
#         print('The shape of the labels after removing the indices is:', y.shape)
#         assert not np.isnan(X).any(), "NaN values in X after IndexRemover"
#         assert not np.isnan(y).any(), "NaN values in y after IndexRemover"
#         return X, y
#
#     def remove_indices(self, y):
#         return np.delete(y, self.smoother.removed_indices, axis=0)


class IndexRemover(BaseEstimator, TransformerMixin):
    def __init__(self, smoother):
        self.smoother = smoother

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print(f"IndexRemover: y before transformation: {y}")
        if y is not None:
            y = self.remove_indices(y)
        print(f"IndexRemover: y after transformation: {y}")
        return X, y

    def remove_indices(self, y):
        return np.delete(y, self.smoother.removed_indices, axis=0)


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
        X_transformed = self.model_.transform(X)
        assert not np.isnan(X_transformed).any(), "NaN values in X after UMAP transformation"
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
        reducer_kwargs, save_dir_path, use_rand_search=False, manual_params=None, rat_id=None, savedir=None,
        num_windows=None
):
    y = bhv[regress].values

    random_search_results = []

    # Create your custom folds
    n_timesteps = spks.shape[0]

    custom_folds = create_folds(n_timesteps, num_folds=5, num_windows=num_windows)
    # Example, you can use your custom folds here
    # pipeline = Pipeline([
    #     ('reducer', CustomUMAP()),
    #     ('estimator', MultiOutputRegressor(regressor()))
    # ])



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
        smoother = LFADSSmoother()
        index_remover = IndexRemover(smoother)

        pipeline = Pipeline([
            ('smoother', smoother),
            ('reducer', CustomUMAP()),
            ('estimator', MultiOutputRegressor(regressor()))
        ])
        param_grid = {
            'reducer__n_components': [3, 4, 5, 6, 7, 8, 9, 10],
            'reducer__min_dist': [0.0001, 0.001, 0.01, 0.1, 0.3],
            'reducer__n_neighbors': [10, 20, 30, 40, 50, 60, 70],
            'reducer__random_state': [42],
            'estimator__estimator__n_neighbors': [2, 5, 10, 20, 30, 40, 50, 60, 70],
            'estimator__estimator__metric': ['cosine', 'euclidean', 'minkowski'], }

        # Initialize BayesSearchCV
        # logger.info('Starting the random search, at line 209')
        # random_search = RandomizedSearchCV(
        #     pipeline,
        #     param_distributions=param_grid,
        #     n_iter=1000,
        #     cv=custom_folds,
        #     verbose=3,
        #     n_jobs=-1,
        #     scoring='neg_mean_squared_error'
        # )
        #
        # # Fit BayesSearchCV
        # random_search.fit(spks, y)


        # Randomly select one set of parameters
        param_list = list(ParameterSampler(param_grid, n_iter=1))
        params = param_list[0]
        pipeline.set_params(**params)
        pipeline.fit(spks, y)
        #get the train and test scores
        train_score = pipeline.score(spks, y)

        #
        # # Get the best parameters and score
        # best_params = random_search.best_params_
        # best_score = random_search.best_score_
    else:
        # Manually set the parameters

        # Initialize lists to store the scores
        train_scores = []
        test_scores = []

        # Loop over the custom folds
        count = 0
        for train_index, test_index in custom_folds:
            # Split the data into training and testing sets
            spks_train, spks_test = spks[train_index], spks[test_index]
            y_train, y_test = y[train_index], y[test_index]

            spks_train, removed_indices_train = tools.apply_lfads_smoothing(spks_train)
            spks_test, removed_indices_test = tools.apply_lfads_smoothing(spks_test)

            spks_train = scipy.stats.zscore(spks_train, axis=0)

            spks_test = scipy.stats.zscore(spks_test, axis=0)
            # remove the removed labels
            print('removing the removed indices')
            y_test = np.delete(y_test, removed_indices_test, axis=0)
            y_train = np.delete(y_train, removed_indices_train, axis=0)

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
            # add a color bar
            cbar = plt.colorbar(sc, ax=ax)
            ax.set_title('UMAP test embeddings color-coded by head angle rel. \n  to goal for fold: ' + str(
                count) + 'rat id:' + str(rat_id))
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
    # data_dir = '/ceph/scratch/carlag/honeycomb_neural_data/rat_7/6-12-2019/'
    base_dir = 'C:/neural_data/'

    for data_dir in [f'{base_dir}/rat_7/6-12-2019', f'{base_dir}/rat_10/23-11-2021',
                     f'{base_dir}/rat_8/15-10-2019', f'{base_dir}/rat_9/10-12-2021',
                     f'{base_dir}/rat_3/25-3-2019']:
        spike_dir = os.path.join(data_dir, 'physiology_data')
        dlc_dir = os.path.join(data_dir, 'positional_data')
        labels = np.load(
            f'{dlc_dir}/labels_250_scale_to_angle_range_False.npy')
        col_list = np.load(f'{dlc_dir}/col_names_250_scale_to_angle_range_False.npy')

        spike_data = np.load(f'{spike_dir}/inputs_10052024_250.npy')
        old_spike_data = np.load(f'{spike_dir}/inputs_overlap_False_window_size_250.npy')
        # check if they are the same array
        # if np.allclose(spike_data, old_spike_data):
        #     print('The two arrays are the same')
        # else:
        #     print('The two arrays are not the same')

        window_df = pd.read_csv(
            f'C:/neural_data/mean_p_value_vs_window_size_across_rats_grid_100250windows_scale_to_angle_range_False.csv')
        # find the rat_id
        rat_id = data_dir.split('/')[-2]
        # filter for window_size
        window_df = window_df[window_df['window_size'] == 250]
        num_windows = window_df[window_df['rat_id'] == rat_id]['minimum_number_windows'].values[0]
        # print out the first couple of rows of the lfp_data
        # previous_results, score_dict = DataHandler.load_previous_results('lfp_phase_manifold_withspkdata')
        rat_id = data_dir.split('/')[-2]
        # manual_params = previous_results[rat_id]

        spike_data_copy = copy.deepcopy(spike_data)
        tolerance = 1e-10  # or any small number that suits your needs
        if np.any(np.abs(np.std(spike_data_copy, axis=0)) < tolerance):
            print('There are neurons with constant firing rates')
            # remove those neurons
            spike_data_copy = spike_data_copy[:, np.abs(np.std(spike_data_copy, axis=0)) >= tolerance]

        # X_for_umap, removed_indices = tools.apply_lfads_smoothing(spike_data_copy)
        X_for_umap = spike_data_copy
        # X_for_umap = scipy.stats.zscore(X_for_umap, axis=0)

        # labels_for_umap = np.delete(labels, removed_indices, axis=0)

        labels_for_umap = labels
        label_df = pd.DataFrame(labels_for_umap,
                                columns=col_list)

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

        regress = ['x', 'y', 'cos_relative_direction', 'sin_relative_direction']  # changing to two target variables

        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        now_day = datetime.now().strftime("%Y-%m-%d")
        filename = f'params_all_trials_randsearch_250bin_num_windows{num_windows}_jake_fold_allvars_{now}.npy'
        filename_mean_score = f'mean_score_all_trials_randsearch_250bin_numwindows{num_windows}_jake_fold_{now}.npy'
        save_dir_path = Path(
            f'{data_dir}/randsearch_allvars_lfadssmooth_empiricalwindow_zscoredlabels_1000iter_independentvar_smoothaftersplit_v2_{now_day}')
        save_dir_path.mkdir(parents=True, exist_ok=True)
        # initalise a logger
        # logger = logging.getLogger(__name__)
        # logger.setLevel(logging.INFO)
        # # create a file handler
        # handler = logging.FileHandler(save_dir_path / f'rand_search_cv_{now}.log')
        # handler.setLevel(logging.INFO)
        # # create a logging format
        # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # handler.setFormatter(formatter)
        # # add the handlers to the logger
        # logger.addHandler(handler)
        # logger.info('Starting the training and testing of the lfp data with the spike data')
        # remove numpy array, just get mapping from manual_params
        # manual_params = manual_params.item()

        # get the num_windows from an already calculated csv file

        best_params, mean_score = train_and_test_on_umap_randcv(
            X_for_umap,
            label_df,
            regress,
            regressor,
            regressor_kwargs,
            reducer,
            reducer_kwargs, save_dir_path, use_rand_search=True, manual_params=None, savedir=save_dir_path,
            rat_id=rat_id, num_windows=num_windows
        )
        np.save(save_dir_path / filename, best_params)
        np.save(save_dir_path / filename_mean_score, mean_score)


if __name__ == '__main__':
    #
    main()