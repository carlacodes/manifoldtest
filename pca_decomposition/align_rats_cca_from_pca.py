# from pathlib import Path
import copy
import matplotlib.pyplot as plt
from scipy.stats import randint
from sklearn.neighbors import KNeighborsRegressor
from pathlib import Path
from sklearn.cross_decomposition import CCA
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.multioutput import MultiOutputRegressor
from mvlearn.embed import GCCA
from sklearn.decomposition import PCA
import seaborn as sns
from scipy.stats import ttest_ind
from manifold_neural.helpers import cca_tools
from manifold_neural.helpers import visualisation
from numpy import mean, std, var, sqrt
import scipy.linalg as linalg
import os
import scipy
import pickle as pkl
os.environ['JOBLIB_TEMP_FOLDER'] = 'C:/tmp'

def passthrough_func(X):
    return X



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




def train_and_test_on_pca_randcv(
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
    param_grid = {'estimator__n_neighbors': [5], 'reducer__n_components': [6], 'estimator__metric': ['euclidean'], 'reducer__n_neighbors': [10], 'reducer__min_dist': [0.3], 'reducer__random_state': [42]}
    param_grid_200windows = {'estimator__n_neighbors': [10], 'reducer__n_components': [7], 'estimator__metric': ['minkowski'], 'reducer__n_neighbors': [20], 'reducer__min_dist': [0.1]}


    y = bhv[regress].values

    random_search_results = []
    random_search_results_shuffled = []

    # Create your custom folds
    n_timesteps = spks.shape[0]
    custom_folds = create_folds(n_timesteps, num_folds=10, num_windows=1000)

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
        regressor_kwargs.update(
            {k.replace('estimator__', ''): v for k, v in params.items() if k.startswith('estimator__')})
        reducer_kwargs.update({k.replace('reducer__', ''): v for k, v in params.items() if k.startswith('reducer__')})
        # Initialize the regressor with current parameters
        current_regressor = MultiOutputRegressor(regressor(**regressor_kwargs))
        current_regressor_shuffled = MultiOutputRegressor(regressor(**regressor_kwargs))
        # Initialize the reducer with current parameters
        current_reducer = reducer(**reducer_kwargs)
        current_reducer_shuffled = copy.deepcopy(current_reducer)

        scores = []
        scores_train = []
        count = 0
        scores_shuffled = []
        scores_train_shuffled = []


        for train_index, test_index in custom_folds:
            X_train, X_test = spks[train_index], spks[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Apply dimensionality reduction
            X_train_reduced = current_reducer.fit_transform(X_train)

            #take the inverse transform of the reduced data
            # X_train_reduced_mapped_back = current_reducer.inverse_transform(X_train_reduced)
            X_test_reduced = current_reducer.transform(X_test)

            X_train_reduced_shuffled = X_train_reduced.copy()
            np.random.shuffle(X_train_reduced_shuffled)

            X_test_reduced_shuffled = X_test_reduced.copy()
            np.random.shuffle(X_test_reduced_shuffled)



            # Fit the regressor
            current_regressor.fit(X_train_reduced, y_train)
            current_regressor_shuffled.fit(X_train_reduced_shuffled, y_train)


            # Evaluate the regressor: using the default for regressors which is r2
            score = current_regressor.score(X_test_reduced, y_test)
            score_shuffled = current_regressor_shuffled.score(X_test_reduced_shuffled, y_test)

            score_train = current_regressor.score(X_train_reduced, y_train)
            score_train_shuffled = current_regressor_shuffled.score(X_train_reduced_shuffled, y_train)


            scores.append(score)
            scores_shuffled.append(score_shuffled)

            scores_train_shuffled.append(score_train_shuffled)
            scores_train.append(score_train)

            y_pred = current_regressor.predict(X_test_reduced)
            fig, ax = plt.subplots(1, 1)
            ax.scatter(y_test, y_pred)
            ax.set_title('y_test vs y_pred for fold: ' + str(count))
            plt.show()

            fig, ax = plt.subplots(1, 1)
            plt.plot(y_pred[:, 0], label='y_pred', alpha = 0.5)
            plt.plot(y_test[:, 0], label='y_test', alpha = 0.5)
            ax.set_title('y_pred (sin theta) for fold: ' + str(count))
            ax.set_xlabel('time in SAMPLES')
            plt.savefig('C:/neural_data/rat_7/6-12-2019/cluster_results/y_pred_vs_y_test_sin_fold_' + str(count) + '.png')
            plt.show()

            fig, ax = plt.subplots(1, 1)
            plt.plot(y_pred[:, 1], label='y_pred', alpha = 0.5)
            plt.plot(y_test[:, 1], label='y_test', alpha = 0.5)
            ax.set_title('y_pred (cos theta) for fold: ' + str(count))
            ax.set_xlabel('time in SAMPLES')
            plt.legend()
            plt.savefig('C:/neural_data/rat_7/6-12-2019/cluster_results/y_pred_vs_y_test_cos_fold_' + str(count) + '.png')
            plt.show()

            ##now plot the shuffled data
            y_pred_shuffled = current_regressor_shuffled.predict(X_test_reduced_shuffled)
            fig, ax = plt.subplots(1, 1)
            ax.scatter(y_test, y_pred_shuffled, c='orange')
            ax.set_title('y_test vs y_pred for fold: ' + str(count) + ' shuffled')
            plt.savefig('C:/neural_data/rat_7/6-12-2019/cluster_results/y_pred_vs_y_test_shuffled_fold_' + str(count) + '.png')
            plt.show()

            fig, ax = plt.subplots(1, 1)
            plt.plot(y_pred_shuffled[:, 0], label='y_pred', alpha = 0.5)
            plt.plot(y_test[:, 0], label='y_test', alpha = 0.5)
            ax.set_title('y_pred (sin theta) for fold: ' + str(count) + ' shuffled')
            ax.set_xlabel('time in SAMPLES')
            plt.legend()
            plt.savefig('C:/neural_data/rat_7/6-12-2019/cluster_results/y_pred_vs_y_test_sin_fold_' + str(count) + 'shuffled.png')
            plt.show()

            fig, ax = plt.subplots(1, 1)
            plt.plot(y_pred_shuffled[:, 1], label='y_pred', alpha = 0.5)
            plt.plot(y_test[:, 1], label='y_test', alpha = 0.5)
            ax.set_title('y_pred (cos theta) for fold: ' + str(count) + ' shuffled')
            ax.set_xlabel('time in SAMPLES')
            plt.legend()
            plt.savefig('C:/neural_data/rat_7/6-12-2019/cluster_results/y_pred_vs_y_test_cos_fold_' + str(count) + 'shuffled.png')
            count += 1

        #now do the same for the SHUFFLED embeddings



        # Calculate mean score for the current parameter combination
        mean_score = np.mean(scores)
        mean_score_train = np.mean(scores_train)

        mean_score_shuffled = np.mean(scores_shuffled)
        mean_score_train_shuffled = np.mean(scores_train_shuffled)

        random_search_results.append((params, mean_score))
        random_search_results_shuffled.append((params, mean_score_shuffled))

    # Select the best parameters based on mean score
    best_params, _ = max(random_search_results, key=lambda x: x[1])
    #get the best mean score which is the second entry in the tuple
    _, mean_score = max(random_search_results, key=lambda x: x[1])

    best_params_shuffled, _ = max(random_search_results_shuffled, key=lambda x: x[1])
    _, mean_score_shuffled = max(random_search_results_shuffled, key=lambda x: x[1])
    return best_params, mean_score, best_params_shuffled, mean_score_shuffled


def load_previous_results():
    param_dict = {}
    score_dict = {}
    for rat_dir in ['C:/neural_data/rat_10/23-11-2021','C:/neural_data/rat_7/6-12-2019', 'C:/neural_data/rat_8/15-10-2019', 'C:/neural_data/rat_9/10-12-2021', 'C:/neural_data/rat_3/25-3-2019']:
        rat_id = rat_dir.split('/')[-2]
        pca_decomp_directory = f'{rat_dir}/cluster_results/pca_decomposition'
        #find all the files in the directory
        files = os.listdir(pca_decomp_directory)

        for window in [1000]:
            for bin_size in [250]:
                #find the file names
                for file in files:
                    if file.__contains__(f'{bin_size}bin_{window}windows'):
                        if file.__contains__('mean_score'):
                                score_dict[rat_id] = np.load(f'{pca_decomp_directory}/{file}')
                        elif file.__contains__('params'):
                            with open(f'{pca_decomp_directory}/{file}', 'rb') as f:
                                param_dict[rat_id] =  np.load(f'{pca_decomp_directory}/{file}', allow_pickle=True)
    return param_dict, score_dict


def plot_knn_decoding_results(y_pred, y_1_test, savefig_dir, i, score, score_train, score_shuffled, current_regressor_1, X_train_reduced_1, ):
    fig, ax = plt.subplots(1, 1)
    plt.plot(y_pred[:, 0], label='y_pred', alpha=0.5)
    plt.plot(y_1_test[:, 0], label='y_test', alpha=0.5)
    ax.set_title('y_pred (sin theta) for fold: ' + str(i) + ', r_2 score:' + str(score), fontsize=10)
    ax.set_xlabel('time in SAMPLES')
    plt.savefig(
        f'{savefig_dir}/y_pred_vs_y_test_sin_fold_' + str(i) + '.png')
    plt.show()

    fig, ax = plt.subplots(1, 1)
    plt.plot(y_pred[:, 1], label='y_pred', alpha=0.5)
    plt.plot(y_1_test[:, 1], label='y_test', alpha=0.5)
    ax.set_title('y_pred (cos theta) for fold: ' + str(i) + ', r_2 score: ' + str(score))
    ax.set_xlabel('time in SAMPLES')
    plt.legend()
    plt.savefig(
        f'{savefig_dir}/y_pred_vs_y_test_cos_fold_' + str(i) + '.png')
    plt.show()

    # do the same for the training data
    y_pred_train = current_regressor_1.predict(X_train_reduced_1)
    fig, ax = plt.subplots(1, 1)
    plt.plot(y_pred_train[:, 0], label='y_pred', alpha=0.5)
    plt.plot(y_1_train[:, 0], label='y_train', alpha=0.5)
    ax.set_title('y_pred (sin theta) for fold: ' + str(i) + ', r_2 score:' + str(score_train), fontsize=10)
    ax.set_xlabel('time in SAMPLES')
    plt.savefig(
        f'{savefig_dir}/y_pred_vs_y_train_sin_fold_' + str(i) + '.png')
    plt.show()

    fig, ax = plt.subplots(1, 1)
    plt.plot(y_pred_train[:, 1], label='y_pred', alpha=0.5)
    plt.plot(y_1_train[:, 1], label='y_train', alpha=0.5)
    ax.set_title('y_pred (cos theta) for fold: ' + str(i) + 'r_2 score: ' + str(score_train))
    ax.set_xlabel('time in SAMPLES')
    plt.legend()
    plt.savefig(
        f'{savefig_dir}/y_pred_vs_y_train_cos_fold_' + str(i) + '.png')

    ##now plot the shuffled data
    y_pred_shuffled = current_regressor_1_shuffled.predict(X_test_reduced_1_shuffled)
    fig, ax = plt.subplots(1, 1)
    ax.scatter(y_1_test, y_pred_shuffled, c='orange')
    ax.set_title('y_test vs y_pred for fold: ' + str(i) + ' shuffled, r_2 score: ' + str(score_shuffled))
    plt.savefig(f'{savefig_dir}/y_pred_vs_y_test_shuffled_fold_' + str(
        i) + '.png')
    plt.show()

    fig, ax = plt.subplots(1, 1)
    plt.plot(y_pred_shuffled[:, 0], label='y_pred', alpha=0.5)
    plt.plot(y_1_test[:, 0], label='y_test', alpha=0.5)
    ax.set_title('y_pred (sin theta) for fold: ' + str(i) + ' shuffled , r_2 score: ' + str(score_shuffled))
    ax.set_xlabel('time in SAMPLES')
    plt.legend()
    plt.savefig(f'{savefig_dir}/y_pred_vs_y_test_sin_fold_' + str(
        i) + 'shuffled.png')
    plt.show()

    fig, ax = plt.subplots(1, 1)
    plt.plot(y_pred_shuffled[:, 1], label='y_pred', alpha=0.5)
    plt.plot(y_1_test[:, 1], label='y_test', alpha=0.5)
    ax.set_title('y_pred (cos theta) for fold: ' + str(i) + ' shuffled, r_2 score: ' + str(score_shuffled))
    ax.set_xlabel('time in SAMPLES')
    plt.legend()
    plt.savefig(f'{savefig_dir}/y_pred_vs_y_test_cos_fold_' + str(
        i) + 'shuffled.png')
    plt.show()
    return

def run_cca_on_rat_data(data_store, param_dict, fold_store):
    regressor_kwargs = {'n_neighbors': 70}

    reducer = PCA
    regressor = KNeighborsRegressor

    reducer_kwargs_1 = {
        'n_components': 3,
    }
    reducer_kwargs_2 = {
        'n_components': 3,
    }
    regressor_kwargs_1 = {'n_neighbors': 70}
    regressor_kwargs_2 = {'n_neighbors': 70}

    corr_dict = {}
    score_dict = {}
    for rat_id_1 in data_store.keys():
        scores = []
        scores_shuffled = []
        scores_train = []
        scores_train_shuffled = []
        print(f'Working on rat {rat_id_1}')
        for rat_id_2 in data_store.keys():
            print(f'Working on rat {rat_id_2}')
            if rat_id_1 == rat_id_2:
                print(f'Skipping {rat_id_1} and {rat_id_2}')
                continue
            params_1 = param_dict[rat_id_1]
            params_2 = param_dict[rat_id_2]
            #remove np array
            params_1 = params_1.item()
            params_2 = params_2.item()

            X_rat_1 = data_store[rat_id_1]['X']
            X_rat_2 = data_store[rat_id_2]['X']
            labels_rat_1 = data_store[rat_id_1]['labels']
            labels_rat_2 = data_store[rat_id_2]['labels']

            data_1_sin = labels_rat_1['angle_sin'].values
            data_1_cos = labels_rat_1['angle_cos'].values
            data_2_sin = labels_rat_2['angle_sin'].values
            data_2_cos = labels_rat_2['angle_cos'].values

            folds_rat_1 = fold_store[rat_id_1]
            folds_rat_2 = fold_store[rat_id_2]
            #check which folds are shorter
            folds_rat_1_len = [len(fold) for fold in folds_rat_1[0]]
            folds_rat_2_len = [len(fold) for fold in folds_rat_2[0]]
            print(f'The lengths of the folds for rat {rat_id_1} are {folds_rat_1_len}')
            # if len(set(folds_rat_1_len)) > len(set(folds_rat_2_len)):
            #     custom_folds = folds_rat_2
            # else:
            #     custom_folds = folds_rat_1

            for i in range(len(folds_rat_1)):
                regressor_kwargs_1.update(
                    {k.replace('estimator__', ''): v for k, v in params_1.items() if k.startswith('estimator__')})
                reducer_kwargs_1.update(
                    {k.replace('reducer__', ''): v for k, v in params_1.items() if k.startswith('reducer__')})

                regressor_kwargs_2.update(
                    {k.replace('estimator__', ''): v for k, v in params_2.items() if k.startswith('estimator__')})
                reducer_kwargs_2.update(
                    {k.replace('reducer__', ''): v for k, v in params_2.items() if k.startswith('reducer__')})


                # Initialize the reducer with current parameters
                current_reducer_1 = reducer(**reducer_kwargs_1)
                current_reducer_2 = reducer(**reducer_kwargs_2)
                #todo: need to check if truncating is ok
                if len(folds_rat_1[i][1]) > len(folds_rat_2[i][1]):
                    folds_rat_1[i] = (folds_rat_1[i][0], folds_rat_1[i][1][0:len(folds_rat_2[i][1])])
                elif len(folds_rat_1[i][1]) < len(folds_rat_2[i][1]):
                    folds_rat_2[i] = (folds_rat_2[i][0], folds_rat_2[i][1][0:len(folds_rat_1[i][1])])


                X_train_1, X_test_1 = X_rat_1[folds_rat_1[i][0]], X_rat_1[folds_rat_1[i][1]]
                data_1_train_sin, data_1_test_sin = data_1_sin[folds_rat_1[i][0]], data_1_sin[folds_rat_1[i][1]]
                data_1_train_cos, data_1_test_cos = data_1_cos[folds_rat_1[i][0]], data_1_cos[folds_rat_1[i][1]]

                y_1_train = np.stack((data_1_train_sin, data_1_train_cos), axis = 1)
                y_1_test = np.stack((data_1_test_sin, data_1_test_cos), axis = 1)


                X_train_2, X_test_2 = X_rat_2[folds_rat_2[i][0]], X_rat_2[folds_rat_2[i][1]]
                data_2_train_sin, data_2_test_sin = data_2_sin[folds_rat_2[i][0]], data_2_sin[folds_rat_2[i][1]]
                data_2_train_cos, data_2_test_cos = data_2_cos[folds_rat_2[i][0]], data_2_cos[folds_rat_2[i][1]]

                # Apply dimensionality reduction
                X_train_reduced_1 = current_reducer_1.fit_transform(X_train_1)
                X_train_reduced_2 = current_reducer_2.fit_transform(X_train_2)
                np.array_equal(X_train_reduced_1, X_train_reduced_2)

                X_test_reduced_1 = current_reducer_1.transform(X_test_1)
                X_test_reduced_2 = current_reducer_2.transform(X_test_2)
                #check if X_test_reduced_1 and X_test_reduced_2 are the same
                np.array_equal(X_test_reduced_1, X_test_reduced_2)

                ##get the train r2 and test r2

                # Fit the regressor
                current_regressor_1 = MultiOutputRegressor(regressor(**regressor_kwargs))
                current_regressor_1.fit(X_train_reduced_1, y_1_train)

                current_regressor_1_shuffled = MultiOutputRegressor(regressor(**regressor_kwargs))
                X_train_reduced_1_shuffled = X_train_reduced_1.copy()
                X_test_reduced_1_shuffled = X_test_reduced_1.copy()



                np.random.shuffle(X_train_reduced_1_shuffled)
                np.random.shuffle(X_test_reduced_1_shuffled)

                current_regressor_1_shuffled.fit(X_train_reduced_1_shuffled, y_1_train)

                # Evaluate the regressor: using the default for regressors which is r2
                score = current_regressor_1.score(X_test_reduced_1, y_1_test)
                score_shuffled = current_regressor_1_shuffled.score(X_test_reduced_1_shuffled, y_1_test)

                score_train = current_regressor_1.score(X_train_reduced_1, y_1_train)
                score_train_shuffled = current_regressor_1_shuffled.score(X_train_reduced_1_shuffled, y_1_train)

                scores.append(score)
                scores_shuffled.append(score_shuffled)

                scores_train_shuffled.append(score_train_shuffled)
                scores_train.append(score_train)

                y_pred = current_regressor_1.predict(X_test_reduced_1)
                savefig_dir = f'C:/neural_data/r2_decoding_figures/pca/{rat_id_1}'
                if not os.path.exists(savefig_dir):
                    os.makedirs(savefig_dir)




                #apply cca to the reduced data
                A, B, r, U, V = cca_tools.canoncorr(X_test_reduced_1, X_test_reduced_2, fullReturn=True)
                #get the mean of the correlation coefficients
                avg_corr = np.mean(r)

                # correlation_matrix = np.corrcoef(data1_c.T, data2_c.T)

                # Since corrcoef returns a matrix, we only need the off-diagonal elements which represent the correlation between the two datasets.
                # correlation = correlation_matrix[np.triu_indices(data1_c.shape[1], k=1)]

                # print("Correlation coefficients:", correlation)
                # #average the correlation coefficients
                # avg_corr = np.mean(correlation)
                print(f'The average correlation coefficient is {avg_corr} between rats: {rat_id_1} and {rat_id_2}')
                #add the correlation coefficient to a dictionary
                corr_dict[rat_id_1 + '_' + rat_id_2] = r

                #for each pair plot the resultant data
                fig, ax = plt.subplots(1, 1)
                ax.scatter(U[:, 0], V[:, 0])
                #round the r[0] value to 3 decimal places
                r_rounded = np.round(r, 3)
                ax.set_title(f'CCA component 1 for rats {rat_id_1} and {rat_id_2}, r[0]: {r_rounded[0]}')
                plt.savefig('../figures/cca/cca_component_1_' + rat_id_1 + '_' + rat_id_2 + '.png')
                plt.show()
                coef = [A, B]
                #convert coef to a matrix
                coef = np.array(coef)
                U_1, _, Vh_1 = linalg.svd(A, full_matrices=False, compute_uv=True, overwrite_a=False, check_finite=False)
                aligned_data_1 = X_test_reduced_1 @ U_1 @ Vh_1
                U_2, _, Vh_2 = linalg.svd(B, full_matrices=False, compute_uv=True, overwrite_a=False, check_finite=False)
                aligned_data_2 = X_test_reduced_2 @ U_2 @ Vh_2

                data_1_c_sin = np.interp(data_1_test_sin, (data_1_test_sin.min(), data_1_test_sin.max()), (0, 255)).astype(int)
                data_1_c_cos = np.interp(data_1_test_cos, (data_1_test_cos.min(), data_1_test_cos.max()), (0, 255)).astype(int)
                data_2_c_sin = np.interp(data_2_test_sin, (data_2_test_sin.min(), data_2_test_sin.max()), (0, 255)).astype(int)
                data_2_c_cos = np.interp(data_2_test_cos, (data_2_test_cos.min(), data_2_test_cos.max()), (0, 255)).astype(int)



                colormap = visualisation.colormap_2d()
                color_data_rat_1 = colormap[data_1_c_sin, data_1_c_cos]
                color_data_rat_2 = colormap[data_2_c_sin, data_2_c_cos]



                fig = plt.figure()
                # Create first subplot for aligned_data_1
                ax1 = fig.add_subplot(121, projection='3d')  # 121 means: 1 row, 2 columns, first plot
                ax1.scatter(aligned_data_1[:, 0], aligned_data_1[:, 1], aligned_data_1[:, 2], c=color_data_rat_1)
                ax1.set_title(f'{rat_id_1}')

                # Create second subplot for aligned_data_2
                ax2 = fig.add_subplot(122, projection='3d')  # 122 means: 1 row, 2 columns, second plot
                ax2.scatter(aligned_data_2[:, 0], aligned_data_2[:, 1], aligned_data_2[:, 2], c=color_data_rat_2)
                ax2.set_title(f'{rat_id_2}')
                r_rounded = np.round(r, 3)
                r_rounded = float(r_rounded[0])
                plt.suptitle(f'Aligned PCA embeddings for rats {rat_id_1} and {rat_id_2}, r[0]: {r_rounded}')
                plt.savefig('../figures/cca/aligned_PCA_embedding_data_' + rat_id_1 + '_' + rat_id_2 + '.png', bbox_inches='tight', dpi = 300)
                plt.show()

                fig, ax = plt.subplots(1, 1)
                ax.imshow(colormap)
                ax.set_title('Color map for {}'.format(rat_id_1))
                plt.savefig(f'../figures/cca/color_map_fold_{rat_id_1}.png')
                plt.show()



                #plot the unaligned data
                fig = plt.figure()
                ax1 = fig.add_subplot(121, projection='3d')
                ax1.scatter(X_test_reduced_1[:, 0], X_test_reduced_1[:, 1], X_test_reduced_1[:, 2], c=color_data_rat_1)
                ax1.set_title(f'{rat_id_1}')

                ax2 = fig.add_subplot(122, projection='3d')
                ax2.scatter(X_test_reduced_2[:, 0], X_test_reduced_2[:, 1], X_test_reduced_2[:, 2], c=color_data_rat_2)
                ax2.set_title(f'{rat_id_2}')
                plt.suptitle(f'PCA embeddings for rats {rat_id_1} and {rat_id_2}')
                plt.savefig('../figures/cca/original_PCA_embedding_data_' + rat_id_1 + '_' + rat_id_2 + '.png',
                            bbox_inches='tight', dpi=300)
                plt.show()





    return

def run_gcca_on_rat_data(data_store, param_dict, fold_store):
    regressor_kwargs = {'n_neighbors': 70}

    reducer = PCA
    X_reduced_store_train = {}
    X_reduced_store_test = {}

    reducer_kwargs_1 = {
        'n_components': 3,
        # 'n_neighbors': 70,
        # 'min_dist': 0.3,
    }
    reducer_kwargs_2 = {
        'n_components': 3,
        # 'n_neighbors': 70,
        # 'min_dist': 0.3,
        'metric': 'euclidean',
        'n_jobs': 1,
    }
    regressor_kwargs_1 = {'n_neighbors': 70}

    corr_dict = {}
    X_reduced_store_test = {}
    X_reduced_store_test_list = []
    for rat_id_1 in data_store.keys():
        params_1 = param_dict[rat_id_1]
        #remove np array
        params_1 = params_1.item()

        X_rat_1 = data_store[rat_id_1]['X']
        folds_rat_1 = fold_store[rat_id_1]
        #check which folds are shorter
        folds_rat_1_len = [len(fold) for fold in folds_rat_1[0]]
        print(f'The lengths of the folds for rat {rat_id_1} are {folds_rat_1_len}')


        for i in range(1):
            regressor_kwargs_1.update(
                {k.replace('estimator__', ''): v for k, v in params_1.items() if k.startswith('estimator__')})
            reducer_kwargs_1.update(
                {k.replace('reducer__', ''): v for k, v in params_1.items() if k.startswith('reducer__')})




            # Initialize the reducer with current parameters
            current_reducer_1 = reducer(**reducer_kwargs_1)
            #todo: need to check if truncating is ok
            X_train_1, X_test_1 = X_rat_1[folds_rat_1[i][0]], X_rat_1[folds_rat_1[i][1]]



            # Apply dimensionality reduction
            X_train_reduced_1 = current_reducer_1.fit_transform(X_train_1)

            X_test_reduced_1 = current_reducer_1.transform(X_test_1)
            X_reduced_store_train[rat_id_1] = X_train_reduced_1
            X_reduced_store_test[rat_id_1] = X_test_reduced_1
            X_reduced_store_test_list.append(X_test_reduced_1)

    #apply gcca to the reduced data
    gcca = GCCA()
    #check if the number of rows is the same
    for rat_id in data_store.keys():
        # X_reduced_store_train[rat_id] = X_reduced_store_train[rat_id][0:1000, :]
        X_reduced_store_test[rat_id] = X_reduced_store_test[rat_id][0:1000, :]

    for i in range(len(X_reduced_store_test_list)):
        X_reduced_store_test_list[i] = X_reduced_store_test_list[i][0:1000, :]

    latents = gcca.fit_transform(X_reduced_store_test_list)
    # Assuming latents is a list of 2D arrays, where each 2D array corresponds to a dataset
    corr_dict_gcca = {}
    # Assuming latents is a list of 2D arrays, where each 2D array corresponds to a dataset
    for i in range(latents[0].shape[1]):  # Iterate over dimensions
        latent_neural_model_1 = latents[:, :, i]
        print(latent_neural_model_1.shape)
        # Calculate the correlation between the latent variables
        corr = np.corrcoef(latent_neural_model_1)
        #store the correlation in a dictionary
        corr_dict_gcca[f'latent_{i}'] = corr
    return



def main():
    data_dir = 'C:/neural_data/rat_7/6-12-2019'
    param_dict, scoredict = load_previous_results()
    #loop over the data dirs
    data_dirs = [ 'C:/neural_data/rat_10/23-11-2021','C:/neural_data/rat_7/6-12-2019', 'C:/neural_data/rat_8/15-10-2019', 'C:/neural_data/rat_9/10-12-2021', 'C:/neural_data/rat_3/25-3-2019']
    data_store_big = {}
    fold_store = {}
    for data_dir in data_dirs:
        rat_id = data_dir.split('/')[-2]
        spike_dir = os.path.join(data_dir, 'physiology_data')
        dlc_dir = os.path.join(data_dir, 'positional_data')
        labels = np.load(f'{dlc_dir}/labels_1203_with_dist2goal_scale_data_False_zscore_data_False_overlap_False_window_size_250.npy')
        spike_data = np.load(f'{spike_dir}/inputs_overlap_False_window_size_250.npy')

        spike_data_trial = spike_data


        # check for neurons with constant firing rates
        tolerance = 1e-10  # or any small number that suits your needs
        if np.any(np.abs(np.std(spike_data_trial, axis=0)) < tolerance):
            print('There are neurons with constant firing rates')
            # remove those neurons
            spike_data_trial = spike_data_trial[:, np.abs(np.std(spike_data_trial, axis=0)) >= tolerance]
        # THEN DO THE Z SCORE
        X_for_pca = scipy.stats.zscore(spike_data_trial, axis=0)

        if np.isnan(X_for_pca).any():
            print('There are nans in the data')

        X_for_pca = scipy.ndimage.gaussian_filter(X_for_pca, 2, axes=0)
        n_timesteps = X_for_pca.shape[0]

        custom_folds = create_folds(n_timesteps, num_folds=10, num_windows=1000)

        labels_for_pca = labels[:, 0:6]
        labels_for_pca = scipy.ndimage.gaussian_filter(labels_for_pca, 2, axes=0)

        label_df = pd.DataFrame(labels_for_pca,
                                columns=['x', 'y', 'dist2goal', 'angle_sin', 'angle_cos', 'dlc_angle_zscore'])
        label_df['time_index'] = np.arange(0, label_df.shape[0])
        #add label_df and X_for_pca to a dictionary
        data_store = {'X': X_for_pca, 'labels': label_df}
        data_store_big[rat_id] = data_store
        fold_store[rat_id] = custom_folds



    run_cca_on_rat_data(data_store_big, param_dict, fold_store)
    run_gcca_on_rat_data(data_store_big, param_dict, fold_store)


if __name__ == '__main__':
    #
    main()