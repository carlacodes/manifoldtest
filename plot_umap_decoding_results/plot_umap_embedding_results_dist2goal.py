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
import shap
import pandas as pd
import plotly.graph_objects as go

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

def plot_kneighborsregressor_splits(reducer, knn, X_test_reduced, X_train_reduced, y_train, y_test, save_dir_path=None, fold_num=None,  rat_id = None):
    # Create a grid to cover the embedding space
    # Visualize the SHAP values
    # Visualize the SHAP values
    K = 100  # Number of samples
    K_vis = 800
    X_train_reduced_sampled = shap.sample(X_train_reduced, K)
    X_test_reduced_sampled = shap.sample(X_test_reduced, K_vis)


    # Use n_jobs for parallel computation
    n_jobs = -1  # Use all available cores
    explainer = shap.KernelExplainer(knn.predict, X_train_reduced_sampled, n_jobs=n_jobs)

    # Compute SHAP values for the test data
    shap_values = explainer.shap_values(X_test_reduced_sampled)

    # Visualize the SHAP values
    shap.summary_plot(shap_values[0], X_test_reduced_sampled, plot_type='dot', show = False)
    plt.title('SHAP values for the test data, rat ID: ' + str(rat_id))
    plt.xlabel('SHAP value (impact on distance to goal)')
    plt.ylabel('UMAP feature')
    plt.savefig(f'{save_dir_path}/shap_values_fold_{fold_num}2904.png', dpi=300, bbox_inches='tight')
    plt.close('all')


    return

def train_and_test_on_umap_randcv(
        spks,
        bhv,
        regress,
        regressor,
        regressor_kwargs,
        reducer,
        reducer_kwargs, param_dict, rat_id = None, plot_shaps = False
):
    # param_grid = {
    #     'estimator__n_neighbors': [2, 5, 10, 30, 40, 50, 60, 70],
    #     'reducer__n_components': [3, 4, 5, 6, 7, 8, 9],
    #     'estimator__metric': ['euclidean', 'cosine', 'minkowski'],
    # }
    param_grid = param_dict[rat_id]
    #convert to dictionary
    param_grid = param_grid.item()

    y = bhv[regress].values

    random_search_results = []

    # Create your custom folds
    n_timesteps = spks.shape[0]
    custom_folds = create_folds(n_timesteps, num_folds=10, num_windows=1000)
    # Example, you can use your custom folds here
    savedir = f'C:/neural_data/r2_decoding_figures/umap/{rat_id}/dist2goal/'
    #check if the directory exists
    if not os.path.exists(savedir):
        os.makedirs(savedir, exist_ok=True)

    for _ in range(1):  # 100 iterations for RandomizedSearchCV
        params = {key: (values) for key, values in param_grid.items()}
        regressor_kwargs.update(
            {k.replace('estimator__', ''): v for k, v in params.items() if k.startswith('estimator__')})
        reducer_kwargs.update({k.replace('reducer__', ''): v for k, v in params.items() if k.startswith('reducer__')})

        # Initialize the regressor with current parameters
        current_regressor = regressor(**regressor_kwargs)
        current_regressor_shuffled = regressor(**regressor_kwargs)

        # Initialize the reducer with current parameters
        current_reducer = reducer(**reducer_kwargs)
        current_reducer_shuffled = copy.deepcopy(current_reducer)
        count = 0
        scores = []
        scores_shuffled = []
        scores_train = []
        scores_train_shuffled = []
        random_search_results_train = []
        random_search_results_shuffled = []
        random_search_results_train_shuffled = []

        spks_shuffled = copy.deepcopy(spks)
        # Shuffle along the first axis
        np.random.shuffle(spks_shuffled)
        # Transpose, shuffle and transpose back to shuffle along the second axis
        # spks_shuffled = spks_shuffled.T
        # np.random.shuffle(spks_shuffled)
        # spks_shuffled = spks_shuffled.T

        for train_index, test_index in custom_folds:
            X_train, X_test = spks[train_index], spks[test_index]
            y_train, y_test = y[train_index], y[test_index]

            X_train_shuffled, X_test_shuffled = spks_shuffled[train_index], spks_shuffled[test_index]


            y_train_shuffled = copy.deepcopy(y_train)
            # np.random.shuffle(y_train_shuffled)
            y_test_shuffled = copy.deepcopy(y_test)
            # np.random.shuffle(y_test_shuffled)


            # Apply dimensionality reduction
            X_train_reduced = current_reducer.fit_transform(X_train)
            X_test_reduced = current_reducer.transform(X_test)

            # Fit the regressor
            current_regressor.fit(X_train_reduced, y_train)

            X_train_reduced_shuffled = current_reducer_shuffled.fit_transform(X_train_shuffled)

            X_test_reduced_shuffled = current_reducer_shuffled.transform(X_test_shuffled)

            # Fit the regressor
            current_regressor.fit(X_train_reduced, y_train)
            current_regressor_shuffled.fit(X_train_reduced_shuffled, y_train_shuffled)

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
            if plot_shaps and count == 0:
                plot_kneighborsregressor_splits(reducer, current_regressor, X_test_reduced, X_train_reduced, y_train, y_test, save_dir_path=savedir, fold_num=count, rat_id = rat_id)
            #plot the umap embeddings on a 3d scatter plot
            fig = plt.figure()

            ax = fig.add_subplot(111, projection='3d')
            sc = ax.scatter(X_test_reduced[:, 0], X_test_reduced[:, 1], X_test_reduced[:, 2], c=y_pred[:, 0], cmap='viridis')
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            ax.set_zlabel('UMAP 3')
            #add a color bar
            cbar = plt.colorbar(sc, ax=ax)
            ax.set_title('UMAP test embeddings color-coded by dist. to goal \n for fold: ' + str(count) + ' rat id: ' +str(rat_id))
            plt.savefig(f'{savedir}/umap_embeddings_fold_2904' + str(count) + '2904.png')


            fig = plt.figure()

            ax = fig.add_subplot(111, projection='3d')
            sc = ax.scatter(X_test_reduced_shuffled[:, 0], X_test_reduced_shuffled[:, 1], X_test_reduced_shuffled[:, 2], c=y_pred[:, 0], cmap='magma')
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            ax.set_zlabel('UMAP 3')
            #add a color bar
            cbar = plt.colorbar(sc, ax=ax)
            ax.set_title('UMAP SHUFFLED test embeddings color-coded by dist. to goal \n for fold: ' + str(count) + ' rat id: ' +str(rat_id))
            plt.savefig(f'{savedir}/umap_embeddings_SHUFFLED_fold_2904' + str(count) + '2904.png')
            #plt.show()



            fig, ax = plt.subplots(1, 1)
            ax.plot(X_test_reduced[0:120, 0], label = 'UMAP 1', alpha = 0.5)
            ax.plot(X_test_reduced_shuffled[0:120, 0], label = 'UMAP 1 SHUFFLED', alpha = 0.5)
            ax.set_title('UMAP 1 vs UMAP 1 SHUFFLED for fold, (distance 2 goal): ' + str(count))
            ax.set_xlabel('time in SAMPLES')
            plt.legend()
            plt.savefig(f'{savedir}/umap1_vs_umap1_shuffled_fold_' + str(count) + '2904.png', dpi = 300, bbox_inches = 'tight')

            # Create a 3D line plot
            fig = go.Figure(data=[go.Scatter3d(
                x=X_test_reduced[:, 0],
                y=X_test_reduced[:, 1],
                z=X_test_reduced[:, 2],
                mode='markers',
                marker=dict(
                    color=y_pred[:, 0],  # set color to prediction values
                    colorscale='Viridis',  # choose a colorscale
                    size=6
                )
            )])
            fig.update_layout(
                scene=dict(
                    xaxis_title='UMAP 1',
                    yaxis_title='UMAP 2',
                    zaxis_title='UMAP 3'
                )
            )

            # Set plot title
            fig.update_layout(title_text='UMAP test embeddings color-coded by dist. to goal for fold: ' + str(
                count) + ' rat id: ' + str(rat_id))

            # Save the figure as an HTML file
            fig.write_html(f'{savedir}/umap_embeddings_fold_' + str(count) + '2904.html')



            fig = go.Figure(data=[go.Scatter3d(
                x=X_test_reduced_shuffled[:, 0],
                y=X_test_reduced_shuffled[:, 1],
                z=X_test_reduced_shuffled[:, 2],
                mode='markers',
                marker=dict(
                    color=y_pred[:, 0],  # set color to prediction values
                    colorscale='Magma',  # choose a colorscale
                    size=6
                )
            )])

            # Set plot title
            fig.update_layout(title_text='UMAP test embeddings color-coded by dist. to goal for fold (shuffled): ' + str(
                count) + ' rat id: ' + str(rat_id))

            # Save the figure as an HTML file
            fig.write_html(f'{savedir}/umap_embeddings_shuffled_fold_' + str(count) + '2904.html')
            # fig.show()
            #close plotly object



            


            fig, ax = plt.subplots(1, 1)
            ax.scatter(y_test, y_pred)
            ax.set_title('y_test vs y_pred for fold: ' + str(count))
            #plt.show()

            fig, ax = plt.subplots(1, 1)
            plt.plot(y_pred[0:120, 0], label='y_pred', alpha=0.5)
            plt.plot(y_test[0:120, 0], label='y_test', alpha=0.5)
            ax.set_title('y_pred, rat ID:'+ str(rat_id) +'(dist2goal) for fold: ' + str(count)+ ' r2_score: ' + str(score))
            ax.set_xlabel('time in SAMPLES')
            plt.savefig(
                f'{savedir}/y_pred_vs_y_test_dist2goal_fold_' + str(count) + '2904.png', dpi = 300, bbox_inches = 'tight')
            #plt.show()





            #do the same for the train data
            y_pred_train = current_regressor.predict(X_train_reduced)
            fig, ax = plt.subplots(1, 1)
            ax.scatter(y_train, y_pred_train)
            ax.set_title('y_train vs y_pred for fold: ' + str(count))
            #plt.show()

            fig, ax = plt.subplots(1, 1)
            plt.plot(y_pred_train[:, 0], label='y_pred', alpha=0.5)
            plt.plot(y_train[:, 0], label='y_train', alpha=0.5)
            ax.set_title('y_pred (dist2goal) for fold: ' + str(count) + ' r2_score: ' + str(score_train))
            ax.set_xlabel('time in SAMPLES')
            plt.legend()
            plt.savefig(
                f'{savedir}/y_pred_vs_y_train_dist2goal_fold_' + str(count) + '2904.png')

            #plt.show()

            fig, ax = plt.subplots(1, 1)
            #do a smaller window for plotting purposes
            plt.plot(y_pred_train[0:120, 0], label='y_pred', alpha=0.5)
            plt.plot(y_train[0:120, 0], label='y_train', alpha=0.5)
            ax.set_title('y_pred (dist2goal) for fold: ' + str(count) + ' r2_score: ' + str(score_train))
            ax.set_xlabel('time in SAMPLES')
            plt.legend()
            plt.savefig(
                f'{savedir}/y_pred_vs_y_train_dist2goal_120_samples_fold_' + str(count) + '2904.png', dpi = 300, bbox_inches = 'tight')

            #plt.show()
            plt.close('all')






            ##now plot the shuffled data
            y_pred_shuffled = current_regressor_shuffled.predict(X_test_reduced_shuffled)
            fig, ax = plt.subplots(1, 1)
            ax.scatter(y_test, y_pred_shuffled, c='purple')
            ax.set_title('y_test vs y_pred for fold: ' + str(count) + ' shuffled, r2_score: ' + str(score_shuffled))
            plt.savefig(
                f'{savedir}/y_pred_vs_y_test_shuffled_fold_' + str(count) + '2904.png')
            #plt.show()

            fig, ax = plt.subplots(1, 1)
            plt.plot(y_pred_shuffled[0:120, 0], label='y_pred', alpha=0.5, c = 'purple')
            plt.plot(y_test[0:120, 0], label='y_test', alpha=0.5, c= 'darkorange')
            ax.set_title('y_pred (dist2goal) for fold: ' + str(count) + ' shuffled, r2_score: ' + str(score_shuffled))
            ax.set_xlabel('time in SAMPLES')
            plt.legend()
            plt.savefig(f'{savedir}/y_pred_vs_y_test_dist2goal_fold_' + str(
                count) + 'shuffled2904.png', dpi = 300, bbox_inches = 'tight')
            #plt.show()

            fig, ax = plt.subplots(1, 1)
            plt.plot(y_pred_shuffled[0:120, 0], label='y_pred', alpha=0.5, c = 'purple')
            plt.plot(y_test[0:120, 0], label='y_test', alpha=0.5, c= 'darkorange')
            ax.set_title('y_pred (dist2goal) for fold: ' + str(count) + ' shuffled, r2_score: ' + str(score_shuffled))
            ax.set_xlabel('time in SAMPLES')
            plt.legend()
            plt.savefig(f'{savedir}/y_pred_vs_y_test_dist2goal_first120samples_fold_' + str(
                count) + 'shuffled2904.png', dpi = 300, bbox_inches = 'tight')
            #plt.show()
            plt.close('all')


            count += 1


            # Evaluate the regressor
            score = current_regressor.score(X_test_reduced, y_test)
            scores.append(score)

        # Calculate mean score for the current parameter combination
        mean_score = np.mean(scores)
        mean_score_shuffled = np.mean(scores_shuffled)
        mean_score_train = np.mean(scores_train)
        mean_score_train_shuffled = np.mean(scores_train_shuffled)

        random_search_results.append((params, mean_score))
        random_search_results_train.append((params, mean_score_train))
        random_search_results_shuffled.append((params, mean_score_shuffled))
        random_search_results_train_shuffled.append((params, mean_score_train_shuffled))


    # Select the best parameters based on mean score
    best_params, _ = max(random_search_results, key=lambda x: x[1])
    _, mean_score_max = max(random_search_results, key=lambda x: x[1])
    _, mean_score_max_train = max(random_search_results_train, key=lambda x: x[1])
    _, mean_score_max_shuffled = max(random_search_results_shuffled, key=lambda x: x[1])
    _, mean_score_max_train_shuffled = max(random_search_results_train_shuffled, key=lambda x: x[1])


    return best_params, mean_score_max, mean_score_max_train, mean_score_max_shuffled, mean_score_max_train_shuffled



def load_previous_results(directory_of_interest):
    param_dict = {}
    score_dict = {}
    for rat_dir in ['C:/neural_data/rat_10/23-11-2021','C:/neural_data/rat_7/6-12-2019', 'C:/neural_data/rat_8/15-10-2019', 'C:/neural_data/rat_9/10-12-2021', 'C:/neural_data/rat_3/25-3-2019']:
        rat_id = rat_dir.split('/')[-2]
        param_directory = f'{rat_dir}/{directory_of_interest}'
        #find all the files in the directory
        files = os.listdir(param_directory)

        for window in [1000]:
            for bin_size in [250]:
                #find the file names
                for file in files:
                    if file.__contains__(f'{bin_size}bin_{window}windows'):
                        if file.__contains__('mean_score'):
                                score_dict[rat_id] = np.load(f'{param_directory}/{file}')
                        elif file.__contains__('params'):
                            with open(f'{param_directory}/{file}', 'rb') as f:
                                param_dict[rat_id] =  np.load(f'{param_directory}/{file}', allow_pickle=True)
    return param_dict, score_dict

def main():
    data_dir = 'C:/neural_data/rat_7/6-12-2019'
    param_dict, score_dict = load_previous_results(
        'dist2goal_results')
    big_df_savedir = 'C:/neural_data/r2_decoding_figures/umap/'
    var_regress = 'dist2goal'
    big_df = pd.DataFrame()
    #'C:/neural_data/rat_7/6-12-2019', 'C:/neural_data/rat_8/15-10-2019', 'C:/neural_data/rat_9/10-12-2021',
    for data_dir in ['C:/neural_data/rat_10/23-11-2021', 'C:/neural_data/rat_3/25-3-2019','C:/neural_data/rat_7/6-12-2019', 'C:/neural_data/rat_8/15-10-2019', 'C:/neural_data/rat_9/10-12-2021']:
        spike_dir = os.path.join(data_dir, 'physiology_data')
        dlc_dir = os.path.join(data_dir, 'positional_data')
        labels = np.load(f'{dlc_dir}/labels_1203_with_dist2goal_scale_data_False_zscore_data_False_overlap_False_window_size_250.npy')
        spike_data = np.load(f'{spike_dir}/inputs_overlap_False_window_size_250.npy')
        compare_data = np.load(f'{spike_dir}/inputs_250.npy')
        #check if spike_data is the same as compare_data
        if np.array_equal(spike_data, compare_data):
            print('Spike data is the same as compare data')

        spike_data_trial = spike_data
        data_dir_path = Path(data_dir)


        # check for neurons with constant firing rates
        tolerance = 1e-10  # or any small number that suits your needs
        if np.any(np.abs(np.std(spike_data_trial, axis=0)) < tolerance):
            print('There are neurons with constant firing rates')
            # remove those neurons
            spike_data_trial = spike_data_trial[:, np.abs(np.std(spike_data_trial, axis=0)) >= tolerance]
        # THEN DO THE Z SCORE
        # X_for_umap = scipy.stats.zscore(spike_data_trial, axis=0)
        X_for_umap = spike_data_trial
        if np.isnan(X_for_umap).any():
            print('There are nans in the data')

        # X_for_umap = scipy.ndimage.gaussian_filter(X_for_umap, 2, axes=0)


        labels_for_umap = labels[:, 0:6]
        labels_for_umap = scipy.ndimage.gaussian_filter(labels_for_umap, 2, axes=0)

        label_df = pd.DataFrame(labels_for_umap,
                                columns=['x', 'y', 'dist2goal', 'angle_sin', 'angle_cos', 'dlc_angle_zscore'])
        label_df['time_index'] = np.arange(0, label_df.shape[0])

        regressor = KNeighborsRegressor
        regressor_kwargs = {'n_neighbors': 70, 'metric': 'euclidean'}

        reducer = UMAP

        reducer_kwargs = {
            'n_components': 3,
        }

        regress = ['dist2goal']  # changing to two target variables

        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        now_day = datetime.now().strftime("%Y-%m-%d")
        filename = f'params_all_trials_randomizedsearchcv_250bin_1000windows_jake_fold_sinandcos_{now}.npy'
        filename_mean_score = f'mean_score_all_trials_randomizedsearchcv_250bin_1000windows_jake_fold_sinandcos_{now_day}.npy'
        save_dir_path = data_dir_path / 'umap_decomposition'/ 'dist2goal_results'
        save_dir_path.mkdir(parents=True, exist_ok=True)

        best_params,mean_score_max, mean_score_max_train, mean_score_max_shuffled, mean_score_max_train_shuffled  = train_and_test_on_umap_randcv(
            X_for_umap,
            label_df,
            regress,
            regressor,
            regressor_kwargs,
            reducer,
            reducer_kwargs, param_dict, rat_id = data_dir.split('/')[-2], plot_shaps=False
        )
        results_df = pd.DataFrame({'mean_score_max': [mean_score_max], 'mean_score_max_train': [mean_score_max_train],
                                   'mean_score_max_shuffled': [mean_score_max_shuffled],
                                   'mean_score_max_train_shuffled': [mean_score_max_train_shuffled],
                                   'best_params': [best_params], 'rat_id': [data_dir.split('/')[-2]]})
        # append to a big dataframe
        big_df = pd.concat([big_df, results_df], axis=0)
    big_df.to_csv(f'{big_df_savedir}/umap_decomposition_results_2904_{var_regress}.csv')

        # np.save(save_dir_path / filename, best_params)
        # np.save(save_dir_path / filename_mean_score, mean_score)


if __name__ == '__main__':
    #
    main()