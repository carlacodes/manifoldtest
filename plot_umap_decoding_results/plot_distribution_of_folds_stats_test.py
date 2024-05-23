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
from manifold_neural.helpers import visualisation
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
''' Modified from Jules Lebert's code
spks was a numpy arrray of size trial* timebins*neuron, and bhv is  a pandas dataframe where each row represents a trial, the trial is the index '''
import os
import scipy
import pickle as pkl
import shap
import plotly.graph_objects as go
os.environ['JOBLIB_TEMP_FOLDER'] = 'C:/tmp'
from sklearn.cross_decomposition import CCA
from sklearn.model_selection import StratifiedKFold
from helpers import tools
from scipy.stats import wilcoxon


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
    random_search_results_train = []
    random_search_results_shuffled = []
    random_search_results_train_shuffled = []

    # Create your custom folds
    n_timesteps = spks.shape[0]
    custom_folds = create_folds(n_timesteps, num_folds=10, num_windows=1000)
    # Example, you can use your custom folds here
    savedir = f'C:/neural_data/r2_decoding_figures/umap/{rat_id}/angle_rel_to_goal'
    #check if the directory exists
    if not os.path.exists(savedir):
        os.makedirs(savedir, exist_ok=True)

    for _ in range(1):  # 100 iterations for RandomizedSearchCV
        params = {key: (values) for key, values in param_grid.items()}
        regressor_kwargs.update(
            {k.replace('estimator__', ''): v for k, v in params.items() if k.startswith('estimator__')})
        reducer_kwargs.update({k.replace('reducer__', ''): v for k, v in params.items() if k.startswith('reducer__')})

        # Initialize the regressor with current parameters
        current_regressor = MultiOutputRegressor(regressor(**regressor_kwargs))
        current_regressor_shuffled = MultiOutputRegressor(regressor(**regressor_kwargs))

        # Initialize the reducer with current parameters
        current_reducer = reducer(**reducer_kwargs)
        current_reducer_shuffled = copy.deepcopy(current_reducer)
        count = 0
        scores = []
        scores_shuffled = []
        scores_train = []
        scores_train_shuffled = []

        spks_shuffled = copy.deepcopy(spks)
        # Shuffle along the first axis
        np.random.shuffle(spks_shuffled)
        # Transpose, shuffle and transpose back to shuffle along the second axis
        spks_shuffled = spks_shuffled.T
        np.random.shuffle(spks_shuffled)
        spks_shuffled = spks_shuffled.T


        for train_index, test_index in custom_folds:
            X_train, X_test = spks[train_index], spks[test_index]
            y_train, y_test = y[train_index], y[test_index]

            X_train_shuffled, X_test_shuffled = spks_shuffled[train_index], spks_shuffled[test_index]

            # Apply dimensionality reduction
            X_train_reduced = current_reducer.fit_transform(X_train)
            X_test_reduced = current_reducer.transform(X_test)

            # Fit the regressor
            current_regressor.fit(X_train_reduced, y_train)

            X_train_reduced_shuffled = current_reducer_shuffled.fit_transform(X_train_shuffled)

            X_test_reduced_shuffled = current_reducer_shuffled.transform(X_test_shuffled)

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
            if plot_shaps and count == 8 and rat_id == 'rat_3':
                #just plot on the first fold now as an example
                plot_kneighborsregressor_splits(current_reducer, current_regressor, X_test_reduced, X_train_reduced, y_train, y_test, save_dir_path=savedir, fold_num=count, rat_id=rat_id)

            colormap = visualisation.colormap_2d()
            data_x_c = np.interp(y_test[:,0], (y_test[:,0].min(), y_test[:,0].max()), (0, 255)).astype(
                int)
            data_y_c = np.interp(y_test[:,1], (y_test[:,1].min(), y_test[:,1].max()), (0, 255)).astype(
                int)
            color_data=colormap[data_x_c, data_y_c]

            #get the actual angle relative to goal and create a 1d color map by takin the inverse sin
            #and inverse cos
            actual_angle = np.arcsin(y_pred[:, 0])

            # Initialize a CCA model
            cca = CCA(n_components=1)

            # Fit the model with your data
            cca.fit(X_test_reduced, y_test)

            # Now, cca.x_weights_ stores the coefficients for each UMAP embedding
            print("Coefficients: ", cca.x_weights_)


            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            sc = ax.scatter(X_test_reduced[:, 0], X_test_reduced[:, 1], X_test_reduced[:, 2], c=actual_angle, cmap='viridis')
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            ax.set_zlabel('UMAP 3')
            #add a color bar
            cbar = plt.colorbar(sc, ax=ax)
            ax.set_title('UMAP test embeddings color-coded by head angle rel. \n  to goal for fold: ' + str(count) + 'rat id:' +str(rat_id))
            plt.savefig(f'{savedir}/umap_embeddings_fold_' + str(count) + '.png', dpi=300, bbox_inches='tight')


            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            sc = ax.scatter(X_test_reduced_shuffled[:, 0], X_test_reduced_shuffled[:, 1], X_test_reduced_shuffled[:, 2], c=actual_angle, cmap='magma')
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            ax.set_zlabel('UMAP 3')
            #add a color bar
            cbar = plt.colorbar(sc, ax=ax)
            ax.set_title('UMAP shuffled test embeddings color-coded by head angle rel. \n  to goal for fold: ' + str(count) + 'rat id:' +str(rat_id))
            plt.savefig(f'{savedir}/umap_embeddings_SHUFFLED_fold_' + str(count) + '.png', dpi=300, bbox_inches='tight')

            fig, ax = plt.subplots(1, 1)
            ax.plot(X_test_reduced[0:120, 0], label = 'UMAP 1', alpha = 0.5)
            ax.plot(X_test_reduced_shuffled[0:120, 0], label = 'UMAP 1 SHUFFLED', alpha = 0.5)
            ax.set_title('UMAP 1 vs UMAP 1 SHUFFLED for fold, head angle (allocentric): ' + str(count))
            ax.set_xlabel('time in SAMPLES')
            plt.legend()
            plt.savefig(f'{savedir}/umap1_vs_umap1_shuffled_fold_' + str(count) + '.png', dpi = 300, bbox_inches = 'tight')

            # Create a 3D scatter plot
            fig = go.Figure(data=[go.Scatter3d(
                x=X_test_reduced[:, 0],
                y=X_test_reduced[:, 1],
                z=X_test_reduced[:, 2],
                mode='markers',
                marker=dict(
                    size=6,
                    color=actual_angle,  # set color to prediction values
                    colorscale='Viridis',  # choose a colorscale
                    opacity=0.8
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
            fig.update_layout(title_text='UMAP test embeddings color-coded by angle for fold: ' + str(
                count) + ' rat id: ' + str(rat_id))

            # Save the figure as an HTML file
            fig.write_html(f'{savedir}/umap_embeddings_fold_interactive_' + str(count) + '.html')


            fig = go.Figure(data=[go.Scatter3d(
                x=X_test_reduced_shuffled[:, 0],
                y=X_test_reduced_shuffled[:, 1],
                z=X_test_reduced_shuffled[:, 2],
                mode='markers',
                marker=dict(
                    size=6,
                    color=actual_angle,  # set color to prediction values
                    colorscale='Magma',  # choose a colorscale
                    opacity=0.8
                )
            )])

            # Set plot title
            fig.update_layout(title_text='UMAP test embeddings color-coded by angle (goal-centred), SHUFFLED, for fold: ' + str(
                count) + ' rat id: ' + str(rat_id))

            # Save the figure as an HTML file
            fig.write_html(f'{savedir}/umap_embeddings_fold_interactive_SHUFFLED' + str(count) + '.html')
            # Create a 3D line plot
            # fig = go.Figure(data=[go.Scatter3d(
            #     x=X_test_reduced[:, 0],
            #     y=X_test_reduced[:, 1],
            #     z=X_test_reduced[:, 2],
            #     mode='lines',
            #     line=dict(
            #         color=actual_angle,  # set color to prediction values
            #         colorscale='Viridis',  # choose a colorscale
            #         width=2
            #     )
            # )])
            #
            # # Set plot title
            # fig.update_layout(title_text='UMAP test embeddings color-coded by angle for fold: ' + str(
            #     count) + ' rat id: ' + str(rat_id))
            #
            # # Save the figure as an HTML file
            # fig.write_html(f'{savedir}/umap_embeddings_fold_interative_lines_' + str(count) + '.html')

            # Show the figure
            # fig.show()

            #plot the color map
            fig, ax = plt.subplots(1, 1)
            ax.imshow(colormap)
            ax.set_title('Color map for angle relative to goal')
            plt.savefig(f'{savedir}/color_map_fold_{count}.png')
            #plt.show()

            fig, ax = plt.subplots(1, 1)
            ax.scatter(y_test, y_pred)
            ax.set_title('y_test vs y_pred for fold: ' + str(count))
            #plt.show()

            fig, ax = plt.subplots(1, 1)
            plt.plot(y_pred[:, 0], label='y_pred', alpha=0.5)
            plt.plot(y_test[:, 0], label='y_test', alpha=0.5)
            ax.set_title('y_pred -- sin angle rel. to goal for fold: ' + str(count)+ ' r2_score: ' + str(score))
            ax.set_xlabel('time in SAMPLES')
            plt.savefig(
                f'{savedir}/y_pred_vs_y_test_sinrelativetogoal_fold_' + str(count) + '.png')
            #plt.show()

            fig, ax = plt.subplots(1, 1)
            plt.plot(y_pred[:, 1], label='y_pred', alpha=0.5)
            plt.plot(y_test[:, 1], label='y_test', alpha=0.5)
            ax.set_title('y_pred -- cos angle rel. to goal for fold: ' + str(count) + ' r2_score: ' + str(score))
            ax.set_xlabel('time in SAMPLES')
            plt.legend()
            plt.savefig(
                f'{savedir}/y_pred_vs_y_test_cos_reltogoal_fold_' + str(count) + '.png')
            #get the actual angle
            actual_angle = np.arcsin(y_test[0:120, 0])
            actual_angle_pred = np.arcsin(y_pred[0:120, 0])
            fig, ax = plt.subplots(1, 1)
            ax.plot(actual_angle, label='actual angle', alpha=0.5)
            ax.plot(actual_angle_pred, label='predicted angle', alpha=0.5)
            ax.set_title('Actual and predicted angle rel. to goal for fold: ' + str(count) + ' r2_score: ' + str(score))
            ax.set_xlabel('time in SAMPLES')
            plt.legend()
            plt.savefig(
                f'{savedir}/actual_vs_predicted_converted_angle_fold_' + str(count) + '.png', dpi = 300,  bbox_inches='tight')
            plt.close('all')

            #do the same for the shuffled data



            #plt.show()


            #do the same for the train data
            y_pred_train = current_regressor.predict(X_train_reduced)
            fig, ax = plt.subplots(1, 1)
            ax.scatter(y_train, y_pred_train)
            ax.set_title('y_train vs y_pred for fold: ' + str(count))
            #plt.show()

            fig, ax = plt.subplots(1, 1)
            plt.plot(y_pred_train[:, 0], label='y_pred', alpha=0.5)
            plt.plot(y_train[:, 0], label='y_test', alpha=0.5)
            ax.set_title('y_pred cos angle rel. to goal  for fold: ' + str(count) + ' r2_score: ' + str(score_train))
            ax.set_xlabel('time in SAMPLES')
            plt.legend()
            plt.savefig(
                f'{savedir}/y_pred_vs_y_train_cos_reltogoal_fold_' + str(count) + '.png')

            #plt.show()

            fig, ax = plt.subplots(1, 1)
            plt.plot(y_pred_train[:, 1], label='y_pred', alpha=0.5)
            plt.plot(y_train[:, 1], label='y_test', alpha=0.5)
            ax.set_title('y_pred --  sin angle rel. to goal--  for fold: ' + str(count) + ' r2_score: ' + str(score_train))
            ax.set_xlabel('time in SAMPLES')
            plt.legend()
            plt.savefig(
                f'{savedir}/y_pred_vs_y_train_sin_rel_to_goal_fold_' + str(count) + '.png')
            #plt.show()




            ##now plot the shuffled data
            y_pred_shuffled = current_regressor_shuffled.predict(X_test_reduced_shuffled)
            fig, ax = plt.subplots(1, 1)
            ax.scatter(y_test, y_pred_shuffled, c='purple')
            ax.set_title('y_test vs y_pred for fold: ' + str(count) + ' shuffled, r2_score: ' + str(score_shuffled))
            plt.savefig(
                f'{savedir}/y_pred_vs_y_test_shuffled_fold_' + str(count) + '.png')
            #plt.show()

            fig, ax = plt.subplots(1, 1)
            plt.plot(y_pred_shuffled[:, 0], label='y_pred', alpha=0.5, c = 'purple')
            plt.plot(y_test[:, 0], label='y_test', alpha=0.5, c='darkorange')
            ax.set_title('y_pred (sin head angle rel. to goal) for fold: ' + str(count) + ' shuffled, r2_score: ' + str(score_shuffled))
            ax.set_xlabel('time in SAMPLES')
            plt.legend()
            plt.savefig(f'{savedir}/y_pred_vs_y_test_sin_fold_' + str(
                count) + 'shuffled.png')
            #plt.show()

            fig, ax = plt.subplots(1, 1)
            plt.plot(y_pred_shuffled[:, 1], label='y_pred',c='purple',  alpha=0.5)
            plt.plot(y_test[:, 1], label='y_test', c='darkorange', alpha=0.5)
            ax.set_title('y_pred (cos head angle rel. to goal) \n for fold: ' + str(count) + ' shuffled, r2_score: ' + str(score_shuffled))
            ax.set_xlabel('time in SAMPLES')
            plt.legend()
            plt.savefig(f'{savedir}/y_pred_vs_y_test_cos_fold_' + str(
                count) + 'shuffled.png')

            actual_angle_pred_shuffled = np.arcsin(y_pred_shuffled[:,0])
            fig, ax = plt.subplots(1, 1)
            ax.plot(actual_angle[0:120], label='actual angle', alpha=0.5, c = 'purple')
            ax.plot(actual_angle_pred_shuffled[0:120], label='predicted angle', alpha=0.5, c='darkorange')
            ax.set_title('Actual vs predicted angle rel. to goal \n for fold: ' + str(count) + ' shuffled, r2_score: ' + str(score_shuffled))
            ax.set_xlabel('time in SAMPLES')
            plt.legend()
            plt.savefig(
                f'{savedir}/actual_vs_predicted_converted_angle_fold_' + str(count) + 'shuffled.png', dpi = 300,  bbox_inches='tight')



            plt.close('all')



            count += 1


            # Evaluate the regressor
            score = current_regressor.score(X_test_reduced, y_test)
            scores.append(score)

        # Calculate mean score for the current parameter combination
        mean_score = np.mean(scores)
        mean_score_train = np.mean(scores_train)

        mean_score_shuffled = np.mean(scores_shuffled)
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
    # 'C:/neural_data/rat_3/25-3-2019'
    for rat_dir in ['C:/neural_data/rat_10/23-11-2021','C:/neural_data/rat_7/6-12-2019', 'C:/neural_data/rat_8/15-10-2019', 'C:/neural_data/rat_9/10-12-2021', 'C:/neural_data/rat_3/25-3-2019']:
        rat_id = rat_dir.split('/')[-2]
        param_directory = f'{rat_dir}/{directory_of_interest}'
        #find all the files in the directory
        files = os.listdir(param_directory)

        for window in [1000]:
            for bin_size in [250]:
                #find the file names
                for file in files:
                    if file.__contains__(f'{window}windows'):
                        if file.__contains__('mean_score'):
                                score_dict[rat_id] = np.load(f'{param_directory}/{file}')
                        elif file.__contains__('params'):
                            with open(f'{param_directory}/{file}', 'rb') as f:
                                param_dict[rat_id] =  np.load(f'{param_directory}/{file}', allow_pickle=True)
    return param_dict, score_dict

def plot_kneighborsregressor_splits(reducer, knn, X_test_reduced, X_train_reduced, y_train, y_test, save_dir_path=None, fold_num=None, rat_id = 'None'):
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
    shap_values = explainer.shap_values(X_test_reduced_sampled, n_jobs=n_jobs)

    # Visualize the SHAP values
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    shap.summary_plot(shap_values[0], X_test_reduced_sampled, plot_type='dot', show = False)
    plt.title(f'SHAP values for the test data, rat id: {rat_id}')
    plt.xlabel('SHAP value (impact on sin head angle relative to goal)')
    plt.ylabel('UMAP feature')
    plt.savefig(f'{save_dir_path}/shap_values_sin_fold_{fold_num}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    shap.summary_plot(shap_values[1], X_test_reduced_sampled, plot_type='dot', show = False)
    plt.title(f'SHAP values for the test data, rat id: {rat_id}')
    plt.xlabel('SHAP value (impact on cos head angle relative to goal)')
    plt.ylabel('UMAP feature')
    plt.savefig(f'{save_dir_path}/shap_values_cos_fold_{fold_num}.png', dpi=300, bbox_inches='tight')
    plt.close('all')




    return

def run_ks_test_on_distributions(data_dir, param_dict, score_dict, big_df_savedir):
    df_across_windows = pd.DataFrame()
    df_across_windows_angle = pd.DataFrame()
    for data_dir in ['C:/neural_data/rat_7/6-12-2019', 'C:/neural_data/rat_10/23-11-2021',
                     'C:/neural_data/rat_8/15-10-2019', 'C:/neural_data/rat_9/10-12-2021',
                     'C:/neural_data/rat_3/25-3-2019']:
        for window_size in [250]:
            rat_id = data_dir.split('/')[-2]
            spike_dir = os.path.join(data_dir, 'physiology_data')
            dlc_dir = os.path.join(data_dir, 'positional_data')
            labels = np.load(
                f'{dlc_dir}/labels_{window_size}.npy')
            spike_data = np.load(f'{spike_dir}/inputs_{window_size}.npy')

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

            labels_for_umap = labels[:, 0:5]
            labels_for_umap = scipy.ndimage.gaussian_filter(labels_for_umap, 2, axes=0)

            label_df = pd.DataFrame(labels_for_umap,
                                    columns=[ 'x', 'y', 'dist2goal', 'hd', 'relative_direction_to_goal'])
            label_df['time_index'] = np.arange(0, label_df.shape[0])
            label_df['angle_sin'] = np.sin(label_df['hd'])
            label_df['angle_cos'] = np.cos(label_df['hd'])

            # z-score x and y

            xy_labels = np.concatenate(label_df[['x', 'y']].values, axis=0)
            label_df['x_zscore'] = (label_df['x'] - xy_labels.mean()) / xy_labels.std()
            label_df['y_zscore'] = (label_df['y'] - xy_labels.mean()) / xy_labels.std()
            n_timesteps = X_for_umap.shape[0]

            # increment over number of windows to get the minimum number of windows where p-value is non-significant

            # custom_folds = create_folds(n_timesteps, num_folds=10, num_windows=1000)
            time_range_for_testing = (0, 1000)

            big_results_df = pd.DataFrame()
            big_results_df_angle = pd.DataFrame()
            for i in range(10, 10000, 10):
                try:
                    custom_folds_test = create_folds(n_timesteps, num_folds=10, num_windows=i)
                except Exception as e:
                    print('Error in creating the folds, error is: ', e)
                    continue
                results_df = pd.DataFrame()
                results_df_angle = pd.DataFrame()
                for j, (train_index, test_index) in enumerate(custom_folds_test):
                    ks_test_result = scipy.stats.kstest(label_df['x'].values[train_index],
                                                        label_df['x'].values[test_index])
                    ks_test_result_angle_sin = scipy.stats.kstest(label_df['angle_sin'].values[train_index],
                                                                  label_df['angle_cos'].values[test_index])

                    ks_statistic = ks_test_result[0]
                    p_val = ks_test_result[1]

                    ks_statistic_angle_sin = ks_test_result_angle_sin[0]
                    p_val_angle_sin = ks_test_result_angle_sin[1]

                    ks_test_result_y = scipy.stats.kstest(label_df['y'].values[train_index],
                                                          label_df['y'].values[test_index])
                    ks_test_result_cos = scipy.stats.kstest(label_df['angle_cos'].values[train_index],
                                                            label_df['angle_cos'].values[test_index])

                    ks_statistic_angle_cos = ks_test_result_cos[0]
                    p_val_angle_cos = ks_test_result_cos[1]

                    # take the mean of the p-values
                    p_val_y = ks_test_result_y[1]
                    ks_statistic_y = ks_test_result_y[0]

                    p_val_rounded = np.round(np.mean([p_val, p_val_y]), 4)
                    ks_statistic = np.round(np.mean([ks_statistic, ks_statistic_y]), 4)

                    p_val_rounded_angle = np.round(np.mean([p_val_angle_sin, p_val_angle_cos]), 4)
                    ks_statistic_angle = np.round(np.mean([ks_statistic_angle_sin, ks_statistic_angle_cos]), 4)
                    # append to the results dataframe
                    trial_data_angle = {'window_size': i, 'p_value': p_val_rounded_angle, 'ks_stat': ks_statistic_angle,
                                        'fold_number': j}

                    # appebd to the results dataframe
                    trial_data = {'window_size': i, 'p_value': p_val_rounded, 'ks_stat': ks_statistic, 'fold_number': j}
                    results_df = pd.concat([results_df, pd.DataFrame(trial_data, index=[i])])

                    results_df_angle = pd.concat([results_df_angle, pd.DataFrame(trial_data_angle, index=[i])])
                # calculate the mean p-value for the window size
                # take the mean of results_df
                mean_p_val = results_df['p_value'].mean()
                mean_t_stat = results_df['ks_stat'].mean()

                mean_p_val_angle = results_df_angle['p_value'].mean()
                mean_t_stat_angle = results_df_angle['ks_stat'].mean()

                # append to the big results dataframe
                window_size_data = {'num_windows': i, 'mean_p_value': mean_p_val, 'mean_t_stat': mean_t_stat}
                window_size_data_angle = {'num_windows': i, 'mean_p_value': mean_p_val_angle,
                                          'mean_t_stat': mean_t_stat_angle}

                big_results_df = pd.concat([big_results_df, pd.DataFrame(window_size_data, index=[i])])
                big_results_df_angle = pd.concat(
                    [big_results_df_angle, pd.DataFrame(window_size_data_angle, index=[i])])
            # plot the p-values and t-stats against the window size
            # apply a smoothing filter
            # big_results_df['mean_p_value'] = big_results_df['mean_p_value'].rolling(window=10).mean()
            # big_results_df['mean_t_stat'] = big_results_df['mean_t_stat'].rolling(window=10).mean()
            # calculate the index for when the p-value is consistently above 0.2
            # get the index where the p-value is consistently above 0.2

            # first threshold for where the p-value is consistently above 0.2
            threshold = 0.05
            # get the index where the p-value is consistently above 0.2
            index_above_threshold = big_results_df[big_results_df['mean_p_value'] > threshold]
            index_above_threshold_angle = big_results_df_angle[big_results_df_angle['mean_p_value'] > threshold]

            # get the applicable thresholds
            threshold_indices = index_above_threshold.index
            threshold_indices_angle = index_above_threshold_angle.index
            diff = np.diff(threshold_indices)
            diff_angle = np.diff(threshold_indices_angle)
            # find the minimum index where every difference after is 10
            index_saved = None
            for i, val in enumerate(diff):
                if val == 10 and index_saved == None:
                    index_saved = i
                if val != 10 and index_saved == None:
                    index_saved = None
                elif val != 10 and i > index_saved:
                    index_saved = None
            first_index = index_saved

            for i, val in enumerate(diff_angle):
                if val == 10 and index_saved == None:
                    index_saved = i
                if val != 10 and index_saved == None:
                    index_saved = None
                elif val != 10 and i > index_saved:
                    index_saved = None
            first_index_angle = index_saved

            corresponding_num_windows = threshold_indices[first_index]
            corresponding_num_windows_angle = threshold_indices_angle[first_index_angle]

            fig, ax = plt.subplots(1, 1)
            ax.plot(big_results_df['num_windows'], big_results_df['mean_p_value'], label='mean p-value')
            ax.set_title(f'Mean p-value vs num_windows for rat: \n  {data_dir_path} and window size: {window_size}')
            ax.set_xlabel('num_windows')
            ax.set_ylabel('mean p-value')
            # append to an animal and

            plt.savefig(f'{big_df_savedir}/mean_p_value_vs_num_windows_rat_id_{rat_id}_window_size_{window_size}.png',
                        dpi=300, bbox_inches='tight')
            # plt.show()

            fig, ax = plt.subplots(1, 1)
            ax.plot(big_results_df_angle['num_windows'], big_results_df_angle['mean_p_value'], label='mean p-value')
            ax.set_title(
                f'Mean p-value vs num_windows for angle, rat: \n  {data_dir_path} and window size: {window_size}')
            ax.set_xlabel('num_windows')
            ax.set_ylabel('mean p-value')
            plt.savefig(
                f'{big_df_savedir}/mean_p_value_vs_num_windows_angle_rat_id_{rat_id}_window_size_{window_size}.png',
                dpi=300, bbox_inches='tight')
            # append to an animal and
            plt.close('all')
            # append to the big dataframe
            big_results_df['rat_id'] = rat_id
            big_results_df['window_size'] = window_size
            big_results_df['minimum_number_windows'] = corresponding_num_windows
            df_across_windows = pd.concat([df_across_windows, big_results_df])

            big_results_df_angle['rat_id'] = rat_id
            big_results_df_angle['window_size'] = window_size
            big_results_df_angle['minimum_number_windows'] = corresponding_num_windows_angle
            df_across_windows_angle = pd.concat([df_across_windows_angle, big_results_df_angle])
    # get the mean minimum number of windows
    df_across_windows['mean_minimum_number_windows'] = df_across_windows.groupby('rat_id')[
        'minimum_number_windows'].transform('mean')
    df_across_windows['mean_minimum_number_windows_by_windowsize'] = df_across_windows.groupby('window_size')[
        'minimum_number_windows'].transform('mean')
    df_across_windows['mean_minimum_number_windows_across_rats_and_windowsize'] = df_across_windows[
        'mean_minimum_number_windows'].mean()

    df_across_windows_angle['mean_minimum_number_windows_across_rats'] = df_across_windows_angle.groupby('rat_id')[
        'minimum_number_windows'].transform('mean')
    df_across_windows_angle['mean_minimum_number_windows_across_windowsize'] = \
    df_across_windows_angle.groupby('window_size')['minimum_number_windows'].transform('mean')
    df_across_windows_angle['mean_minimum_number_windows_across_rats_and_windowsize'] = df_across_windows_angle[
        'minimum_number_windows'].mean()

    np.unique(df_across_windows['mean_minimum_number_windows_across_rats_and_windowsize'])
    np.unique(df_across_windows_angle['mean_minimum_number_windows_across_rats_and_windowsize'])
    np.unique(df_across_windows_angle['mean_minimum_number_windows_across_windowsize'])

    # export to csv
    df_across_windows.to_csv(f'{big_df_savedir}/mean_p_value_vs_window_size_across_rats.csv')
    df_across_windows_angle.to_csv(f'{big_df_savedir}/mean_p_value_vs_window_size_across_rats_angle.csv')
    return df_across_windows


def run_stratified_kfold_test():
    # Define the number of cells per dimension
    df_big = pd.DataFrame()
    df_big_angle = pd.DataFrame()

    for data_dir in ['C:/neural_data/rat_7/6-12-2019', 'C:/neural_data/rat_10/23-11-2021',
                     'C:/neural_data/rat_8/15-10-2019', 'C:/neural_data/rat_9/10-12-2021',
                     'C:/neural_data/rat_3/25-3-2019']:
        rat_id = data_dir.split('/')[-2]
        df_big_rat = pd.DataFrame()
        df_big_rat_angle = pd.DataFrame()
        for window_size in [250]:
            rat_id = data_dir.split('/')[-2]
            spike_dir = os.path.join(data_dir, 'physiology_data')
            dlc_dir = os.path.join(data_dir, 'positional_data')
            labels = np.load(f'{dlc_dir}/labels_{window_size}_scale_to_angle_range_True.npy')
            col_list = np.load(f'{dlc_dir}/col_names_{window_size}_scale_to_angle_range_True.npy')
            spike_data = np.load(f'{spike_dir}/inputs_{window_size}.npy')

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

            labels_for_umap = labels[:, 0:5]
            # labels_for_umap = scipy.ndimage.gaussian_filter(labels_for_umap, 2, axes=0)

            label_df = pd.DataFrame(labels_for_umap,
                                    columns=[ 'x', 'y', 'dist2goal', 'hd', 'relative_direction_to_goal'])
            label_df['time_index'] = np.arange(0, label_df.shape[0])
            label_df['angle_sin'] = np.sin(label_df['hd'])
            label_df['angle_cos'] = np.cos(label_df['hd'])

            # z-score x and y
            window_size_df = pd.DataFrame()
            window_size_df_angle = pd.DataFrame()
            xy_labels = np.concatenate(label_df[['x', 'y']].values, axis=0)
            label_df['x_zscore'] = (label_df['x'] - xy_labels.mean()) / xy_labels.std()
            label_df['y_zscore'] = (label_df['y'] - xy_labels.mean()) / xy_labels.std()
            n_timesteps = X_for_umap.shape[0]
            n_cells = 9

            # Create a new column 'region' that represents the cell each data point belongs to
            label_df['region'] = pd.cut(label_df['x'], n_cells, labels=False) + \
                           pd.cut(label_df['y'], n_cells, labels=False) * n_cells + \
                           pd.cut(label_df['angle_sin'], n_cells, labels=False) * n_cells * n_cells + \
                           pd.cut(label_df['angle_cos'], n_cells, labels=False) * n_cells * n_cells * n_cells
            # No nw, you can use 'region' for Stratified Cross Validation
            skf = StratifiedKFold(n_splits=10, shuffle = False)
            results_df = pd.DataFrame()
            results_df_angle = pd.DataFrame()
            for j, (train_index, test_index) in enumerate(skf.split(label_df, label_df['region'])):
                # These are your train/test indices
                print(" length of TRAIN:", len(train_index), "length of TEST:", len(test_index))
                #plot the distribution of the indexes
                #find the number of contiguous indices in the train and test set
                conseq_train = np.diff(train_index)
                conseq_test = np.diff(test_index)
                conseq_train = np.where(conseq_train == 1)[0]
                conseq_test = np.where(conseq_test == 1)[0]
                #map back to where these indices are in the original dataframe
                conseq_train_df = train_index[conseq_train]
                conseq_test_df = test_index[conseq_test]

                #plot the distribution of the contiguous indices
                # fig, ax = plt.subplots(1, 1)
                # plt.hist(conseq_train, bins = 100, alpha = 0.5, label = 'train')
                # plt.hist(conseq_test, bins = 100, alpha = 0.5, label = 'test')
                # plt.xlabel('index')
                # plt.ylabel('count')
                # plt.legend()
                # plt.title(f'Distribution of contiguous indices for fold number: {j} rat id: {rat_id}')
                # savefig_dir = f'{data_dir_path}/figures'
                # if not os.path.exists(savefig_dir):
                #     os.makedirs(savefig_dir, exist_ok=True)
                # plt.savefig(f'{savefig_dir}/contiguous_indices_fold_number_{j}_ratid_{rat_id}.png')
                # plt.show()

                fig, ax = plt.subplots(1, 1)
                plt.hist(train_index, bins = 100, alpha = 0.5, label = 'train')
                plt.hist(test_index, bins = 100, alpha = 0.5, label = 'test')
                plt.xlabel('index')
                plt.ylabel('count')
                plt.legend()
                plt.title(f'Distribution of train and test indices for fold number: {j} rat id: {rat_id}')
                savefig_dir = f'{data_dir_path}/figures'
                if not os.path.exists(savefig_dir):
                    os.makedirs(savefig_dir, exist_ok=True)
                plt.savefig(f'{savefig_dir}/train_test_distribution_fold_number_{j}_ratid_{rat_id}.png')
                plt.show()
                plt.close('all')
                y_train, y_test = label_df.iloc[train_index], label_df.iloc[test_index]

                #run a ks test on the distributions
                ks_test_result_x = scipy.stats.ks_2samp(y_train['x'], y_test['x'])
                ks_test_result_y = scipy.stats.ks_2samp(y_train['y'], y_test['y'])
                ks_test_result_angle_sin = scipy.stats.ks_2samp(y_train['angle_sin'], y_test['angle_sin'])
                ks_test_result_angle_cos = scipy.stats.ks_2samp(y_train['angle_cos'], y_test['angle_cos'])

                ks_statistic_x = ks_test_result_x[0]
                p_val_x = ks_test_result_x[1]

                ks_statistic_y = ks_test_result_y[0]
                p_val_y = ks_test_result_y[1]

                ks_statistic_angle_sin = ks_test_result_angle_sin[0]
                p_val_angle_sin = ks_test_result_angle_sin[1]

                ks_statistic_angle_cos = ks_test_result_angle_cos[0]
                p_val_angle_cos = ks_test_result_angle_cos[1]


                #take the mean of the p-values
                p_val = np.mean([p_val_x, p_val_y])
                ks_statistic = np.mean([ks_statistic_x, ks_statistic_y])

                p_val_angle = np.mean([p_val_angle_sin, p_val_angle_cos])
                ks_statistic_angle = np.mean([ks_statistic_angle_sin, ks_statistic_angle_cos])
                #append to a list
                results = {'p_value': p_val, 'ks_stat': ks_statistic}
                results_angle = {'p_value': p_val_angle, 'ks_stat': ks_statistic_angle}
                results_df = pd.concat([results_df, pd.DataFrame(results, index=[0])])
                results_df_angle = pd.concat([results_df_angle, pd.DataFrame(results_angle, index=[0])])
                #append to a dataframe
            #take the mean
            indiv_window_size_df = results_df.mean()
            indiv_window_size_df_angle = results_df_angle.mean()

            window_size_df = pd.concat([window_size_df,indiv_window_size_df])
            window_size_df_angle = pd.concat([window_size_df_angle,indiv_window_size_df_angle])
         #append to the big dataframe
        window_size_df['rat_id'] = rat_id
        window_size_df['window_size'] = window_size
        window_size_df_angle['rat_id'] = rat_id
        window_size_df_angle['window_size'] = window_size
        df_big_rat = pd.concat([df_big_rat, window_size_df])
        df_big_rat_angle = pd.concat([df_big_rat_angle, window_size_df_angle])
        df_big = pd.concat([df_big, df_big_rat])
        df_big_angle = pd.concat([df_big_angle, df_big_rat_angle])
        #





    return df_big, df_big_angle


def run_ks_test_on_distributions_3d_grid(data_dir, param_dict, score_dict, big_df_savedir, scale_to_angle_range = False):
    df_across_windows = pd.DataFrame()
    df_across_windows_angle = pd.DataFrame()
    for data_dir in ['C:/neural_data/rat_7/6-12-2019', 'C:/neural_data/rat_10/23-11-2021',
                     'C:/neural_data/rat_8/15-10-2019', 'C:/neural_data/rat_9/10-12-2021',
                     'C:/neural_data/rat_3/25-3-2019']:
        for window_size in [100]:
            rat_id = data_dir.split('/')[-2]
            spike_dir = os.path.join(data_dir, 'physiology_data')
            dlc_dir = os.path.join(data_dir, 'positional_data')
            labels = np.load(f'{dlc_dir}/labels_{window_size}_scale_to_angle_range_{scale_to_angle_range}.npy')
            col_list = np.load(f'{dlc_dir}/col_names_{window_size}_scale_to_angle_range_True.npy')
            spike_data = np.load(f'{spike_dir}/inputs_10052024_{window_size}.npy')
            spike_data_copy = copy.deepcopy(spike_data)
            tolerance = 1e-10  # or any small number that suits your needs
            if np.any(np.abs(np.std(spike_data_copy, axis=0)) < tolerance):
                print('There are neurons with constant firing rates')
                # remove those neurons
                spike_data_copy = spike_data_copy[:, np.abs(np.std(spike_data_copy, axis=0)) >= tolerance]

            # spike_data_trial = spike_data
            # data_dir_path = Path(data_dir)

            X_for_umap, removed_indices = tools.apply_lfads_smoothing(spike_data_copy)
            X_for_umap = scipy.stats.zscore(X_for_umap, axis=0)

            labels_for_umap = labels
            # remove the indices
            labels_for_umap = np.delete(labels_for_umap, removed_indices, axis=0)

            label_df = pd.DataFrame(labels_for_umap,
                                    columns=col_list)
            label_df['time_index'] = np.arange(0, label_df.shape[0])


            big_results_df = pd.DataFrame()
            big_results_df_angle = pd.DataFrame()
            n_timesteps = X_for_umap.shape[0]
            label_df['position'] = list(zip(label_df['x'], label_df['y']))
            label_df['hd_goal'] = np.arcsin(label_df['sin_relative_direction'])

            # Sort by position and angle
            # sorted_df = label_df.sort_values(by=['position', 'hd_goal'])

            n_cells = 8
            # Create a new column 'region' that represents the cell each data point belongs to
            label_df['region'] = pd.cut(label_df['x'], n_cells, labels=False) + \
                                 pd.cut(label_df['y'], n_cells, labels=False) * n_cells + \
                                 pd.cut(label_df['hd_goal'], n_cells, labels=False) * n_cells * n_cells
            min_region = label_df['region'].min()
            max_region = label_df['region'].max()
            from scipy.stats import ks_2samp

            for i in range(10, 10000, 10):
                try:
                    custom_folds_test = create_folds(n_timesteps, num_folds=10, num_windows=i)
                except Exception as e:
                    print('Error in creating the folds, error is: ', e)
                    continue
                results_df = pd.DataFrame()
                results_df_angle = pd.DataFrame()
                for j, (train_index, test_index) in enumerate(custom_folds_test):

                    # create a 3d grid
                    y_train, y_test = label_df.iloc[train_index], label_df.iloc[test_index]
                    # for each region, calculate the ks test for angle
                    ks_results = {}

                    train_grouped = y_train.groupby('region').size()

                    test_grouped = y_test.groupby('region').size()
                    #for regions that are missing in the range of min_region and max_region, add a zero
                    for region in range(min_region, max_region+1):
                        if region not in train_grouped:
                            train_grouped[region] = 0
                        if region not in test_grouped:
                            test_grouped[region] = 0

                    #convert train_grouped and test_grouped to an array for the ks test
                    train_grouped_for_testing = train_grouped.values
                    #normalise
                    train_grouped_for_testing = train_grouped_for_testing / np.max(train_grouped_for_testing)

                    test_grouped_for_testing = test_grouped.values
                    test_grouped_for_testing = test_grouped_for_testing / np.max(test_grouped_for_testing)
                    # fig,ax = plt.subplots(1, 1)
                    # ax.plot(train_grouped_for_testing, label='train')
                    # ax.plot(test_grouped_for_testing, label='test')
                    # ax.set_title(f'Number of samples in each region for fold number: {j} rat id: {rat_id}')
                    # ax.set_xlabel('region')
                    # ax.set_ylabel('count')
                    # ax.legend()
                    # plt.show()
                    plt.close('all')
                    # ks_results = ks_2samp(train_grouped_for_testing, test_grouped_for_testing)
                    len(train_grouped_for_testing)
                    len(test_grouped_for_testing)

                    stat, p = wilcoxon(train_grouped_for_testing, test_grouped_for_testing)

                    ks_results_dict = {'D':stat , 'p-value':p}

                    ks_results_df = pd.DataFrame.from_dict(ks_results_dict, orient='index').T
                    ks_results_df['num_windows'] = i
                    ks_results_df['fold_number'] = j
                    #concatenate to the results dataframe
                    results_df = pd.concat([results_df, ks_results_df])


                #check if all p-values are above 0.05
                p_values = results_df['p-value'].values
                #check how many times p-values are below 0.05
                if np.sum(p_values < 0.05) == 0:
                    print('All p-values are above 0.05')
                    results_df['above_threshold'] = 1
                else:
                    results_df['above_threshold'] = 0
                big_results_df = pd.concat([big_results_df, results_df])


            # plot the p-values and t-stats against the window size
            # apply a smoothing filter
            # big_results_df['mean_p_value'] = big_results_df['mean_p_value'].rolling(window=10).mean()
            # big_results_df['mean_t_stat'] = big_results_df['mean_t_stat'].rolling(window=10).mean()
            # calculate the index for when the p-value is consistently above 0.2
            # get the index where the p-value is consistently above 0.2

            big_results_df = big_results_df.reset_index()
            # # get the index where the p-value is consistently above 0.2
            index_above_threshold = big_results_df[big_results_df['above_threshold'] > 0]

            # get the applicable thresholds
            threshold_indices = index_above_threshold.index
            diff = np.diff(threshold_indices)
            # find the minimum index where every difference after is 1
            index_saved = None
            for h, val in enumerate(diff):
                if val == 1 and index_saved == None:
                    index_saved = h
                if val != 1 and index_saved == None:
                    index_saved = None
                elif val != 1 and h > index_saved:
                    index_saved = None
            first_index = index_saved


            if first_index == None:
                print('No p-values are above 0.05')
                continue
            corresponding_index = threshold_indices[first_index]
            corresponding_num_windows = big_results_df['num_windows'].values[corresponding_index]

            # fig, ax = plt.subplots(1, 1)
            # ax.plot(big_results_df['num_windows'], big_results_df['mean_p_value'], label='mean p-value')
            # ax.set_title(f'Mean p-value vs num_windows for rat: \n  {data_dir_path} and window size: {window_size}')
            # ax.set_xlabel('num_windows')
            # ax.set_ylabel('mean p-value')
            # # append to an animal and
            #
            # plt.savefig(f'{big_df_savedir}/mean_p_value_vs_num_windows_rat_id_{rat_id}_window_size_{window_size}.png',
            #             dpi=300, bbox_inches='tight')
            # # plt.show()


            # append to an animal and
            plt.close('all')
            # append to the big dataframe
            big_results_df['rat_id'] = rat_id
            big_results_df['window_size'] = window_size
            big_results_df['minimum_number_windows'] = corresponding_num_windows
            df_across_windows = pd.concat([df_across_windows, big_results_df])


    # get the mean minimum number of windows
    df_across_windows['mean_minimum_number_windows'] = df_across_windows.groupby('rat_id')[
        'minimum_number_windows'].transform('mean')
    df_across_windows['mean_minimum_number_windows_by_windowsize'] = df_across_windows.groupby('window_size')[
        'minimum_number_windows'].transform('mean')
    df_across_windows['mean_minimum_number_windows_across_rats_and_windowsize'] = df_across_windows[
        'mean_minimum_number_windows'].mean()

    #get the maximum as well
    df_across_windows['max_minimum_number_windows'] = df_across_windows.groupby('rat_id')[
        'minimum_number_windows'].transform('max')
    df_across_windows['max_minimum_number_windows_by_windowsize'] = df_across_windows.groupby('window_size')[
        'minimum_number_windows'].transform('max')
    df_across_windows['max_minimum_number_windows_across_rats_and_windowsize'] = df_across_windows[
        'max_minimum_number_windows'].max()




    np.unique(df_across_windows['mean_minimum_number_windows_across_rats_and_windowsize'])
    np.unique(df_across_windows['mean_minimum_number_windows_by_windowsize'])

    # export to csv
    df_across_windows.to_csv(f'{big_df_savedir}/mean_p_value_vs_window_size_across_rats_grid_100windows_scale_to_angle_range_{scale_to_angle_range}.csv')
    return df_across_windows


def main():
    data_dir = 'C:/neural_data/rat_7/6-12-2019'
    param_dict, score_dict = load_previous_results(
        'angle_rel_to_goal')
    big_df = pd.DataFrame()
    big_df_savedir = 'C:/neural_data/r2_decoding_figures/umap/'
    # run_stratified_kfold_test()
    # run_ks_test_on_distributions(data_dir, param_dict, score_dict, big_df_savedir)
    run_ks_test_on_distributions_3d_grid(data_dir, param_dict, score_dict, big_df_savedir, scale_to_angle_range=True)
    #'C:/neural_data/rat_3/25-3-2019'




    #save the big dataframe


if __name__ == '__main__':
    #
    main()