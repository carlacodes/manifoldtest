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

def colormap_2d():
    # get the viridis colormap
    v_cmap = plt.get_cmap('viridis')
    v_colormap_values = v_cmap(np.linspace(0, 1, 256))

    # get the cool colormap
    c_cmap = plt.get_cmap('cool')
    c_colormap_values = c_cmap(np.linspace(0, 1, 256))

    # get the indices of each colormap for the 2d map
    v_v, c_v = np.meshgrid(np.arange(256), np.arange(256))

    # create a new 2d array with the values of the colormap
    colormap = np.zeros((256, 256, 4))

    for x in range(256):
        for y in range(256):
            v_val = v_colormap_values[v_v[x, y], :]
            c_val = c_colormap_values[c_v[x, y], :]

            # take the average of the two colormaps
            colormap[x, y, :] = (v_val + c_val) / 2

    return colormap

def unsupervised_pca(spks, bhv):
    # Assuming `spks` is your data
    print(spks[0])
    test_spks = spks[0]
    #apply smoothing to spks
    spks_smoothed = gaussian_filter1d(spks, 4, axis=1)
    epsilon = 1e-10
    # Small constant to prevent division by zero
    spks_normalized = (spks_smoothed - np.mean(spks_smoothed, axis=1, keepdims=True)) / (np.std(spks_smoothed, axis=1, keepdims=True) + epsilon)
    #get the high variance neurons
    variance = np.var(spks, axis=1)
    #only keep the neurons with high variance
    high_variance_neuron_grid = variance > np.percentile(variance, 25)
    #check which columns have no variance, more than 0.0
    cols_to_remove = []
    #get the dimensions of the high variance neuron grid
    for i in range(0, high_variance_neuron_grid.shape[1]):
        selected_col = high_variance_neuron_grid[:, i]
        #convert true to 1 and false to 0
        selected_col = selected_col.astype(int)
        print(np.sum(selected_col))
        if np.sum(selected_col) < high_variance_neuron_grid.shape[1]/2:
            print("No variance in column", i)
            cols_to_remove.append(i)

    #only keep the high variance neurons
    #remove the neurons with no variance
    spks_normalized = np.delete(spks_normalized, cols_to_remove, axis=2)

    spks_reshaped = spks_smoothed.reshape(spks_normalized.shape[0], -1)
    #apply
    test_spks_reshaped = spks_reshaped[0]
    print(spks_reshaped[0])

    # Use PCA as the reducer
    reducer = PCA(n_components=3)

    embedding = reducer.fit_transform(spks_reshaped)

    # Plot the PCA decomposition
    plt.scatter(embedding[:, 0], embedding[:, 1])
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('PCA projection of the dataset', fontsize=24)

    # Assuming `bhv` is your behavioral data
    # Create a DataFrame for the PCA components
    pca_df = pd.DataFrame(embedding, columns=['PCA1', 'PCA2', 'PCA3'])

    # Concatenate the PCA DataFrame with the behavioral data
    bhv_with_pca = pd.concat([bhv, pca_df], axis=1)
    #plot the bhv angle against the pca
    plt.scatter(bhv_with_pca['PCA1'], bhv_with_pca['PCA3'], c=bhv_with_pca['dlc_xy'])
    plt.title('PCA projection of the dataset', fontsize=24)
    plt.xticks(fontsize=16)
    plt.xlabel('PCA1', fontsize=20)
    plt.yticks(fontsize=16)
    plt.ylabel('PCA3', fontsize=20)
    plt.savefig('figures/latent_projections/pca_angle.png', bbox_inches='tight')
    plt.show()
    #do a 3D plot of the pca
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter( bhv_with_pca['PCA1'], bhv_with_pca['PCA2'], bhv_with_pca['PCA3'],  c=bhv['dlc_angle'])
    plt.colorbar(scatter)
    ax.set_xlabel('PCA1')
    ax.set_ylabel('PCA2')
    ax.set_zlabel('PCA3')
    plt.title('PCA projection of the dataset', fontsize=24)
    plt.savefig('figures/latent_projections/pca_angle_3d.png', bbox_inches='tight')
    plt.show()

    return
def unsupervised_umap(spks, bhv, remove_low_variance_neurons = True, neuron_type = 'unknown', filter_neurons = False, n_components = 3):
    # Assuming `spks` is your data
    print(spks[0])
    test_spks = spks[0]
    #apply smoothing to spks
    spks_mean = np.nanmean(spks, axis=0)
    spks_std = np.nanstd(spks, axis=0)
    spks_std[spks_std == 0] = np.finfo(float).eps
    spks = (spks - spks_mean) / spks_std


    # spks_smoothed = gaussian_filter1d(spks, 4, axis=1)
    epsilon = 1e-10
    # Small constant to prevent division by zero
    # spks_normalized = (spks_smoothed - np.mean(spks_smoothed, axis=1, keepdims=True)) / (np.std(spks_smoothed, axis=1, keepdims=True) + epsilon)
    scaler = StandardScaler()
    # spks_normalized = scaler.fit_transform(spks_smoothed)
    spks_normalized = spks
    #get the high variance neurons
    if remove_low_variance_neurons:
        variance = np.var(spks, axis=1)
        #only keep the neurons with high variance
        high_variance_neuron_grid = variance > np.percentile(variance, 25)
        #check which columns have no variance, more than 0.0
        cols_to_remove = []
        #get the dimensions of the high variance neuron grid
        for i in range(0, high_variance_neuron_grid.shape[1]):
            selected_col = high_variance_neuron_grid[:, i]
            #convert true to 1 and false to 0
            selected_col = selected_col.astype(int)
            print(np.sum(selected_col))
            if np.sum(selected_col) < high_variance_neuron_grid.shape[1]/2:
                print("No variance in column", i)
                cols_to_remove.append(i)

        #only keep the high variance neurons
        #remove the neurons with no variance
        spks_normalized = np.delete(spks_normalized, cols_to_remove, axis=2)



    # spks_reshaped = spks_smoothed.reshape(spks_normalized.shape[0], -1)
    spks_reshaped = scaler.fit_transform(spks_normalized.reshape(spks_normalized.shape[0], -1))
    #apply
    test_spks_reshaped = spks_reshaped[0]
    print(spks_reshaped[0])



    # # Remove low-variance neurons
    # variances = np.var(spks_normalized, axis=2)
    # high_variance_neurons = variances > np.percentile(variances, 25)
    # # Adjust percentile as needed
    # spks_high_variance = spks_normalized[high_variance_neurons]
    # # Now bin the data
    # spks_binned = np.array([np.mean(spks_high_variance[:, bin:bin + bin_size], axis=1) for bin in bins]).T


    reducer = umap.UMAP(n_components=n_components, n_neighbors=70, min_dist=0.3, metric='euclidean')

    # spks_reshaped = spks.reshape(spks_binned.shape[0], -1)

    embedding = reducer.fit_transform(spks_reshaped)


    # Plot the UMAP decomposition

    if n_components == 2:
        plt.scatter(embedding[:, 0], embedding[:, 1])
        plt.gca().set_aspect('equal', 'datalim')
        plt.title('UMAP projection of the dataset', fontsize=24)

        # Assuming `bhv` is your behavioral data
        # Create a DataFrame for the UMAP components
        umap_df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])

        # Concatenate the UMAP DataFrame with the behavioral data
        bhv_with_umap = pd.concat([bhv, umap_df], axis=1)

        list_of_vars = ['x', 'y', 'angle', 'time_index']

        for var in list_of_vars:
            fig, ax = plt.subplots( figsize = (20, 20))
            scatter = ax.scatter(bhv_with_umap['UMAP1'], bhv_with_umap['UMAP2'], c=bhv[var])
            plt.colorbar(scatter)
            ax.set_xlabel('UMAP1')
            ax.set_ylabel('UMAP2')
            plt.title(f'UMAP projection of the dataset, \n color-coded by: {var}', fontsize=15)
            if filter_neurons:
                plt.savefig(f'figures/latent_projections/umap_angle_3d_colored_by_{var}_neuron_type_{neuron_type}.png',
                            bbox_inches='tight', dpi=300)
            else:
                plt.savefig(f'figures/latent_projections/umap_angle_3d_colored_by_{var}_num_components_{n_components}_all_neurons.png',
                            bbox_inches='tight', dpi=300)
            plt.show()

    elif n_components == 3:
        colormap = colormap_2d()


        plt.scatter(embedding[:, 0], embedding[:, 1])
        plt.gca().set_aspect('equal', 'datalim')
        plt.title('UMAP projection of the dataset', fontsize=24)
        # Assuming `bhv` is your behavioral data
        # Create a DataFrame for the UMAP components
        umap_df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2', 'UMAP3'])

        # Concatenate the UMAP DataFrame with the behavioral data
        bhv_with_umap = pd.concat([bhv, umap_df], axis=1)

        list_of_vars = [('x', 'y'), ('angle_sin', 'angle_cos')]

        for var in list_of_vars:
            data_1 = bhv[var[0]]
            data_2 = bhv[var[1]]
            data_1_c = np.interp(data_1, (data_1.min(), data_1.max()), (0, 255)).astype(int)
            data_2_c = np.interp(data_2, (data_2.min(), data_2.max()), (0, 255)).astype(int)

            color_data = colormap[data_1_c, data_2_c]

            fig = plt.figure(figsize=(20, 20))
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter( bhv_with_umap['UMAP1'], bhv_with_umap['UMAP2'], bhv_with_umap['UMAP3'],  c = color_data)
            #add a separate figure that shows the colour map


            # plt.colorbar(scatter)
            ax.set_xlabel('UMAP1')
            ax.set_ylabel('UMAP2')
            ax.set_zlabel('UMAP3')
            plt.title(f'UMAP projection of the dataset, \n color-coded by: {var}', fontsize=30)
            if filter_neurons:
                plt.savefig(f'figures/latent_projections/umap_angle_3d_colored_by_{var}_neuron_type_{neuron_type}_190324.png', bbox_inches='tight', dpi=500)
            else:
                plt.savefig(f'figures/latent_projections/umap_angle_3d_colored_by_{var}_all_neurons_num_components_{n_components}.png', bbox_inches='tight', dpi=300)
            plt.show()
        #save the colour map in a separate figure
            fig, ax = plt.subplots()
            ax.imshow(colormap, extent=(0, 1, 0, 1))
            ax.set_xlabel(var[0])
            ax.set_ylabel(var[1])
            plt.title(f'Colour map for {var[0]} and {var[1]}')
            plt.savefig(f'figures/latent_projections/colour_map_{var[0]}_and_{var[1]}.png', bbox_inches='tight', dpi=300)


        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_subplot(111, projection='3d')

        # Sort the data by time_index to ensure the trajectory is plotted correctly
        bhv_with_umap_sorted = bhv_with_umap.sort_values(by='time_index')

        # Create a colormap
        cmap = plt.get_cmap('viridis')

        # for var in list_of_vars:
        #     # Normalize your data to 0-1 for matching with the colormap
        #     norm = plt.Normalize(bhv_with_umap_sorted[var].min(), bhv_with_umap_sorted[var].max())
        #
        #     for i in range(1, len(bhv_with_umap_sorted)):
        #         ax.plot(bhv_with_umap_sorted['UMAP1'][i - 1:i + 1], bhv_with_umap_sorted['UMAP2'][i - 1:i + 1],
        #                 bhv_with_umap_sorted['UMAP3'][i - 1:i + 1], color=cmap(norm(bhv_with_umap_sorted[var].iloc[i])))
        #
        #     ax.set_xlabel('UMAP1')
        #     ax.set_ylabel('UMAP2')
        #     ax.set_zlabel('UMAP3')
        #     plt.title(f'UMAP projection of the dataset, color-coded by: {var}', fontsize=15)
        #     if filter_neurons:
        #         plt.savefig(f'figures/latent_projections/umap_angle_3d_colored_line_plot_by_{var}_neuron_type_{neuron_type}.png',
        #                     bbox_inches='tight', dpi=300)
        #     else:
        #         plt.savefig(
        #             f'figures/latent_projections/umap_angle_3d_colored_line_plot_by_{var}_all_neurons_num_components_{n_components}.png',
        #             bbox_inches='tight', dpi=300)
        #     plt.show()


    return


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

def decompose_lfp_data(bhv_umap, bin_interval, bin_width):
    path_to_load = Path('C:/neural_data/rat_7')
    phase_array, trial_array, theta_array, df_theta_and_angle = load_theta_data(path_to_load, spike_data=[])
    theta_phase = df_theta_and_angle['theta_phase']
    theta_phase = theta_phase.values

    time_ms = df_theta_and_angle['time_ms']
    time_ms = time_ms.values
    time_seconds = time_ms / 1000
    #reshape into a 2d array of trial*theta_phase

    #reshape theta phase to a 2d array of trial*theta_phase
    theta_phase_reshaped = theta_phase.reshape(-1, 1)
    theta_phase = theta_phase[0:df_theta_and_angle.shape[0]:int(bin_interval/bin_width)]
    bin_interval = 5
    # reorganize the data into a numpy array of time stamp arrays
    # get the maximum trial number in seconds

    # I want to bin the data into 0.5s bins
    time_min = time_ms[0]/1000
    time_max = time_ms[-1]/1000

    length = int((time_max - time_min) / bin_interval)
    bin_width = 0.1

    # create a 3d array of zeros
    lfp_big = np.zeros((length, int(bin_interval / bin_width) - 1))
    for i in range(0, length):
        # get the corresponding time stamps
        time_start = (i * bin_interval) + time_min
        time_end = ((i + 1) * bin_interval) + time_min
        #find the stop and start indices
        start_index = np.where(time_seconds > time_start)[0][0]
        end_index = np.where(time_seconds < time_end)[0][-1]
        theta_phase = theta_phase_reshaped[start_index:end_index]
        # lfp_big[i, :] = theta_phase

    #reshape df_theta_and_angle to match the length of the spks
    #interpolate the dlc_angle_big to match the length of
    return


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
        #not sure if scaler is actually needed here TODO: check if scaler is needed
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
    # results_old = np.load(f'{data_dir}/results_2024-02-22_16-57-58.npy', allow_pickle=True).item(0)

    # data_dir = '/media/jake/DataStorage_6TB/DATA/neural_network/og_honeycomb/rat7/6-12-2019'

    # load spike data
    # spike_dir = os.path.join(data_dir, 'spike_sorting')
    # units = load_pickle('units_w_behav_correlates', spike_dir)

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
    unsupervised_umap(X_for_umap, label_df, remove_low_variance_neurons=False, n_components=3)

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