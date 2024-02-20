from pathlib import Path
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor
from mpl_toolkits import mplot3d

from sklearn.ensemble import GradientBoostingRegressor
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from helpers import datahandling
from helpers.datahandling import DataHandler
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, permutation_test_score, GridSearchCV, \
    RandomizedSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.dummy import DummyClassifier
from sklearn.metrics import mean_squared_error, r2_score
from umap import UMAP
# import vowel_in_noise.electrophysiology.population_analysis as vowel_pop
# from vowel_in_noise import plot_utils
from sklearn.model_selection import train_test_split
import umap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
mpl.use('Qt5Agg')  # or can use 'TkAgg', whatever you have/prefer

''' Modified from Jules Lebert's code
spks is a numpy arrray of size trial* timebins*neuron, and bhv is  a pandas dataframe where each row represents a trial, the trial is the index '''
from sklearn.decomposition import PCA

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
def unsupervised_umap(spks, bhv, remove_low_variance_neurons = True, neuron_type = 'unknown'):
    # Assuming `spks` is your data
    print(spks[0])
    test_spks = spks[0]
    #apply smoothing to spks
    spks_smoothed = gaussian_filter1d(spks, 4, axis=1)
    epsilon = 1e-10
    # Small constant to prevent division by zero
    # spks_normalized = (spks_smoothed - np.mean(spks_smoothed, axis=1, keepdims=True)) / (np.std(spks_smoothed, axis=1, keepdims=True) + epsilon)
    scaler = StandardScaler()
    # spks_normalized = scaler.fit_transform(spks_smoothed)
    spks_normalized = spks_smoothed
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
    reducer = umap.UMAP(n_components=5, n_neighbors=70, min_dist=0.3, metric='cosine')

    # spks_reshaped = spks.reshape(spks_binned.shape[0], -1)

    embedding = reducer.fit_transform(spks_reshaped)


    # Plot the UMAP decomposition
    plt.scatter(embedding[:, 0], embedding[:, 1])
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP projection of the dataset', fontsize=24)

    # Assuming `bhv` is your behavioral data
    # Create a DataFrame for the UMAP components
    umap_df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2', 'UMAP3', 'UMAP4', 'UMAP5'])

    # Concatenate the UMAP DataFrame with the behavioral data
    bhv_with_umap = pd.concat([bhv, umap_df], axis=1)
    #plot the bhv angle against the umap
    scatter = plt.scatter(bhv_with_umap['UMAP1'], bhv_with_umap['UMAP3'], c=bhv_with_umap['dlc_angle'])
    plt.colorbar(scatter)
    plt.title(f"UMAP projection of the dataset, neuron: {neuron_type}", fontsize=24)
    plt.xticks(fontsize=16)
    plt.xlabel('UMAP1', fontsize=20)
    plt.yticks(fontsize=16)
    plt.ylabel('UMAP3', fontsize=20)

    # plt.colorbar()
    plt.savefig(f'figures/latent_projections/umap_angle_{neuron_type}.png', bbox_inches='tight')
    plt.show()
    #do a 3D plot of the umap
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter( bhv_with_umap['UMAP1'], bhv_with_umap['UMAP2'], bhv_with_umap['UMAP3'],  c=bhv['dlc_angle'])
    plt.colorbar(scatter)
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.set_zlabel('UMAP3')
    plt.title('UMAP projection of the dataset', fontsize=24)
    plt.savefig(f'figures/latent_projections/umap_angle_3d_{neuron_type}.png', bbox_inches='tight', dpi=300)
    plt.show()

    list_of_vars = ['dlc_angle', 'dlc_xy', 'dlc_angle_phase', 'dist_to_goal', 'velocity', 'dlc_body_angle', 'dir_to_goal', 'dlc_angle_phase_body', 'dlc_phase_dir_to_goal']

    for var in list_of_vars:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter( bhv_with_umap['UMAP1'], bhv_with_umap['UMAP2'], bhv_with_umap['UMAP3'],  c=bhv[var])
        plt.colorbar(scatter)
        ax.set_xlabel('UMAP1')
        ax.set_ylabel('UMAP2')
        ax.set_zlabel('UMAP3')
        plt.title(f'UMAP projection of the dataset, color-coded by: {var}', fontsize=15)
        plt.savefig(f'figures/latent_projections/umap_angle_3d_colored_by_{var}_neuron_type_{neuron_type}.png', bbox_inches='tight', dpi=300)
        plt.show()


    return


def process_window(
        w,
        spks,
        window_size,
        y,
        reducer_pipeline,
        regressor,
        regressor_kwargs
):
    reg = regressor(**regressor_kwargs)

    window = spks[:, w:w + window_size, :].reshape(spks.shape[0], -1)

    # Split the data into training and testing sets
    window_train, window_test, y_train, y_test = train_test_split(window, y, test_size=0.2, random_state=42)
    # y_train = np.ravel(y_train)
    # y_test = np.ravel(y_test)
    # Fit the reducer on the training data
    scaler = StandardScaler()
    scaler.fit(window_train)
    window_train = scaler.transform(window_train)
    window_test = scaler.transform(window_test)
    print("Before any transformation:", window_train.shape)
    reducer_pipeline.fit(window_train, y=y_train)
    print("After pipeline transformation:", window_train.shape)

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
        n_jobs=-1,
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
    # results_cv = Parallel(n_jobs=n_jobs, verbose=1)(
    #     delayed(process_window)(w, spks, window_size, y, reducer_pipeline, regressor,
    #                             regressor_kwargs) for w in tqdm(range(spks.shape[1] - window_size)))
    results_cv = Parallel(n_jobs=n_jobs, verbose=1, prefer="threads")(
        delayed(process_window)(w, spks, window_size, y, reducer_pipeline, regressor,
                                regressor_kwargs) for w in tqdm(range(spks.shape[1] - window_size)))
    results_perm = []
    if n_permutations > 0:
        for n in tqdm(range(n_permutations)):
            # y_perm = np.random.permutation(y)
            offset = 2 * np.pi * np.random.random()
            y_perm = np.random.permutation((y + offset) % (2 * np.pi))
            reg = DummyRegressor(strategy='mean')
            results_perm_n = []
            for w in tqdm(range(spks.shape[1] - window_size)):
                window = spks[:, w:w + window_size, :].reshape(spks.shape[0], -1)

                # Fit the regressor on the reference space
                reg.fit(window, y_perm)

                # Predict on the non-reference space
                y_pred = reg.predict(window)

                # Compute the mean squared error and R2 score
                mse_score = mean_squared_error(y_perm, y_pred)
                r2_score_val = r2_score(y_perm, y_pred)

                results = {
                    'mse_score': mse_score,
                    'r2_score': r2_score_val,
                    'w': w,
                }

                results_perm_n.append(results)
            results_perm.append(results_perm_n)

    results = {
        'cv': results_cv,
        'perm': results_perm,
    }

    return results

def main():
    # Load and preprocess data here
    # df_all = load_data_from_paths(Path('C:/neural_data/'))
    data_path = Path('C:/neural_data/rat_7/')
    save_path = Path('C:/neural_data/results')
    if not save_path.exists():
        save_path.mkdir()
    #import the datahandler class
    dh = DataHandler.load_data_from_paths(data_path)
    # spks i sa numpy array of size trial* timebins*neuron
    # bhv is a pandas dataframe where each row represents a trial
    big_spk_array = []
    #load the behavioural data from the C drive:

    positional_data = scipy.io.loadmat(data_path / 'positionalDataByTrialType.mat')
    pos_cell = positional_data['pos']
    hcomb_data_pos = pos_cell[0][0][0][0]
    time = hcomb_data_pos['videoTime']
    ts = hcomb_data_pos['ts']
    dlc_angle = hcomb_data_pos['dlc_angle']
    sample = hcomb_data_pos['sample']
    dist_to_goal = hcomb_data_pos['dist2goal']
    velocity = hcomb_data_pos['velocity']
    dlc_xy = hcomb_data_pos['dlc_XYsmooth']
    dlc_body_angle = hcomb_data_pos['dlc_bodyAngle']
    dir_to_goal = hcomb_data_pos['dirRelative2Goal']

    trial_number_max = np.max(dh['trial_number'])
    time_max = np.max(sample[-1])/30000
    neuron_types = dh['neuron_type'].unique()
    neuron_type = 'interneuron'
    for i in dh['unit_id'].unique():
        dataframe_unit = dh.loc[(dh['unit_id'] == i) & (dh['neuron_type'] == neuron_type)]
        spk_times = dataframe_unit['spike_times_samples']
        spk_times = spk_times.values
        spk_times = np.array(spk_times.tolist())
        spk_times = spk_times.flatten()
        spk_times = spk_times[~np.isnan(spk_times)]
        #rearrange spks to a numpy array of trial*timebins*neuron
        dataframe_unit['trial_number'] = dataframe_unit['trial_number'].astype(int)
        #round trial number to integer
        bin_interval = 10
        #reorganize the data into a numpy array of time stamp arrays
        #get the maximum trial number in seconds

        #I want to bin the data into 0.5s bins
        length = int(time_max/bin_interval)
        bin_width = 0.5

        #create a 3d array of zeros
        hist_rate_big = np.zeros((length, int(bin_interval/bin_width)-1))
        spk_times = spk_times / 30000
        for j in range(0, length):
            #get the corresponding time stamps
            time_start = j*bin_interval
            time_end = (j+1)*bin_interval
            hist, bin_edges = np.histogram(spk_times, bins = np.arange(time_start, time_end, bin_width))
            hist_rate = hist / bin_width
            hist_rate_big[j, :] = hist_rate
        big_spk_array.append(hist_rate_big)


    spks = np.array(big_spk_array)
    # bin_width = 0.5
    # for i in dh['unit_id'].unique():
    #     dataframe_unit = dh.loc[dh['unit_id'] == i]
    #     spk_times = dataframe_unit['spike_times_samples']
    #     spk_times = spk_times.values
    #     spk_times = np.array(spk_times.tolist())
    #     spk_times = spk_times.flatten()
    #     spk_times = spk_times[~np.isnan(spk_times)]
    #     dataframe_unit['trial_number'] = dataframe_unit['trial_number'].astype(int)
    #
    #     # Increase bin size
    #     bin_interval = 20  # Increase as needed
    #
    #     # Use overlapping bins
    #     overlap = 10  # Adjust as needed
    #     bins = range(0, spk_times.shape[0] - bin_interval, bin_interval - overlap)
    #
    #     hist_rate_big = np.zeros((len(bins), int(bin_interval / bin_width) - 1))
    #     spk_times = spk_times / 30000
    #     for j, bin in enumerate(bins):
    #         time_start = bin
    #         time_end = bin + bin_interval
    #         hist, bin_edges = np.histogram(spk_times, bins=np.arange(time_start, time_end, bin_width))
    #         hist_rate = hist / bin_width
    #         hist_rate_big[j, :] = hist_rate
    #     if hist_rate_big.size > 0:
    #         big_spk_array.append(hist_rate_big)

    # spks = np.array(big_spk_array)
    #reshape into trial*timebins*neuron
    spks = np.swapaxes(spks, 0, 1)
    spks = np.swapaxes(spks, 1, 2)
    #remove the last dimension
    #only use the columns of dlc_angle and dlc_xy
    #do the same for the positional data

    dlc_angle_big = []
    dlc_xy_big = []
    sample_big = []
    dist_to_goal_big = []
    velocity_big = []
    dlc_body_angle_big = []
    dir_to_goal_big = []
    for i in range(0, len(dlc_angle)):
        dlc_angle_trial = dlc_angle[i]
        dlc_xy_trial = dlc_xy[i]
        sample_trial = sample[i]
        dlc_angle_big = np.append(dlc_angle_big, dlc_angle_trial)
        dlc_xy_big = np.append(dlc_xy_big, dlc_xy_trial)
        sample_big = np.append(sample_big, sample_trial)
        dist_to_goal_big = np.append(dist_to_goal_big, dist_to_goal[i])
        velocity_big = np.append(velocity_big, velocity[i])
        dlc_body_angle_big = np.append(dlc_body_angle_big, dlc_body_angle[i])
        dir_to_goal_big = np.append(dir_to_goal_big, dir_to_goal[i])

    #interpolate dlc_angle_big to match the length of spks
    #interpolate the dlc_angle_big to match the length of sp
    dlc_angle_big = np.array(dlc_angle_big)
    dlc_xy_big = np.array(dlc_xy_big)
    dist_to_goal_big = np.array(dist_to_goal_big)
    velocity_big = np.array(velocity_big)
    dlc_body_angle_big = np.array(dlc_body_angle_big)
    dir_to_goal_big = np.array(dir_to_goal_big)

    #interpolate the dlc_angle_big to match the length of
    dlc_angle_new = np.interp(np.arange(0, len(dlc_angle_big), len(dlc_angle_big)/len(spks)), np.arange(0, len(dlc_angle_big)), dlc_angle_big)
    dlc_xy_new = np.interp(np.arange(0, len(dlc_xy_big), len(dlc_xy_big)/len(spks)), np.arange(0, len(dlc_xy_big)), dlc_xy_big)
    dist_to_goal_new = np.interp(np.arange(0, len(dist_to_goal_big), len(dist_to_goal_big)/len(spks)), np.arange(0, len(dist_to_goal_big)), dist_to_goal_big)
    velocity_new = np.interp(np.arange(0, len(velocity_big), len(velocity_big)/len(spks)), np.arange(0, len(velocity_big)), velocity_big)
    dlc_body_angle_new = np.interp(np.arange(0, len(dlc_body_angle_big), len(dlc_body_angle_big)/len(spks)), np.arange(0, len(dlc_body_angle_big)), dlc_body_angle_big)
    dir_to_goal_new = np.interp(np.arange(0, len(dir_to_goal_big), len(dir_to_goal_big)/len(spks)), np.arange(0, len(dir_to_goal_big)), dir_to_goal_big)

    #convert dlc_angle_new to radians
    dlc_angle_new = np.radians(dlc_angle_new)
    dir_to_goal_new = np.radians(dir_to_goal_new)
    hilbert_transform = scipy.signal.hilbert(dlc_angle_new)

    hilbert_transform_body = scipy.signal.hilbert(dlc_body_angle_new)
    instantaneous_phase_body = np.angle(hilbert_transform_body)
    hilbert_transform_dir = scipy.signal.hilbert(dir_to_goal_new)
    instantaneous_phase_dir = np.angle(hilbert_transform_dir)

    instantaneous_phase = np.angle(hilbert_transform)

    bhv_umap = pd.DataFrame({'dlc_angle': dlc_angle_new, 'dlc_xy': dlc_xy_new, 'dlc_angle_phase': instantaneous_phase, 'dist_to_goal': dist_to_goal_new, 'velocity': velocity_new, 'dlc_body_angle': dlc_body_angle_new, 'dir_to_goal': dir_to_goal_new, 'dlc_angle_phase_body': instantaneous_phase_body, 'dlc_phase_dir_to_goal': instantaneous_phase_dir})
    bhv = pd.DataFrame({'dlc_angle': instantaneous_phase})
    #run the unsupervised umap
    # unsupervised_pca(spks, bhv_umap)
    unsupervised_umap(spks, bhv_umap, remove_low_variance_neurons=False, neuron_type=neuron_type)

    # time_window = [-0.2, 0.9]
    window_for_decoding = 6  # in s
    window_size = int(window_for_decoding / bin_width)  # in bins
    smooth_spikes = True
    # t = np.arange(time_window[0], time_window[1], bin_width)
    # t = np.round(t, 3)
    n_runs = 5

    regressor = SVR
    regressor_kwargs = {'kernel': 'poly', 'C': 1}

    # regressor = GradientBoostingRegressor
    # regressor_kwargs = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3}

    reducer = UMAP
    reducer_kwargs = {
        'n_components': 2,
        'n_neighbors': 10,
        'min_dist': 0.001,
        'metric': 'cosine',
        'n_jobs': 1,
    }

    # space_ref = ['No Noise', 'Noise']
    #temporarily remove the space_ref variable, I don't want to incorporate separate data yet
    regress = 'dlc_angle'  # Assuming 'head_angle' is the column in your DataFrame for regression

    # Use KFold for regression
    kf = KFold(n_splits=5, shuffle=True)

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f'results_{now}.npy'
    results_between = {}
    results_within = {}
    n_permutations = 100
    for run in range(n_runs):
        results_between[run] = {}
        results_within[run] = {}
        # for space in space_ref:
        results_between[run] = train_ref_classify_rest(
            spks,
            bhv,
            regress,
            regressor,
            regressor_kwargs,
            reducer,
            reducer_kwargs,
            window_size,
            n_permutations=n_permutations,
        )

        # results_within[run] = train_within(
        #     spks,
        #     bhv,
        #     kf,
        #     regress,
        #     regressor,
        #     regressor_kwargs,
        #     reducer,
        #     reducer_kwargs,
        #     window_size,
        #     n_permutations=n_permutations,
        # )

        # Save results
    results = {'between': results_between, 'within': results_within}
    save_path.mkdir(exist_ok=True)
    np.save(save_path / filename, results)


if __name__ == '__main__':
    main()