from pathlib import Path
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
import json
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
''' Modified from Jules Lebert's code
spks is a numpy arrray of size trial* timebins*neuron, and bhv is  a pandas dataframe where each row represents a trial, the trial is the index '''

def unsupervised_umap(spks, bhv):
    # Assuming `spks` is your data
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(spks)

    # Plot the UMAP decomposition
    plt.scatter(embedding[:, 0], embedding[:, 1])
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP projection of the dataset', fontsize=24)

    # Assuming `bhv` is your behavioral data
    # Create a DataFrame for the UMAP components
    umap_df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])

    # Concatenate the UMAP DataFrame with the behavioral data
    bhv = pd.concat([bhv, umap_df], axis=1)
    #plot the bhv angle against the umap
    plt.scatter(bhv['UMAP1'], bhv['UMAP2'], c=bhv['dlc_angle'])
    plt.title('UMAP projection of the dataset', fontsize=24)
    plt.colorbar()
    plt.savefig('/figures/latent_projections/umap_angle.png', bbox_inches='tight')
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

    dlc_xy = hcomb_data_pos['dlc_XYsmooth']
    trial_number_max = np.max(dh['trial_number'])
    time_max = np.max(sample[-1])/30000
    for i in dh['unit_id'].unique():
        dataframe_unit = dh.loc[dh['unit_id'] == i]
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
    #reshape into trial*timebins*neuron
    spks = np.swapaxes(spks, 0, 1)
    spks = np.swapaxes(spks, 1, 2)
    #remove the last dimension
    #only use the columns of dlc_angle and dlc_xy
    #do the same for the positional data

    dlc_angle_big = []
    dlc_xy_big = []
    sample_big = []
    for i in range(0, len(dlc_angle)):
        dlc_angle_trial = dlc_angle[i]
        dlc_xy_trial = dlc_xy[i]
        sample_trial = sample[i]
        dlc_angle_big = np.append(dlc_angle_big, dlc_angle_trial)
        dlc_xy_big = np.append(dlc_xy_big, dlc_xy_trial)
        sample_big = np.append(sample_big, sample_trial)

    #interpolate dlc_angle_big to match the length of spks
    #interpolate the dlc_angle_big to match the length of sp
    dlc_angle_big = np.array(dlc_angle_big)
    #interpolate the dlc_angle_big to match the length of
    dlc_angle_new = np.interp(np.arange(0, len(dlc_angle_big), len(dlc_angle_big)/len(spks)), np.arange(0, len(dlc_angle_big)), dlc_angle_big)
    #convert dlc_angle_new to radians
    dlc_angle_new = np.radians(dlc_angle_new)
    bhv = pd.DataFrame({'dlc_angle': dlc_angle_new})
    #run the unsupervised umap
    unsupervised_umap(spks, bhv)



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