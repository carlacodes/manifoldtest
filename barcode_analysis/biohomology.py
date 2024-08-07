import copy
import pandas as pd
import pickle
import os
import ripserplusplus as rpp
from gph import ripser_parallel
from gtda.homology._utils import _postprocess_diagrams
from itertools import groupby
from operator import itemgetter
import seaborn as sns
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from helpers import utils

def plot_homology_changes_heatmap_interval(dgm_dict, save_dir, start_indices, end_indices):
    """
    Plot how the homology changes over the range of `j` using a heatmap.

    Parameters
    ----------
    dgm_dict: dict
        Dictionary containing persistence diagrams for each `j`.
    save_dir: str
        Directory to save the plot.
    start_indices: list
        List of start indices for each segment.
    end_indices: list
        List of end indices for each segment.
    """
    heatmap_data = []

    for j, dgm in dgm_dict.items():
        dgm_gtda = _postprocess_diagrams([dgm["dgms"]], "ripser", (0, 1, 2), np.inf, True)
        birth_times = dgm_gtda[0][:, 0]
        death_times = dgm_gtda[0][:, 1]
        dimensions = dgm_gtda[0][:, 2]

        for birth, death, dim in zip(birth_times, death_times, dimensions):
            heatmap_data.append([j, birth, death, dim])

    df = pd.DataFrame(heatmap_data, columns=['Interval', 'Birth', 'Death', 'Dimension'])
    df['death_minus_birth'] = df['Death'] - df['Birth']

    for idx, (start, end) in enumerate(zip(start_indices, end_indices)):
        segment_df = df[(df['Interval'] >= start) & (df['Interval'] <= end)]

        plt.figure(figsize=(12, 8))
        heatmap_data = segment_df.pivot_table(index='Interval', columns='Dimension', values='death_minus_birth', aggfunc='mean')

        ax = sns.heatmap(heatmap_data, cmap='viridis')
        # ax.axhline(start, color='r', linestyle='--')
        # ax.axhline(end+1, color='r', linestyle='--')
        y_ticks = heatmap_data.index.values

        mapped_start_indices = [np.where(y_ticks == start)[0][0] for start in start_indices if start in y_ticks]
        mapped_end_indices = [np.where(y_ticks == end)[0][0] for end in end_indices if end in y_ticks]

        # Map start indices to y-tick positions

        # Annotate y-ticks with start indices
        for idx, (start_annotate, end_annotate) in enumerate(zip(mapped_start_indices, mapped_end_indices)):
            ax.annotate('Start', xy=(0, start_annotate), xycoords='data', xytext=(-50, 0), textcoords='offset points', color='r',
                        fontsize=12, ha='right', va='center', arrowprops=dict(arrowstyle='->', color='r'))
            ax.annotate('End', xy=(0, end_annotate), xycoords='data', xytext=(-50, 0), textcoords='offset points', color='r', fontsize=12,
                        ha='right', va='center', arrowprops=dict(arrowstyle='->', color='r'))




        cbar = ax.collections[0].colorbar
        cbar.set_label('Death - Birth')
        plt.title(f'Homology Changes Heatmap Over Intervals {start}-{end} for animal: {save_dir.split("/")[-4]}')
        plt.xlabel('Homology Dimension')
        plt.ylabel('Interval (j)')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/homology_changes_heatmap_over_intervals_{start}_{end}.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    return df

def plot_homology_changes_heatmap(dgm_dict, save_dir, start_indices = None, end_indices = None, cumulative_param = False):
    """
    Plot how the homology changes over the range of `j` using a heatmap.

    Parameters
    ----------
    dgm_dict: dict
        Dictionary containing persistence diagrams for each `j`.
    save_dir: str
        Directory to save the plot.
    """
    heatmap_data = []

    for j, dgm in dgm_dict.items():
        dgm_gtda = _postprocess_diagrams([dgm["dgms"]], "ripser", (0, 1, 2), np.inf, True)
        birth_times = dgm_gtda[0][:, 0]
        death_times = dgm_gtda[0][:, 1]
        dimensions = dgm_gtda[0][:, 2]

        for birth, death, dim in zip(birth_times, death_times, dimensions):
            heatmap_data.append([j, birth, death, dim])


    df = pd.DataFrame(heatmap_data, columns=['Interval', 'Birth', 'Death', 'Dimension'])
    df['death_minus_birth'] = df['Death'] - df['Birth']

    plt.figure(figsize=(12, 8))
    heatmap_data = df.pivot_table(index='Interval', columns='Dimension', values='death_minus_birth', aggfunc='mean')

    ax = sns.heatmap(heatmap_data, cmap='viridis')
    #mark the start and end trial indices
    # for start, end in zip(start_indices, end_indices):
    #     ax.axhline(start, color='r', linestyle='--')
    #     ax.axhline(end, color='r', linestyle='--')
    y_ticks = heatmap_data.index.values

    # for idx, (start, end) in enumerate(zip(start_indices, end_indices)):
    if start_indices is not None:
        mapped_start_indices = [np.where(y_ticks == start)[0][0] for start in start_indices if start in y_ticks]
        mapped_end_indices = [np.where(y_ticks == end)[0][0] for end in end_indices if end in y_ticks]

        # Map start indices to y-tick positions

        # Annotate y-ticks with start indices
        for idx, (start_annotate, end_annotate) in enumerate(zip(mapped_start_indices, mapped_end_indices)):
            ax.annotate('Start', xy=(0, start_annotate), xycoords='data', xytext=(-50, 0), textcoords='offset points',
                        color='r',
                        fontsize=12, ha='right', va='center', arrowprops=dict(arrowstyle='->', color='r'))
            # ax.annotate('End', xy=(0, end_annotate), xycoords='data', xytext=(-50, 0), textcoords='offset points',
            #             color='r', fontsize=12,
            #             ha='right', va='center', arrowprops=dict(arrowstyle='->', color='r'))

    cbar = ax.collections[0].colorbar
    cbar.set_label('Death - Birth')
    plt.title(f'Homology Changes Heatmap Over Intervals for animal: {save_dir.split("/")[-4]}')
    #add a colorbar label
    plt.xlabel('Homology Dimension')
    plt.ylabel('Interval (j)')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/homology_changes_heatmap_over_intervals_cumulative_{cumulative_param }.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    return df


def sinusoidal(x, A, B, C, D):
    return A * np.sin(B * x + C) + D

def fit_sinusoid_data(df, save_dir):

    # Define a sinusoidal function for fitting

    # Extract data from the heatmap
    heatmap_data = df.pivot_table(index='Interval', columns='Dimension', values='death_minus_birth', aggfunc='mean')

    # Prepare to store fit parameters
    fit_params = {}

    # Fit the function for each homology dimension
    for dim in heatmap_data.columns:
        x_data = heatmap_data.index.values
        y_data = heatmap_data[dim].values

        # Initial guess for the parameters
        initial_guess = [1, 1, 0, np.mean(y_data)]

        # Fit the sinusoidal function to the data
        params, _ = curve_fit(sinusoidal, x_data, y_data, p0=initial_guess)
        fit_params[dim] = params
        r_squared = calculate_goodness_of_fit(x_data, y_data, params)

        print(f'R-squared for dimension {dim}: {r_squared}')

        # Plot the original data and the fitted function
        plt.figure(figsize=(10, 6))
        plt.plot(x_data, y_data, 'bo', label='Original Data')
        # plt.plot(x_data, sinusoidal(x_data, *params), 'r-', label='Fitted Function')
        plt.title(f'Sinusoidal Fit for Homology Dimension {dim}')
        plt.xlabel('Interval (j)')
        plt.ylabel('Mean Death - Birth')
        plt.legend()
        plt.show()

    # Print the fit parameters for each dimension
    for dim, params in fit_params.items():
        print(f'Dimension {dim}: A={params[0]}, B={params[1]}, C={params[2]}, D={params[3]}')

    return fit_params



def fit_smooth_function(df, save_dir):
    """
    Fit a smooth function to the data and plot the results.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing the data to fit.
    save_dir: str
        Directory to save the plot.
    """
    # Extract data from the heatmap
    heatmap_data = df.pivot_table(index='Interval', columns='Dimension', values='death_minus_birth', aggfunc='mean')

    # Prepare to store fit parameters
    fit_params = {}

    # Fit the function for each homology dimension
    for dim in heatmap_data.columns:
        x_data = heatmap_data.index.values
        y_data = heatmap_data[dim].values

        # Fit a smooth spline to the data
        spline = UnivariateSpline(x_data, y_data, s=1)
        fit_params[dim] = spline
        r_squared = calculate_goodness_of_fit(x_data, y_data, spline(x_data))
        print(f'R-squared for dimension {dim}: {r_squared}')

        # Plot the original data and the fitted function
        plt.figure(figsize=(10, 6))
        plt.plot(x_data, y_data, 'bo', label='Original Data')
        plt.plot(x_data, spline(x_data), 'r-', label='Fitted Function')
        plt.title(f'Smooth Fit for Homology Dimension {dim}')
        plt.xlabel('Interval (j)')
        plt.ylabel('Mean Death - Birth')
        plt.legend()
        plt.show()

    # Print the fit parameters for each dimension
    for dim, spline in fit_params.items():
        print(f'Dimension {dim}: Spline knots={spline.get_knots()}')

    return fit_params

def calculate_goodness_of_fit(x_data, y_data, y_fitted):
    """
    Calculate the goodness of fit (R-squared) for the fitted function.

    Parameters
    ----------
    x_data: np.array
        Array of x data points.
    y_data: np.array
        Array of y data points.
    y_fitted: np.array
        Array of fitted y data points.

    Returns
    -------
    float
        R-squared value.
    """
    # Calculate residuals
    residuals = y_data - y_fitted

    # Sum of squared residuals (SSR)
    ssr = np.sum(residuals ** 2)

    # Total sum of squares (SST)
    sst = np.sum((y_data - np.mean(y_data)) ** 2)

    # R-squared
    r_squared = 1 - (ssr / sst)

    return r_squared
def plot_homology_changes_over_j(dgm_dict, save_dir):
    """
    Plot how the homology changes over the range of `j`.

    Parameters
    ----------
    dgm_dict: dict
        Dictionary containing persistence diagrams for each `j`.
    save_dir: str
        Directory to save the plot.
    """
    plt.figure(figsize=(12, 8))

    for j, dgm in dgm_dict.items():
        dgm_gtda = _postprocess_diagrams([dgm["dgms"]], "ripser", (0, 1, 2), np.inf, True)
        birth_times = dgm_gtda[0][:, 0]
        death_times = dgm_gtda[0][:, 1]
        dimensions = dgm_gtda[0][:, 2]

        plt.plot([j] * len(birth_times), birth_times, 'go', label='Birth' if j == 0 else "")
        plt.plot([j] * len(death_times), death_times, 'ro', label='Death' if j == 0 else "")

    plt.xlabel('Interval (j)')
    plt.ylabel('Filtration Value')
    plt.title('Homology Changes Over Intervals')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_dir}/homology_changes_over_intervals.png', dpi=300)
    plt.show()
    plt.close()

def reformat_persistence_diagrams(dgms):
    '''Reformat the persistence diagrams to be in the format required by the giotto package
    Parameters
    ----------
    dgms: list of np.arrays: list of persistence diagrams, each of shape (num_features, 3), i.e. each feature is
           a triplet of (birth, death, dim) as returned by e.g.
           VietorisRipsPersistence
           Returns
           -------
           dgm: np.array: of shape (num_features, 4), i.e. each feature is
           '''

    for i in (0, len(dgms) - 1):
        indiv_dgm = dgms[i]
        # append the dimension
        #add the dimension
        indiv_dgm = np.hstack((indiv_dgm, np.ones((indiv_dgm.shape[0], 1)) * i))
        # append to a larger array
        if i == 0:
            dgm = indiv_dgm
        else:
            dgm = np.vstack((dgm, indiv_dgm))

    ##for each row make an array
    dgm = np.array([np.array(row) for row in dgm])
    #add extra dimension in first dimension
    dgm = np.expand_dims(dgm, axis=0)
    return dgm
def plot_barcode(diag, dim, save_dir=None,fold = 0, **kwargs):
    """ taken from giotto-tda issues
    Plot the barcode for a persistence diagram using matplotlib
    ----------
    diag: np.array: of shape (num_features, 3), i.e. each feature is
           a triplet of (birth, death, dim) as returned by e.g.
           VietorisRipsPersistence
    dim: int: Dimension for which to plot
    **kwargs
    Returns
    -------
    None.

    """
    diag_dim = diag[diag[:, 2] == dim]
    birth = diag_dim[:, 0]; death = diag_dim[:, 1]
    finite_bars = death[death != np.inf]
    if len(finite_bars) > 0:
        inf_end = 2 * max(finite_bars)
    else:
        inf_end = 2
    death[death == np.inf] = inf_end
    plt.figure(figsize=kwargs.get('figsize', (10, 5)))
    for i, (b, d) in enumerate(zip(birth, death)):
        if d == inf_end:
            plt.plot([b, d], [i, i], color='k', lw=kwargs.get('linewidth', 2))
        else:
            plt.plot([b, d], [i, i], color=kwargs.get('color', 'b'), lw=kwargs.get('linewidth', 2))
    plt.title(kwargs.get('title', 'Persistence Barcode, dim ' + str(dim) +'and fold ' + str(fold)))
    plt.xlabel(kwargs.get('xlabel', 'Filtration Value'))
    plt.yticks([])
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(save_dir + '/barcode_fold_h2' + str(fold) +'dim'+ str(dim)+'.png', dpi=300, bbox_inches='tight')
    plt.show()

    # plt.show()


def process_data(reduced_data, trial_indices, segment_length, cumulative=False):
    sorted_list = []
    current_index = 0
    total_length = len(trial_indices)
    cumulative_segment = []

    while current_index < total_length:
        current_trial = trial_indices[current_index]
        segment = []

        # Collect indices for the current segment within the current trial
        while current_index < total_length and (cumulative or len(segment) < segment_length):
            if trial_indices[current_index] == current_trial:
                segment.append(current_index)
                current_index += 1
            else:
                break

        # Handle cumulative mode
        if cumulative:
            if cumulative_segment and trial_indices[cumulative_segment[-1]] != current_trial:
                cumulative_segment = []
            cumulative_segment.extend(segment)
            # Add segment to the list only if it has the desired length
            if len(cumulative_segment) >= segment_length:
                sorted_list.append(cumulative_segment[:])
        else:
            # Non-cumulative mode: simply add the segment if it reaches the desired length
            if len(segment) == segment_length:
                sorted_list.append(segment)

    return sorted_list

def run_persistence_analysis(folder_str, input_df, use_ripser=False, segment_length=40, cumulative_param = True):
    pairs_list = []
    dgm_dict = {}
    sorted_list = []

    reduced_data = np.load(folder_str + '/full_set_transformed.npy')
    trial_info = input_df
    trial_indices = trial_info['trial']

    #get the sorted list
    sorted_list = process_data(reduced_data, trial_indices, segment_length, cumulative=cumulative_param)
    if cumulative_param:
        for i in range(len(sorted_list)):
            sorted_data_trial = reduced_data[sorted_list[i], :]
            #break each sorted_data_trial into chunks of segment_length
            reduced_data_loop_list = []
            for j in range(0, len(sorted_data_trial), segment_length):
                #needs to be cumulative
                reduced_data_loop = sorted_data_trial[0:j + segment_length, :]
                #append to a list
                reduced_data_loop_list.append(reduced_data_loop)
                dgm = ripser_parallel(reduced_data_loop, maxdim=2, n_threads=20, return_generators=True)
                dgm_gtda = _postprocess_diagrams([dgm["dgms"]], "ripser", (0, 1, 2), np.inf, True)
                dgm_dict[j] = dgm


            np.save(folder_str + '/dgm_fold_h2' + '_interval_' + str(i) + f'_cumulative_{cumulative_param}.npy', dgm)

            df_output = plot_homology_changes_heatmap(dgm_dict, folder_str, cumulative_param=cumulative_param)
            # fit_params = utils.fit_sinusoid_data_whole(df_output, folder_str, cumulative_param=cumulative_param)


    else:
        trial_indices = np.array(trial_indices)
        #calcuate where the value of trial_indices changes
        trial_indices_diff = np.diff(trial_indices)
        #get the indices where the trial_indices change
        trial_indices_change = np.where(trial_indices_diff != 0)[0]
        #get the start and end indices
        start_indices = np.insert(trial_indices_change + 1, 0, 0)
        end_indices = np.append(trial_indices_change, len(trial_indices) - 1)
        #get the corresponding sorted_list interval number

        ##find the start indices for each sorted_list component
        start_intervals = []
        end_intervals = []
        for i in range(len(sorted_list)):
            if sorted_list[i][0] in start_indices:
                start_intervals.append(i)
            if sorted_list[i][-1] in end_indices:
                end_intervals.append(i)

        sorted_list = [x for x in sorted_list if x != []]

        for j in range(len(sorted_list)):
            reduced_data_loop = reduced_data[sorted_list[j], :]
            if use_ripser:
                pairs = rpp.run("--format point-cloud --dim " + str(2), reduced_data_loop)[2]
                pairs_list.append(pairs)
                np.save(folder_str + '/pairs_fold_h2' + str(j) + '.npy', pairs)
            else:
                dgm = ripser_parallel(reduced_data_loop, maxdim=2, n_threads=20, return_generators=True)
                dgm_gtda = _postprocess_diagrams([dgm["dgms"]], "ripser", (0, 1, 2), np.inf, True)
                dgm_dict[j] = dgm
                np.save(folder_str + '/dgm_fold_h2' + '_interval_' + str(j) + f'_cumulative_{cumulative_param}.npy', dgm)
        #generate the trial indices where the trial changes
        df_output = plot_homology_changes_heatmap(dgm_dict, folder_str, start_intervals, end_intervals)
        fit_params = utils.fit_sinusoid_data_whole(df_output, folder_str, cumulative_param = cumulative_param)
        # fit_params = utils.fit_sinusoid_data_per_interval(df_output, folder_str, start_intervals, end_intervals)

        if use_ripser:
            with open(folder_str + '/pairs_list_h2.pkl', 'wb') as f:
                pickle.dump(pairs_list, f)
        else:
            with open(folder_str + '/dgm_dict_h2.pkl', 'wb') as f:
                pickle.dump(dgm_dict, f)

    return pairs_list


def main():
    #load the already reduced data
    base_dir = 'C:/neural_data/'
    for dir in [ f'{base_dir}/rat_7/6-12-2019', f'{base_dir}/rat_10/23-11-2021', f'{base_dir}/rat_8/15-10-2019', f'{base_dir}/rat_9/10-12-2021', f'{base_dir}/rat_3/25-3-2019']:
        window_df = pd.read_csv(
            f'C:/neural_data/mean_p_value_vs_window_size_across_rats_grid_250_windows_scale_to_angle_range_False_allo_True.csv')
        # find the rat_id
        rat_id = dir.split('/')[-2]
        # filter for window_size
        window_df = window_df[window_df['window_size'] == 250]
        num_windows = window_df[window_df['rat_id'] == rat_id]['minimum_number_windows'].values[0]
        #read the input label data
        spike_dir = os.path.join(dir, 'physiology_data')
        dlc_dir = os.path.join(dir, 'positional_data')
        labels = np.load(f'{dlc_dir}/labels_250_raw.npy')
        col_list = np.load(f'{dlc_dir}/col_names_250_raw.npy')
        #make input df
        input_df = pd.DataFrame(labels, columns=col_list)


        print('at dir ', dir)
        sub_folder = dir + '/plot_results/'
        #get list of files in the directory
        files = os.listdir(sub_folder)
        #check if more than two dirs
        if len(files) >= 2:
            #choose the most recently modified directory
            files.sort(key=lambda x: os.path.getmtime(sub_folder + x))
            #get the most recently modified directory
            savedir = sub_folder + files[-1]
        else:
            savedir = sub_folder + files[0]

        pairs_list = run_persistence_analysis(savedir, input_df, cumulative_param=True)
        #save the pairs list
        # np.save(savedir + '/pairs_list.npy', pairs_list)










if __name__ == '__main__':
    main()
