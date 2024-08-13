import copy

import gtda.diagrams
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
from persim import bottleneck

##Todo: remove ripser_parallel functionality

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

def plot_homology_changes_heatmap(dgm_dict, save_dir, start_indices = None, end_indices = None, cumulative_param = False, trial_number = None, use_peak_control = False, old_segment_length = None, new_segment_length = None):
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
    #figure out which interval has the peak in persistence


    #find the max mean peak across intervals
    df_zeroth_homology = df[df['Dimension'] == 0]
    mean_peak_by_interval = df_zeroth_homology.groupby('Interval')['death_minus_birth'].mean()
    peak = mean_peak_by_interval.max()
    peak_interval = mean_peak_by_interval.idxmax()


    peak_info = pd.DataFrame({'peak': [peak], 'peak_interval': [peak_interval],  'segment length': [new_segment_length], 'cumulative': [cumulative_param]})
    equal_peak_sanity = None
    if use_peak_control:
        #compare to old peak_info
        peak_info_old = pd.read_csv(f'{save_dir}/peak_info.csv')
        peak_interval_old = peak_info_old['peak_interval'].values[0] + old_segment_length
        #compare to old peak_info
        if peak_interval != peak_interval_old:
            print('not equal')
            equal_peak_sanity = 0
        else:
            print('equal')
            equal_peak_sanity = 1
        peak_info.to_csv(f'{save_dir}/peak_info_control_{use_peak_control}.csv')
    else:
        peak_info.to_csv(f'{save_dir}/peak_info.csv')

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
    if trial_number is not None:
        if use_peak_control:
            plt.title(f'Homology Changes Heatmap Over Intervals for animal: {save_dir.split("/")[-4]}, trial number: {trial_number}, control: {use_peak_control}')
        else:
            plt.title(f'Homology Changes Heatmap Over Intervals for animal: {save_dir.split("/")[-4]}, trial number: {trial_number}')
        plt.xlabel('Homology Dimension')
        plt.ylabel('Interval (j)')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/homology_changes_heatmap_over_intervals_cumulative_{cumulative_param}_trialnum_{trial_number}_control_{use_peak_control}.png', dpi=300,
                    bbox_inches='tight')

    else:
        if use_peak_control:
            plt.title(f'Homology Changes Heatmap Over Intervals for animal: {save_dir.split("/")[-4]}, control: {use_peak_control}')
        else:
            plt.title(f'Homology Changes Heatmap Over Intervals for animal: {save_dir.split("/")[-4]}')
        plt.xlabel('Homology Dimension')
        plt.ylabel('Interval (j)')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/homology_changes_heatmap_over_intervals_cumulative_{cumulative_param}_control_{use_peak_control}.png', dpi=300,
                    bbox_inches='tight')
    #add a colorbar label

    plt.show()
    plt.close()
    return df, equal_peak_sanity


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
def plot_barcode(diag, dim, save_dir=None,count = 0, **kwargs):
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
    hom_group_text = ''
    for i, (b, d) in enumerate(zip(birth, death)):
        if d == inf_end:
            plt.plot([b, d], [i, i], color='k', lw=kwargs.get('linewidth', 2))
        elif dim == 0:
            hom_group_text = 'H0'
            plt.plot([b, d], [i, i], color=kwargs.get('color', 'b'), lw=kwargs.get('linewidth', 2))
        elif dim == 1:
            hom_group_text = 'H1'
            plt.plot([b, d], [i, i], color=kwargs.get('color', 'r'), lw=kwargs.get('linewidth', 2))
        elif dim == 2:
            hom_group_text = 'H2'
            plt.plot([b, d], [i, i], color=kwargs.get('color', 'g'), lw=kwargs.get('linewidth', 2))


    plt.title(kwargs.get('title', 'Persistence Barcode, ' + str(hom_group_text) +' and trial ' + str(count)))
    plt.xlabel(kwargs.get('xlabel', 'Filtration Value'))
    plt.yticks([])
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(save_dir + '/barcode_fold_trialid_' + str(count) +'_dim_'+ str(dim)+'_.png', dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close('all')

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


def calculate_bottleneck_distance(all_diagrams, folder_str):
    #concatenate the diagrams all together into one mega list
    print('..calculating distance matrix')
    mega_diagram_list = []
    for i in range(len(all_diagrams)):
        diagram = all_diagrams[i]
        mega_diagram_list.extend(diagram)


    # Stack diagrams into a single ndarray
    num_diagrams = len(mega_diagram_list)

    distance_matrix_dict = {}
    for l in [0, 1, 2]:
        distance_matrix = np.zeros((num_diagrams, num_diagrams)) + np.nan
        for m in range(num_diagrams):
            for n in range(m + 1, num_diagrams):
                first_array = mega_diagram_list[m]
                first_array = np.squeeze(first_array)  # Remove the extra dimension
                # filter for the dimension
                first_array = first_array[first_array[:, 2] == l]
                # now take the first two columns for persim formatting
                first_array = first_array[:, 0:2]

                second_array = mega_diagram_list[n]
                second_array = np.squeeze(second_array)  # Remove the extra dimension
                # filter for the dimension
                second_array = second_array[second_array[:, 2] == l]
                # now take the first two columns for persim formatting
                second_array = second_array[:, 0:2]
                distance_matrix[m, n] = bottleneck(first_array, second_array)

        # Save the distance matrix
        #remove the diagonal from the matrix
        distance_matrix = np.triu(distance_matrix)
        distance_matrix_dict[l] = distance_matrix
    with open(folder_str + '/distance_matrix_dict.pkl', 'wb') as f:
        pickle.dump(distance_matrix_dict, f)
    #remove the diagonal from the matrix

    return distance_matrix_dict



def run_persistence_analysis(folder_str, input_df, use_ripser=False, segment_length=40, cumulative_param=True, use_peak_control = False):
    pairs_list = []
    dgm_dict_storage = {}
    distance_matrix_dict = {}  # Dictionary to store pairwise distances
    sorted_list = []

    reduced_data = np.load(folder_str + '/full_set_transformed.npy')
    trial_info = input_df
    trial_indices = trial_info['trial']

    # Get the sorted list
    sorted_list = process_data(reduced_data, trial_indices, segment_length, cumulative=True)

    if cumulative_param:
        all_diagrams = []  # List to store all persistence diagrams
        if use_peak_control == False:
            for i in range(len(sorted_list)):
                dgm_dict = {}
                sorted_data_trial = reduced_data[sorted_list[i], :]
                # Break each sorted_data_trial into chunks of segment_length
                reduced_data_loop_list = []
                for j in range(0, len(sorted_data_trial), segment_length):
                    # Needs to be cumulative
                    reduced_data_loop = sorted_data_trial[0:j + segment_length, :]
                    # Append to a list
                    reduced_data_loop_list.append(reduced_data_loop)
                    dgm = ripser_parallel(reduced_data_loop, maxdim=2, n_threads=20, return_generators=True)
                    dgm_gtda = _postprocess_diagrams([dgm["dgms"]], "ripser", (0, 1, 2), np.inf, True)
                    dgm_dict[j] = dgm

                    dgm_dict_storage[(i, j)] = dgm_gtda
                    all_diagrams.append(dgm_gtda)  # Collect diagrams for distance calculation

                np.save(folder_str + '/dgm_fold_h2' + '_interval_' + str(i) + f'_cumulative_{cumulative_param}.npy',
                        dgm_dict)

                df_output = plot_homology_changes_heatmap(dgm_dict, folder_str, cumulative_param=cumulative_param,
                                                          trial_number=i)
                # fit_params = utils.fit_sinusoid_data_whole(df_output, folder_str, cumulative_param=cumulative_param)
        elif use_peak_control:
            peak_info = pd.read_csv(folder_str + '/peak_info.csv')
            peak_interval = peak_info['peak_interval'].values[0]
            peak_index = peak_info['peak_index'].values[0]
            segment_length_new = peak_interval + segment_length
            equal_peak_sanity_list = []
            for i in range(len(sorted_list)):
                dgm_dict = {}
                sorted_data_trial = reduced_data[sorted_list[i], :]
                # Break each sorted_data_trial into chunks of segment_length
                reduced_data_loop_list = []
                for j in range(0, len(sorted_data_trial), segment_length_new):
                    # Needs to be cumulative
                    reduced_data_loop = sorted_data_trial[j:j + segment_length_new, :]
                    # Append to a list
                    reduced_data_loop_list.append(reduced_data_loop)
                    dgm = ripser_parallel(reduced_data_loop, maxdim=2, n_threads=20, return_generators=True)
                    dgm_gtda = _postprocess_diagrams([dgm["dgms"]], "ripser", (0, 1, 2), np.inf, True)
                    dgm_dict[j] = dgm

                    dgm_dict_storage[(i, j)] = dgm_gtda
                    all_diagrams.append(dgm_gtda)  # Collect diagrams for distance calculation

                np.save(folder_str + '/dgm_fold_h2' + '_interval_' + str(i) + f'_cumulative_{cumulative_param}_control_{use_peak_control}.npy',
                        dgm_dict)

                df_output, equal_peak_sanity = plot_homology_changes_heatmap(dgm_dict, folder_str, cumulative_param=cumulative_param,
                                                          trial_number=i, use_peak_control = use_peak_control, old_segment_length=segment_length, new_segment_length=segment_length_new)
                equal_peak_sanity_list.append(equal_peak_sanity)
                #get the fraction of 1s
                frac_of_ones = len([x for x in equal_peak_sanity_list if x == 1])/len(equal_peak_sanity_list)



        with open(folder_str + '/all_diagrams_h2_cumulative_trialbysegment.pkl', 'wb') as f:
            pickle.dump(all_diagrams, f)

        with open(folder_str + '/dgm_dict_h2_cumulative_trialbysegment.pkl', 'wb') as f:
            pickle.dump(dgm_dict_storage, f)



    else:
        all_diagrams = []
        for i in range(len(sorted_list)):
            sorted_data_trial = reduced_data[sorted_list[i], :]
            #plot the persistence barcode across the whole trial
            for dim in [0, 1, 2]:
                dgm = ripser_parallel(sorted_data_trial, maxdim=2, n_threads=20, return_generators=True)
                dgm_gtda = _postprocess_diagrams([dgm["dgms"]], "ripser", (0, 1, 2), np.inf, True)
                #remove the first axis
                dgm_gtda = dgm_gtda[0]
                plot_barcode(dgm_gtda, dim, save_dir=folder_str, count=i)


    return all_diagrams, dgm_dict_storage


# def run_persistence_analysis(folder_str, input_df, use_ripser=False, segment_length=40, cumulative_param = True):
#     pairs_list = []
#     dgm_dict_storage = {}
#     sorted_list = []
#
#     reduced_data = np.load(folder_str + '/full_set_transformed.npy')
#     trial_info = input_df
#     trial_indices = trial_info['trial']
#
#     #get the sorted list
#     sorted_list = process_data(reduced_data, trial_indices, segment_length, cumulative=cumulative_param)
#     if cumulative_param:
#         for i in range(len(sorted_list)):
#             dgm_dict = {}
#             sorted_data_trial = reduced_data[sorted_list[i], :]
#             #break each sorted_data_trial into chunks of segment_length
#             reduced_data_loop_list = []
#             for j in range(0, len(sorted_data_trial), segment_length):
#                 #needs to be cumulative
#                 reduced_data_loop = sorted_data_trial[0:j + segment_length, :]
#                 #append to a list
#                 reduced_data_loop_list.append(reduced_data_loop)
#                 dgm = ripser_parallel(reduced_data_loop, maxdim=2, n_threads=20, return_generators=True)
#                 dgm_gtda = _postprocess_diagrams([dgm["dgms"]], "ripser", (0, 1, 2), np.inf, True)
#                 dgm_dict[j] = dgm
#
#
#                 dgm_dict_storage[(i, j)] = dgm
#
#             np.save(folder_str + '/dgm_fold_h2' + '_interval_' + str(i) + f'_cumulative_{cumulative_param}.npy', dgm)
#
#
#             df_output = plot_homology_changes_heatmap(dgm_dict, folder_str, cumulative_param=cumulative_param, trial_number = i)
#             # fit_params = utils.fit_sinusoid_data_whole(df_output, folder_str, cumulative_param=cumulative_param)
#
#         with open(folder_str + '/dgm_dict_h2_cumulative_trialbysegment.pkl', 'wb') as f:
#             pickle.dump(dgm_dict_storage, f)
#
#
#     else:
#         trial_indices = np.array(trial_indices)
#         #calcuate where the value of trial_indices changes
#         trial_indices_diff = np.diff(trial_indices)
#         #get the indices where the trial_indices change
#         trial_indices_change = np.where(trial_indices_diff != 0)[0]
#         #get the start and end indices
#         start_indices = np.insert(trial_indices_change + 1, 0, 0)
#         end_indices = np.append(trial_indices_change, len(trial_indices) - 1)
#         #get the corresponding sorted_list interval number
#
#         ##find the start indices for each sorted_list component
#         start_intervals = []
#         end_intervals = []
#         for i in range(len(sorted_list)):
#             if sorted_list[i][0] in start_indices:
#                 start_intervals.append(i)
#             if sorted_list[i][-1] in end_indices:
#                 end_intervals.append(i)
#
#         sorted_list = [x for x in sorted_list if x != []]
#
#         for j in range(len(sorted_list)):
#             reduced_data_loop = reduced_data[sorted_list[j], :]
#             if use_ripser:
#                 pairs = rpp.run("--format point-cloud --dim " + str(2), reduced_data_loop)[2]
#                 pairs_list.append(pairs)
#                 np.save(folder_str + '/pairs_fold_h2' + str(j) + '.npy', pairs)
#             else:
#                 dgm = ripser_parallel(reduced_data_loop, maxdim=2, n_threads=20, return_generators=True)
#                 dgm_gtda = _postprocess_diagrams([dgm["dgms"]], "ripser", (0, 1, 2), np.inf, True)
#                 dgm_dict[j] = dgm
#                 np.save(folder_str + '/dgm_fold_h2' + '_interval_' + str(j) + f'_cumulative_{cumulative_param}.npy', dgm)
#         #generate the trial indices where the trial changes
#         df_output = plot_homology_changes_heatmap(dgm_dict, folder_str, start_intervals, end_intervals)
#         fit_params = utils.fit_sinusoid_data_whole(df_output, folder_str, cumulative_param = cumulative_param)
#         # fit_params = utils.fit_sinusoid_data_per_interval(df_output, folder_str, start_intervals, end_intervals)
#
#         if use_ripser:
#             with open(folder_str + '/pairs_list_h2.pkl', 'wb') as f:
#                 pickle.dump(pairs_list, f)
#         else:
#             with open(folder_str + '/dgm_dict_h2.pkl', 'wb') as f:
#                 pickle.dump(dgm_dict, f)
#
#     return pairs_list


def main():
    #load the already reduced data
    base_dir = 'C:/neural_data/'
    big_list = []
    calculate_distance = False
    #check if all_diagrams.pkl exists in the base directory
    if os.path.exists(f'{base_dir}/all_diagrams.pkl') and calculate_distance:
        with open(f'{base_dir}/all_diagrams.pkl', 'rb') as f:
            big_list = pickle.load(f)

    else:

        for subdir in [ f'{base_dir}/rat_8/15-10-2019', f'{base_dir}/rat_9/10-12-2021', f'{base_dir}/rat_3/25-3-2019', f'{base_dir}/rat_7/6-12-2019', f'{base_dir}/rat_10/23-11-2021',]:
            window_df = pd.read_csv(
                f'{base_dir}/mean_p_value_vs_window_size_across_rats_grid_250_windows_scale_to_angle_range_False_allo_True.csv')
            # find the rat_id
            rat_id = subdir.split('/')[-2]
            # filter for window_size
            window_df = window_df[window_df['window_size'] == 250]
            num_windows = window_df[window_df['rat_id'] == rat_id]['minimum_number_windows'].values[0]
            #read the input label data
            spike_dir = os.path.join(subdir, 'physiology_data')
            dlc_dir = os.path.join(subdir, 'positional_data')
            labels = np.load(f'{dlc_dir}/labels_250_raw.npy')
            col_list = np.load(f'{dlc_dir}/col_names_250_raw.npy')
            #make input df
            input_df = pd.DataFrame(labels, columns=col_list)


            print('at dir ', subdir)
            sub_folder = subdir + '/plot_results/'
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

            pairs_list, _ = run_persistence_analysis(savedir, input_df, cumulative_param=True, use_peak_control=False)
            #append pairs_list to a big_list
            big_list.append(pairs_list)

        #calculate the bottleneck distance

    # distance_matrix_dict = calculate_bottleneck_distance(big_list, base_dir)
        #save the pairs list
        # np.save(savedir + '/pairs_list.npy', pairs_list)










if __name__ == '__main__':
    main()
