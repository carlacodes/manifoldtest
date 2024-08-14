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
        heatmap_data = segment_df.pivot_table(index='Interval', columns='Dimension', values='death_minus_birth',
                                              aggfunc='mean')

        ax = sns.heatmap(heatmap_data, cmap='viridis')
        # ax.axhline(start, color='r', linestyle='--')
        # ax.axhline(end+1, color='r', linestyle='--')
        y_ticks = heatmap_data.index.values

        mapped_start_indices = [np.where(y_ticks == start)[0][0] for start in start_indices if start in y_ticks]
        mapped_end_indices = [np.where(y_ticks == end)[0][0] for end in end_indices if end in y_ticks]

        # Map start indices to y-tick positions

        # Annotate y-ticks with start indices
        for idx, (start_annotate, end_annotate) in enumerate(zip(mapped_start_indices, mapped_end_indices)):
            ax.annotate('Start', xy=(0, start_annotate), xycoords='data', xytext=(-50, 0), textcoords='offset points',
                        color='r',
                        fontsize=12, ha='right', va='center', arrowprops=dict(arrowstyle='->', color='r'))
            ax.annotate('End', xy=(0, end_annotate), xycoords='data', xytext=(-50, 0), textcoords='offset points',
                        color='r', fontsize=12,
                        ha='right', va='center', arrowprops=dict(arrowstyle='->', color='r'))

        cbar = ax.collections[0].colorbar
        cbar.set_label('Death - Birth')
        plt.title(f'Homology Changes Heatmap Over Intervals {start}-{end} for animal: {save_dir.split("/")[-4]}')
        plt.xlabel('Homology Dimension')
        plt.ylabel('Interval (j)')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/homology_changes_heatmap_over_intervals_{start}_{end}.png', dpi=300,
                    bbox_inches='tight')
        plt.show()
        plt.close()
    return df


def plot_homology_changes_heatmap(dgm_dict, save_dir, start_indices=None, end_indices=None, cumulative_param=False,
                                  trial_number=None, use_peak_control=False, old_segment_length=None,
                                  new_segment_length=None, shuffled_control=False):
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

    peak_info = pd.DataFrame({'peak': [peak], 'peak_interval': [peak_interval], 'segment length': [new_segment_length],
                              'cumulative': [cumulative_param]})
    equal_peak_sanity = None
    if use_peak_control:
        #compare to old peak_info
        peak_interval = peak_interval + new_segment_length
        peak_info_old = pd.read_csv(f'{save_dir}/peak_info_trial_{trial_number}.csv')
        peak_interval_old = peak_info_old['peak_interval'].values[0] + old_segment_length
        #compare to old peak_info
        assert new_segment_length == peak_interval_old

        if peak_interval != peak_interval_old:
            print('not equal')
            equal_peak_sanity = 0
        else:
            print('equal')
            equal_peak_sanity = 1
        peak_info.to_csv(f'{save_dir}/peak_info_control_{use_peak_control}_trial_{trial_number}.csv')
    else:
        peak_info.to_csv(f'{save_dir}/peak_info_trial_{trial_number}_shuffled_{shuffled_control}.csv')

    plt.figure(figsize=(12, 8))
    heatmap_data = df.pivot_table(index='Interval', columns='Dimension', values='death_minus_birth', aggfunc='mean')

    ax = sns.heatmap(heatmap_data, cmap='viridis')
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
            plt.title(
                f'Homology Changes Heatmap Over Intervals for animal: {save_dir.split("/")[-4]}, trial number: {trial_number}, control: {use_peak_control}')
        else:
            plt.title(
                f'Homology Changes Heatmap Over Intervals for animal: {save_dir.split("/")[-4]}, trial number: {trial_number}')
        plt.xlabel('Homology Dimension')
        plt.ylabel('Interval (j)')
        plt.tight_layout()
        plt.savefig(
            f'{save_dir}/homology_changes_heatmap_over_intervals_cumulative_{cumulative_param}_trialnum_{trial_number}_control_{use_peak_control}_shuffled_{shuffled_control}.png',
            dpi=300,
            bbox_inches='tight')

    else:
        if use_peak_control:
            plt.title(
                f'Homology Changes Heatmap Over Intervals for animal: {save_dir.split("/")[-4]}, control: {use_peak_control}')
        else:
            plt.title(f'Homology Changes Heatmap Over Intervals for animal: {save_dir.split("/")[-4]}')
        plt.xlabel('Homology Dimension')
        plt.ylabel('Interval (j)')
        plt.tight_layout()
        plt.savefig(
            f'{save_dir}/homology_changes_heatmap_over_intervals_cumulative_{cumulative_param}_control_{use_peak_control}_shuffled_{shuffled_control}.png',
            dpi=300,
            bbox_inches='tight')
    #add a colorbar label

    plt.show()
    plt.close()
    return df, equal_peak_sanity



def plot_barcode(diag, dim, save_dir=None, count=0, **kwargs):
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
    birth = diag_dim[:, 0];
    death = diag_dim[:, 1]
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

    plt.title(kwargs.get('title', 'Persistence Barcode, ' + str(hom_group_text) + ' and trial ' + str(count)))
    plt.xlabel(kwargs.get('xlabel', 'Filtration Value'))
    plt.yticks([])
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(save_dir + '/barcode_fold_trialid_' + str(count) + '_dim_' + str(dim) + '_.png', dpi=300,
                    bbox_inches='tight')
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


def run_persistence_analysis(folder_str, input_df, segment_length=40, stride=20, cumulative_param=True,
                             use_peak_control=False, cumulative_windows=False, shuffled_control=False):
    dgm_dict_storage = {}
    sinusoid_df_across_trials = None
    reduced_data = np.load(folder_str + '/full_set_transformed.npy')
    trial_info = input_df
    trial_indices = trial_info['trial']

    # Get the sorted list
    sorted_list = process_data(reduced_data, trial_indices, segment_length, cumulative=True)

    if cumulative_param:
        all_diagrams = []  # List to store all persistence diagrams
        if not use_peak_control and not shuffled_control:
            sinusoid_df_across_trials = pd.DataFrame()
            for i in range(len(sorted_list)):
                dgm_dict = {}
                sorted_data_trial = reduced_data[sorted_list[i], :]
                # Break each sorted_data_trial into sliding windows
                reduced_data_loop_list = []
                for start in range(0, len(sorted_data_trial) - segment_length + 1, stride):
                    end = start + segment_length
                    # Needs to be cumulative
                    if cumulative_windows:
                        reduced_data_loop = sorted_data_trial[:end, :]
                    else:
                        reduced_data_loop = sorted_data_trial[start:end, :]

                    # Append to a list
                    reduced_data_loop_list.append(reduced_data_loop)
                    dgm = ripser_parallel(reduced_data_loop, maxdim=2, n_threads=20, return_generators=True)
                    dgm_gtda = _postprocess_diagrams([dgm["dgms"]], "ripser", (0, 1, 2), np.inf, True)
                    dgm_dict[start] = dgm

                    dgm_dict_storage[(i, start)] = dgm_gtda
                    all_diagrams.append(dgm_gtda)  # Collect diagrams for distance calculation

                np.save(folder_str + '/dgm_fold_h2' + '_interval_' + str(i) + f'_cumulative_{cumulative_param}.npy',
                        dgm_dict)

                df_output, _ = plot_homology_changes_heatmap(dgm_dict, folder_str, cumulative_param=cumulative_param,
                                                             trial_number=i)
                fit_params, df_means = utils.fit_sinusoid_data_filtered(df_output, folder_str,
                                                                     cumulative_param=cumulative_param, trial_number=i)
                sinusoid_df_across_trials = pd.concat([sinusoid_df_across_trials, df_means])

        elif shuffled_control:
            all_diagrams = []  # List to store all persistence diagrams
            sinusoid_df_across_trials = pd.DataFrame()
            for i in range(len(sorted_list)):
                dgm_dict = {}
                sorted_data_trial = reduced_data[sorted_list[i], :]
                shuffled_sorted_data_trial = np.copy(sorted_data_trial)
                np.random.shuffle(shuffled_sorted_data_trial)
                # Check if the shuffled data is the same as the original
                assert not np.array_equal(sorted_data_trial, shuffled_sorted_data_trial)
                # Break each shuffled_sorted_data_trial into sliding windows
                reduced_data_loop_list = []
                for start in range(0, len(shuffled_sorted_data_trial) - segment_length + 1, stride):
                    end = start + segment_length
                    # Needs to be cumulative
                    if cumulative_windows:
                        reduced_data_loop = shuffled_sorted_data_trial[:end, :]
                    else:
                        reduced_data_loop = shuffled_sorted_data_trial[start:end, :]

                    # Append to a list
                    reduced_data_loop_list.append(reduced_data_loop)
                    dgm = ripser_parallel(reduced_data_loop, maxdim=2, n_threads=20, return_generators=True)
                    dgm_gtda = _postprocess_diagrams([dgm["dgms"]], "ripser", (0, 1, 2), np.inf, True)
                    dgm_dict[start] = dgm

                    dgm_dict_storage[(i, start)] = dgm_gtda
                    all_diagrams.append(dgm_gtda)  # Collect diagrams for distance calculation

                np.save(folder_str + '/dgm_fold_h2' + '_interval_' + str(i) + f'_cumulative_{cumulative_param}.npy',
                        dgm_dict)

                df_output, _ = plot_homology_changes_heatmap(dgm_dict, folder_str,
                                                             cumulative_param=cumulative_param,
                                                             trial_number=i, shuffled_control=shuffled_control)
                # fit_params, df_means = utils.fit_sinusoid_data_whole(df_output, folder_str,
                #                                                      cumulative_param=cumulative_param,
                #                                                      trial_number=i, shuffled_control=shuffled_control)
                # sinusoid_df_across_trials = pd.concat([sinusoid_df_across_trials, df_means])

        elif use_peak_control:
            equal_peak_sanity_list = []
            for i in range(len(sorted_list)):
                dgm_dict = {}
                sorted_data_trial = reduced_data[sorted_list[i], :]
                # Break each sorted_data_trial into sliding windows with peak control
                peak_info = pd.read_csv(folder_str + f'/peak_info_trial_{i}.csv')
                peak_interval = peak_info['peak_interval'].values[0]
                segment_length_new = peak_interval + segment_length
                for start in range(0, len(sorted_data_trial) - segment_length_new + 1, stride):
                    end = start + segment_length_new
                    reduced_data_loop = sorted_data_trial[start:end, :]
                    # Append to a list
                    dgm = ripser_parallel(reduced_data_loop, maxdim=2, n_threads=20, return_generators=True)
                    dgm_gtda = _postprocess_diagrams([dgm["dgms"]], "ripser", (0, 1, 2), np.inf, True)
                    dgm_dict[start] = dgm

                    dgm_dict_storage[(i, start)] = dgm_gtda
                    all_diagrams.append(dgm_gtda)  # Collect diagrams for distance calculation

                np.save(folder_str + '/dgm_fold_h2' + '_interval_' + str(
                    i) + f'_cumulative_{cumulative_param}_control_{use_peak_control}.npy',
                        dgm_dict)

                df_output, equal_peak_sanity = plot_homology_changes_heatmap(dgm_dict, folder_str,
                                                                             cumulative_param=cumulative_param,
                                                                             trial_number=i,
                                                                             use_peak_control=use_peak_control,
                                                                             old_segment_length=segment_length,
                                                                             new_segment_length=segment_length_new)
                equal_peak_sanity_list.append(equal_peak_sanity)
                # Get the fraction of 1s
                frac_of_ones = len([x for x in equal_peak_sanity_list if x == 1]) / len(equal_peak_sanity_list)

        if sinusoid_df_across_trials is not None:
            # Calculate the mean per dimension
            sinusoid_df_across_trials['mean'] = sinusoid_df_across_trials.groupby(['Dimension'])['R-squared'].transform(
                'mean')
            sinusoid_df_across_trials.to_csv(
                folder_str + f'/r_squared_values_sinusoidfit_whole_cumulative_{cumulative_param}_shuffled_{shuffled_control}.csv',
                index=False)

        with open(folder_str + '/all_diagrams_h2_cumulative_trialbysegment.pkl', 'wb') as f:
            pickle.dump(all_diagrams, f)

        with open(folder_str + '/dgm_dict_h2_cumulative_trialbysegment.pkl', 'wb') as f:
            pickle.dump(dgm_dict_storage, f)

    else:
        all_diagrams = []
        for i in range(len(sorted_list)):
            sorted_data_trial = reduced_data[sorted_list[i], :]
            # Plot the persistence barcode across the whole trial
            for dim in [0, 1, 2]:
                dgm = ripser_parallel(sorted_data_trial, maxdim=2, n_threads=20, return_generators=True)
                dgm_gtda = _postprocess_diagrams([dgm["dgms"]], "ripser", (0, 1, 2), np.inf, True)
                # Remove the first axis
                dgm_gtda = dgm_gtda[0]
                plot_barcode(dgm_gtda, dim, save_dir=folder_str, count=i)

    return all_diagrams, dgm_dict_storage, sinusoid_df_across_trials



def main():
    #load the already reduced data
    base_dir = 'C:/neural_data/'
    big_list = []
    calculate_distance = False
    cumul_windows = False
    shuffle_control = False
    #check if all_diagrams.pkl exists in the base directory
    if os.path.exists(f'{base_dir}/all_diagrams.pkl') and calculate_distance:
        with open(f'{base_dir}/all_diagrams.pkl', 'rb') as f:
            big_list = pickle.load(f)

    else:
        sinusoid_df_across_trials_and_animals = pd.DataFrame()
        for subdir in [f'{base_dir}/rat_9/10-12-2021', f'{base_dir}/rat_3/25-3-2019', f'{base_dir}/rat_7/6-12-2019',
                       f'{base_dir}/rat_10/23-11-2021', f'{base_dir}/rat_8/15-10-2019', ]:
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
                #get the second most recently modified directory
                savedir = sub_folder + files[-2]
            else:
                savedir = sub_folder + files[0]

            pairs_list, _, sinusoid_df_across_trials = run_persistence_analysis(savedir, input_df,
                                                                                cumulative_param=True,
                                                                                use_peak_control=False,
                                                                                shuffled_control=shuffle_control,
                                                                                cumulative_windows=cumul_windows)
            sinusoid_df_across_trials_and_animals = pd.concat(
                [sinusoid_df_across_trials_and_animals, sinusoid_df_across_trials])

            #append pairs_list to a big_list
            big_list.append(pairs_list)
        sinusoid_df_across_trials_and_animals['mean_across_animals'] = \
        sinusoid_df_across_trials_and_animals.groupby(['Dimension'])['R-squared'].transform('mean')
        sinusoid_df_across_trials_and_animals.to_csv(
            f'{base_dir}/r_squared_values_sinusoidfit_whole_cumulative_{cumul_windows}_across_animals_shuffled_{shuffle_control}.csv', index=False)


        #calculate the bottleneck distance

    # distance_matrix_dict = calculate_bottleneck_distance(big_list, base_dir)
    #save the pairs list
    # np.save(savedir + '/pairs_list.npy', pairs_list)
def generate_white_noise_control(df, noise_level=1.0, iterations=1000, savedir = ''):
    r_squared_values = []
    for _ in range(iterations):
        noise = np.random.normal(0, noise_level, len(df))
        noise_df = df.copy()
        noise_df['death_minus_birth'] += noise
        _, r_squared_df = utils.fit_sinusoid_data_filtered(noise_df, savedir , threshold=0)
        r_squared_values.append(r_squared_df['R-squared'].mean())
    return r_squared_values


if __name__ == '__main__':
    main()
