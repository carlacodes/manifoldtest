import copy
import gtda.diagrams
import pandas as pd
import os
from gph import ripser_parallel
from gtda.homology._utils import _postprocess_diagrams
import seaborn as sns
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from helpers import utils
from persim import bottleneck
import pickle
import multiprocessing as mp
import numpy as np
import pickle
from persim import bottleneck

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


def calculate_bottleneck_distance_nonpara(all_diagrams, folder_str):
    #concatenate the diagrams all together into one mega list
    print('..calculating distance matrix')
    mega_diagram_list = []
    for i in range(len(all_diagrams)):
        diagram = all_diagrams[i]
        mega_diagram_list.extend(diagram)

    # Stack diagrams into a single ndarray
    num_diagrams = len(mega_diagram_list)

    distance_matrix_dict = {}
    pair_list = []
    for l in [0, 1, 2]:
        distance_matrix = np.zeros((num_diagrams, num_diagrams)) + np.nan
        for m in range(num_diagrams):
            for n in range(m + 1, num_diagrams):
                if m == n:
                    continue
                elif (n, m) in pair_list:
                    continue
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
                distance_matrix[n, m] = distance_matrix[m, n]
                pair_list.append((m,n))

        # Save the distance matrix
        #remove the diagonal from the matrix
        distance_matrix = np.triu(distance_matrix)
        distance_matrix_dict[l] = distance_matrix
    with open(folder_str + '/distance_matrix_dict.pkl', 'wb') as f:
        pickle.dump(distance_matrix_dict, f)
    #remove the diagonal from the matrix

    return distance_matrix_dict


def compute_distance(args):
    m, n, l, mega_diagram_list = args
    first_array = np.squeeze(mega_diagram_list[m])
    first_array = first_array[first_array[:, 2] == l][:, 0:2]
    second_array = np.squeeze(mega_diagram_list[n])
    second_array = second_array[second_array[:, 2] == l][:, 0:2]
    return m, n, l, bottleneck(first_array, second_array)

def calculate_bottleneck_distance(all_diagrams, folder_str, num_threads=4):
    """
    Calculate the bottleneck distance matrix for all diagrams.

    Parameters
    ----------
    all_diagrams: list
        List of all persistence diagrams.
    folder_str: str
        Directory to save the distance matrix.
    num_threads: int
        Number of threads to use for parallel computation.

    Returns
    -------
    dict
        Dictionary containing the distance matrices for each dimension.
    """
    print('..calculating distance matrix')
    mega_diagram_list = [diagram for diagrams in all_diagrams for diagram in diagrams]
    num_diagrams = len(mega_diagram_list)
    distance_matrix_dict = {}

    for l in [0, 1, 2]:
        distance_matrix = np.full((num_diagrams, num_diagrams), np.nan)
        with mp.Pool(processes=num_threads) as pool:
            args = [(m, n, l, mega_diagram_list) for m in range(num_diagrams) for n in range(m + 1, num_diagrams)]
            results = pool.map(compute_distance, args)
            for m, n, l, distance in results:
                distance_matrix[m, n] = distance
                distance_matrix[n, m] = distance
        distance_matrix_dict[l] = distance_matrix

    with open(f'{folder_str}/distance_matrix_dict.pkl', 'wb') as f:
        pickle.dump(distance_matrix_dict, f)

    return distance_matrix_dict

def run_persistence_analysis(folder_str, input_df, use_ripser=False, segment_length=40, cumulative_param=True):
    pairs_list = []
    dgm_dict_storage = {}
    distance_matrix_dict = {}  # Dictionary to store pairwise distances
    sorted_list = []

    reduced_data = np.load(folder_str + '/full_set_transformed.npy')
    trial_info = input_df
    trial_indices = trial_info['trial']

    # Get the sorted list
    sorted_list = process_data(reduced_data, trial_indices, segment_length, cumulative=cumulative_param)

    if cumulative_param:
        all_diagrams = []  # List to store all persistence diagrams
        for i in range(len(sorted_list)):
            dgm_dict = {}
            sorted_data_trial = reduced_data[sorted_list[i], :]
            # Break each sorted_data_trial into chunks of segment_length

            dgm = ripser_parallel(sorted_data_trial, maxdim=2, n_threads=20, return_generators=True)
            dgm_gtda = _postprocess_diagrams([dgm["dgms"]], "ripser", (0, 1, 2), np.inf, True)
            dgm_dict[i] = dgm


            all_diagrams.append(dgm_gtda)  # Collect diagrams for distance calculation

            np.save(folder_str + '/dgm_fold_h2' + '_interval_' + str(i) + f'_cumulative_{cumulative_param}.npy',
                    dgm_dict)

            # df_output = plot_homology_changes_heatmap(dgm_dict, folder_str, cumulative_param=cumulative_param,
            #                                           trial_number=i)
            # fit_params = utils.fit_sinusoid_data_whole(df_output, folder_str, cumulative_param=cumulative_param)


        with open(folder_str + '/all_diagrams_h2_cumulative_trialbysegment.pkl', 'wb') as f:
            pickle.dump(all_diagrams, f)

        with open(folder_str + '/dgm_dict_h2_cumulative_trialbysegment.pkl', 'wb') as f:
            pickle.dump(dgm_dict_storage, f)


    return all_diagrams, dgm_dict_storage




if __name__ == '__main__':
    #load the already reduced data
    base_dir = 'C:/neural_data/'
    big_list = []
    #check if all_diagrams.pkl exists in the base directory
    # if os.path.exists(f'{base_dir}/all_diagrams.pkl'):
    #     with open(f'{base_dir}/all_diagrams.pkl', 'rb') as f:
    #         big_list = pickle.load(f)
    #
    # else:

    for subdir in [f'{base_dir}/rat_7/6-12-2019', f'{base_dir}/rat_10/23-11-2021', f'{base_dir}/rat_8/15-10-2019', f'{base_dir}/rat_9/10-12-2021', f'{base_dir}/rat_3/25-3-2019']:
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

        pairs_list, _ = run_persistence_analysis(savedir, input_df, cumulative_param=True)
        distance_matrix_dict = calculate_bottleneck_distance_nonpara(pairs_list, base_dir)

        #append pairs_list to a big_list
        #save to pkl file
        with open(f'{subdir}/distance_matrix_dict.pkl', 'wb') as f:
            pickle.dump(distance_matrix_dict, f)
        # big_list.append(pairs_list)


        #calculate the bottleneck distance

    # distance_matrix_dict = calculate_bottleneck_distance(big_list, base_dir)
        #save the pairs list
        # np.save(savedir + '/pairs_list.npy', pairs_list)






