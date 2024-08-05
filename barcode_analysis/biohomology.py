import copy
import pandas as pd
from datetime import datetime
from sklearn.multioutput import MultiOutputRegressor
import pickle
import os
import ripserplusplus as rpp
from gph import ripser_parallel
from gtda.diagrams import BettiCurve
from gtda.homology._utils import _postprocess_diagrams
from plotly import graph_objects as go
from gtda.plotting import plot_diagram, plot_point_cloud
from helpers.utils import create_folds
from itertools import groupby
from operator import itemgetter
import seaborn as sns
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def plot_homology_changes_heatmap(dgm_dict, save_dir):
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
    cbar = ax.collections[0].colorbar
    cbar.set_label('Death - Birth')
    plt.title(f'Homology Changes Heatmap Over Intervals for animal: {save_dir.split("/")[-4]}')
    #add a colorbar label
    plt.xlabel('Homology Dimension')
    plt.ylabel('Interval (j)')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/homology_changes_heatmap_over_intervals.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    return df


def calculate_goodness_of_fit(x_data, y_data, params):
    # Calculate the fitted values
    y_fitted = sinusoidal(x_data, *params)

    # Calculate residuals
    residuals = y_data - y_fitted

    # Sum of squared residuals (SSR)
    ssr = np.sum(residuals ** 2)

    # Total sum of squares (SST)
    sst = np.sum((y_data - np.mean(y_data)) ** 2)

    # R-squared
    r_squared = 1 - (ssr / sst)

    return r_squared

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
        plt.plot(x_data, sinusoidal(x_data, *params), 'r-', label='Fitted Function')
        plt.title(f'Sinusoidal Fit for Homology Dimension {dim}')
        plt.xlabel('Interval (j)')
        plt.ylabel('Mean Death - Birth')
        plt.legend()
        plt.show()

    # Print the fit parameters for each dimension
    for dim, params in fit_params.items():
        print(f'Dimension {dim}: A={params[0]}, B={params[1]}, C={params[2]}, D={params[3]}')

    return fit_params


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



def run_persistence_analysis(folder_str, input_df, use_ripser=False):
    pairs_list = []
    dgm_dict = {}
    sorted_list = []

    reduced_data = np.load(folder_str + '/full_set_transformed.npy')
    trial_info = input_df
    trial_indices = trial_info['trial']

    for i in range(int(len(reduced_data) / 40)):
        reduced_data_loop = reduced_data[i * 40:(i + 1) * 40]
        if len(set(trial_indices[i * 40:(i + 1) * 40])) == 1:
            sorted_list.append(list(range(i * 40, (i + 1) * 40)))
        else:
            subdivided_indices = []
            for k, g in groupby(enumerate(range(i * 40, (i + 1) * 40)), lambda ix: ix[0] - ix[1]):
                continuous_segment = list(map(itemgetter(1), g))
                trial_subdivisions = []
                for k, g in groupby(continuous_segment, key=lambda x: trial_indices[x]):
                    if len(list(g)) >= 20:
                        trial_subdivisions.append(list(g))
                subdivided_indices.extend(trial_subdivisions)
            sorted_list.extend(subdivided_indices)

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
            np.save(folder_str + '/dgm_fold_h2' + '_interval_' + str(j) + '.npy', dgm)

    df_output = plot_homology_changes_heatmap(dgm_dict, folder_str)
    fit_params = fit_sinusoid_data(df_output, folder_str)

    if use_ripser:
        with open(folder_str + '/pairs_list_h2.pkl', 'wb') as f:
            pickle.dump(pairs_list, f)
    else:
        with open(folder_str + '/dgm_dict_h2.pkl', 'wb') as f:
            pickle.dump(dgm_dict, f)

    return pairs_list

# def run_persistence_analysis(folder_str, input_df, use_ripser=False):
#     pairs_list = []
#     dgm_dict = {}
#     sorted_list = []
#
#
#     reduced_data = (
#         np.load(folder_str + '/full_set_transformed.npy'))
#     #import the folds
#     # folds_data = pd.read_csv(folder_str + '/custom_folds.csv')
#     # #load the test indices
#     # fold_data = folds_data['train'][i]
#     # #convert this to an int array
#     # #remove the first bracket and the last bracket
#     # fold_data = fold_data.strip('[]')  # Remove brackets
#     #
#     # fold_data = np.array(fold_data.split(',')).astype(int)
#     # #load the corresponding trial info
#     trial_info = input_df
#     trial_indices = trial_info['trial']
#     #sort the data into segments of 40 smamples and check they are all part of the same trial
#
#
#     for i in range(int(len(reduced_data)/ 40)):
#         reduced_data_loop = reduced_data[i*40:(i+1)*40]
#         #check they are all part of the same trial
#         if len(set(trial_indices[i*40:(i+1)*40])) == 1:
#             sorted_list.append(list(range(i * 40, (i + 1) * 40)))
#         else:
#             #subdivide the segments into continuous segments
#             subdivided_indices = []
#             for k, g in groupby(enumerate(range(i * 40, (i + 1) * 40)), lambda ix: ix[0] - ix[1]):
#                 continuous_segment = list(map(itemgetter(1), g))
#                 # Further subdivide by trial indices
#                 trial_subdivisions = []
#                 for k, g in groupby(continuous_segment, key=lambda x: trial_indices[x]):
#                     if len(list(g)) >= 20:
#                         trial_subdivisions.append(list(g))
#                 subdivided_indices.extend(trial_subdivisions)
#             sorted_list.extend(subdivided_indices)
#
#     #remove the empty lists
#     sorted_list = [x for x in sorted_list if x != []]
#
#
#
#     for j in range(len(sorted_list)):
#         reduced_data_loop = reduced_data[sorted_list[j], :]
#         if use_ripser:
#             pairs = rpp.run("--format point-cloud --dim " + str(2), reduced_data_loop)[2]
#             print('pairs shape', pairs.shape)
#             #append pairs to a list
#             pairs_list.append(pairs)
#
#             #
#             #plot th persistence as a scatter plot
#             fig, ax = plt.subplots(1, 3, figsize=(15, 5))
#             flattened_pairs = pairs.flatten()
#             flattened_pairs = flattened_pairs[0]
#             pairs_birth = pairs['birth']
#
#             plt.scatter(pairs['birth'], pairs['death'])
#             plt.xlabel('birth')
#             plt.ylabel('death')
#             plt.title('Persistence scatter plot')
#             # plt.show()
#             #save the individual pairs with the count
#             np.save(folder_str + '/pairs_fold_h2' + str(j) + '.npy', pairs)
#         else:
#             dgm = ripser_parallel(reduced_data_loop, maxdim=2, n_threads=20, return_generators=True)
#             dgm_gtda = _postprocess_diagrams([dgm["dgms"]], "ripser", (0, 1, 2), np.inf, True)
#             diagram = plot_diagram(dgm_gtda[0], homology_dimensions=(0, 1,2))
#             # diagram.show()
#             diagram.write_html(folder_str + '/dgm_fold_h2_fold'+ '_interval_' + str(j) + '.html')
#             dgm_dict[i] = dgm
#             #plot the betti curve using the giotto package
#             betti_curve_transformer = BettiCurve(n_bins=1000, n_jobs=20)  # n_bins controls the resolution of the Betti curve
#             betti_curves = betti_curve_transformer.fit_transform(dgm_gtda)
#             fig = betti_curve_transformer.plot(betti_curves, sample=0)
#             #save plotly object figure
#             fig.write_html(folder_str + '/betti_curve_fold_h2_fold' + '_interval_' + str(j) + '.html')
#             #save the individual persistence diagrams
#             #subtract the first dimension from the second dimension
#             dgm_gtda_difference = dgm_gtda[0][:,1] - dgm_gtda[0][:,0]
#             dgm_gtda_copy = copy.deepcopy(dgm_gtda[0])
#             dgm_gtda_difference = dgm_gtda_difference.reshape(-1, 1)
#
#             dgm_gtda_copy = np.hstack((dgm_gtda_copy, dgm_gtda_difference))
#             #convert to a dataframe
#             dgm_gtda_df = pd.DataFrame(dgm_gtda_copy, columns=['birth', 'death', 'dim', 'difference'])
#             #filter for when difference is greater than 0
#             dgm_gtda_df_filtered = dgm_gtda_df[dgm_gtda_df['difference'] >= 0.2]
#             #plot the barcode for the filtered data, where the y axis represents the dimension
#             plt.figure(figsize=(10, 6))
#             # Define the vertical offset for staggering the 0-dimensional bars
#             offset = 0.1
#             dimension_1_base = int(len(dgm_gtda_df_filtered[dgm_gtda_df_filtered['dim'] == 0])*0.1)+10
#             dimension_2_base = int(len(dgm_gtda_df_filtered[dgm_gtda_df_filtered['dim'] == 1])*0.1)+10 + dimension_1_base
#             # Set a base y-position for dimension 1 to ensure it is above dimension 0
#
#             # Initialize a dictionary to keep track of the current offset for each homology dimension
#             current_offsets = {}
#
#             # Prepare a list to store all y-positions for setting ticks later
#             y_positions = []
#
#             # Plot each bar in the barcode
#             for index, row in dgm_gtda_df_filtered.iterrows():
#                 birth = row['birth']
#                 death = row['death']
#                 dimension = int(row['dim'])  # Convert dimension to integer if it's not already
#
#                 # Determine the y-position for the bar
#                 if dimension == 0:
#                     # If dimension is 0, apply the staggered offset
#                     if dimension not in current_offsets:
#                         current_offsets[dimension] = 0  # Initialize the offset for dimension 0
#                     y_position = dimension + current_offsets[dimension]
#                     current_offsets[dimension] += offset  # Increment the offset for the next bar
#                     color_txt = 'g-'
#                 elif dimension == 1:
#                     # Ensure dimension 1 starts above the max of dimension 0
#                     y_position = dimension_1_base
#                     dimension_1_base += offset  # Increment the offset for dimension 1 to avoid overlap
#                     color_txt = 'b-'
#                 else:
#                     # For higher dimensions, no need to stagger, just use the dimension as the y-position
#                     color_txt = 'r-'
#
#                     y_position = dimension_2_base
#                     dimension_2_base += offset  # Increment the offset for dimension 1 to avoid overlap
#
#
#                 # Plot a horizontal line for each feature
#                 plt.plot([birth, death], [y_position, y_position], color_txt, lw=2)
#
#                 # Store the y-position for tick labeling
#                 y_positions.append(y_position)
#
#             # Fix the y-ticks to ensure correct labeling
#             yticks = [0, dimension_1_base]  # Standard positions for dimensions 0 and 1
#             yticklabels = ['Dimension 0', 'Dimension 1']
#
#             plt.yticks(yticks, yticklabels)
#
#             # Add labels and title
#             plt.xlabel('Filtration Value')
#             plt.ylabel('Homology Dimension')
#             plt.title(f'Staggered Barcode of Filtered Persistence Diagram, interval: {j}, animal: {folder_str.split("/")[-4]}')
#
#             # Add grid lines for better readability
#             plt.grid(True, linestyle='--', alpha=0.7)
#             plt.savefig(folder_str + '/barcode_fold_filtered_h2_interval' + str(j) + '.png', dpi=300, bbox_inches='tight')
#             # Show the plot
#             # plt.show()
#             plt.close('all')
#
#
#             plot_barcode(dgm_gtda[0], 1,fold = i,  save_dir=folder_str)
#             plot_barcode(dgm_gtda[0], 2,fold=i, save_dir=folder_str)
#             plot_barcode(dgm_gtda[0], 0, fold= i, save_dir=folder_str)
#             dgm_gtda_df_filtered.to_csv(folder_str + '/dgm_df_filtered_fold_h2' + '_interval_' + str(j) + '.csv')
#             np.save(folder_str + '/dgm_fold_h2' + '_interval_' + str(j) + '.npy', dgm)
#
#           # plot_barcode(diagrams, 1)
#         # plt.show()
#         # plt.close('all')
#     #save pairs_list
#     # np.save(folder_str + '/pairs_list_h2.npy', pairs_list)
#     if use_ripser:
#         with open(folder_str + '/pairs_list_h2.pkl', 'wb') as f:
#             pickle.dump(pairs_list, f)
#
#     #save the dgm_dict
#     else:
#         with open(folder_str + '/dgm_dict_h2.pkl', 'wb') as f:
#             pickle.dump(dgm_dict, f)
#
#     return pairs_list


def main():
    #load the already reduced data
    base_dir = 'C:/neural_data/'
    #f'{base_dir}/rat_7/6-12-2019', f'{base_dir}/rat_10/23-11-2021' f'{base_dir}/rat_8/15-10-2019', f'{base_dir}/rat_9/10-12-2021', f'{base_dir}/rat_3/25-3-2019'
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

        pairs_list = run_persistence_analysis(savedir, input_df)
        #save the pairs list
        # np.save(savedir + '/pairs_list.npy', pairs_list)










if __name__ == '__main__':
    main()
