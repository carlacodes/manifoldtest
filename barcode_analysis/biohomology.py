import copy
from datetime import datetime
from sklearn.multioutput import MultiOutputRegressor
import pickle
import os
import ripserplusplus as rpp
import numpy as np
import matplotlib.pyplot as plt
from gph import ripser_parallel
from gtda.diagrams import BettiCurve
from gtda.homology._utils import _postprocess_diagrams
from plotly import graph_objects as go
from gtda.plotting import plot_diagram, plot_point_cloud

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
def plot_barcode(diag, dim, save_dir=None, **kwargs):
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
    plt.title(kwargs.get('title', 'Persistence Barcode, dim ' + str(dim) +'and fold ' + str(i)))
    plt.xlabel(kwargs.get('xlabel', 'Filtration Value'))
    plt.yticks([])
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(save_dir + '/barcode_fold_h2' + str(i) +'dim'+ str(dim)+'.png', dpi=300, bbox_inches='tight')
    # plt.show()


def run_persistence_analysis(folder_str, use_ripser=False):
    pairs_list = []
    dgm_dict = {}
    for i in range(5):

        print('at count ', i)
        reduced_data = np.load(folder_str + '/X_test_transformed_fold_' + str(i) + '.npy')

        if use_ripser:
            pairs = rpp.run("--format point-cloud --dim " + str(2), reduced_data)[2]
            print('pairs shape', pairs.shape)
            #append pairs to a list
            pairs_list.append(pairs)

            #
            #plot th persistence as a scatter plot
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            flattened_pairs = pairs.flatten()
            flattened_pairs = flattened_pairs[0]
            pairs_birth = pairs['birth']

            plt.scatter(pairs['birth'], pairs['death'])
            plt.xlabel('birth')
            plt.ylabel('death')
            plt.title('Persistence scatter plot')
            plt.show()
            #save the individual pairs with the count
            np.save(folder_str + '/pairs_fold_h2' + str(i) + '.npy', pairs)
        else:
            dgm = ripser_parallel(reduced_data, maxdim=2, n_threads=20, return_generators=True)
            dgm_gtda = _postprocess_diagrams([dgm["dgms"]], "ripser", (0, 1, 2), np.inf, True)
            diagram = plot_diagram(dgm_gtda[0], homology_dimensions=(0, 1,2))
            # diagram.show()
            diagram.write_html(folder_str + '/dgm_fold_h2' + str(i) + '.html')
            dgm_dict[i] = dgm
            #plot the betti curve using the giotto package
            betti_curve_transformer = BettiCurve(n_bins=1000, n_jobs=20)  # n_bins controls the resolution of the Betti curve
            betti_curves = betti_curve_transformer.fit_transform(dgm_gtda)
            fig = betti_curve_transformer.plot(betti_curves, sample=0)
            #save plotly object figure
            fig.write_html(folder_str + '/betti_curve_fold_h2_2' + str(i) + '.html')
            #save the individual persistence diagrams
            #subtract the first dimension from the second dimension
            dgm_gtda_difference = dgm_gtda[0][:,1] - dgm_gtda[0][:,0]
            dgm_gtda_copy = copy.deepcopy(dgm_gtda[0])
            dgm_gtda_difference = dgm_gtda_difference.reshape(-1, 1)

            dgm_gtda_copy = np.hstack((dgm_gtda_copy, dgm_gtda_difference))
            #convert to a dataframe
            import pandas as pd
            dgm_gtda_df = pd.DataFrame(dgm_gtda_copy, columns=['birth', 'death', 'dim', 'difference'])
            #filter for when difference is greater than 0
            dgm_gtda_df_filtered = dgm_gtda_df[dgm_gtda_df['difference'] >= 0.2]
            #plot the barcode for the filtered data, where the y axis represents the dimension
            plt.figure(figsize=(10, 6))
            # Define the vertical offset for staggering the 0-dimensional bars
            offset = 0.1
            dimension_1_base = int(len(dgm_gtda_df_filtered[dgm_gtda_df_filtered['dim'] == 0])*0.1)+10
            dimension_2_base = int(len(dgm_gtda_df_filtered[dgm_gtda_df_filtered['dim'] == 1])*0.1)+10 + dimension_1_base
            # Set a base y-position for dimension 1 to ensure it is above dimension 0

            # Initialize a dictionary to keep track of the current offset for each homology dimension
            current_offsets = {}

            # Prepare a list to store all y-positions for setting ticks later
            y_positions = []

            # Plot each bar in the barcode
            for index, row in dgm_gtda_df_filtered.iterrows():
                birth = row['birth']
                death = row['death']
                dimension = int(row['dim'])  # Convert dimension to integer if it's not already

                # Determine the y-position for the bar
                if dimension == 0:
                    # If dimension is 0, apply the staggered offset
                    if dimension not in current_offsets:
                        current_offsets[dimension] = 0  # Initialize the offset for dimension 0
                    y_position = dimension + current_offsets[dimension]
                    current_offsets[dimension] += offset  # Increment the offset for the next bar
                    color_txt = 'g-'
                elif dimension == 1:
                    # Ensure dimension 1 starts above the max of dimension 0
                    y_position = dimension_1_base
                    dimension_1_base += offset  # Increment the offset for dimension 1 to avoid overlap
                    color_txt = 'b-'
                else:
                    # For higher dimensions, no need to stagger, just use the dimension as the y-position
                    color_txt = 'r-'

                    y_position = dimension_2_base
                    dimension_2_base += offset  # Increment the offset for dimension 1 to avoid overlap


                # Plot a horizontal line for each feature
                plt.plot([birth, death], [y_position, y_position], color_txt, lw=2)

                # Store the y-position for tick labeling
                y_positions.append(y_position)

            # Fix the y-ticks to ensure correct labeling
            yticks = [0, dimension_1_base]  # Standard positions for dimensions 0 and 1
            yticklabels = ['Dimension 0', 'Dimension 1']

            plt.yticks(yticks, yticklabels)

            # Add labels and title
            plt.xlabel('Filtration Value')
            plt.ylabel('Homology Dimension')
            plt.title(f'Staggered Barcode of Filtered Persistence Diagram, fold: {i}, animal: {folder_str.split("/")[-4]}')

            # Add grid lines for better readability
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(folder_str + '/barcode_fold_filtered_h2' + str(i) + '.png', dpi=300, bbox_inches='tight')
            # Show the plot
            plt.show()
            plt.close('all')


            plot_barcode(dgm_gtda[0], 1, save_dir=folder_str)
            plot_barcode(dgm_gtda[0], 2, save_dir=folder_str)
            plot_barcode(dgm_gtda[0], 0, save_dir=folder_str)
            dgm_gtda_df_filtered.to_csv(folder_str + '/dgm_df_filtered_fold_h2' + str(i) + '.csv')
            np.save(folder_str + '/dgm_fold_h2' + str(i) + '.npy', dgm)

          # plot_barcode(diagrams, 1)
        # plt.show()
        # plt.close('all')
    #save pairs_list
    # np.save(folder_str + '/pairs_list_h2.npy', pairs_list)
    if use_ripser:
        with open(folder_str + '/pairs_list_h2.pkl', 'wb') as f:
            pickle.dump(pairs_list, f)

    #save the dgm_dict
    else:
        with open(folder_str + '/dgm_dict_h2.pkl', 'wb') as f:
            pickle.dump(dgm_dict, f)

    return pairs_list



def main():
    #load the already reduced data
    base_dir = 'C:/neural_data/'
    #f'{base_dir}/rat_7/6-12-2019', f'{base_dir}/rat_10/23-11-2021' f'{base_dir}/rat_8/15-10-2019', f'{base_dir}/rat_9/10-12-2021', f'{base_dir}/rat_3/25-3-2019'
    for dir in [ f'{base_dir}/rat_7/6-12-2019', f'{base_dir}/rat_10/23-11-2021', f'{base_dir}/rat_8/15-10-2019', f'{base_dir}/rat_9/10-12-2021', f'{base_dir}/rat_3/25-3-2019']:
        print('at dir ', dir)
        sub_folder = dir + '/plot_results/'
        #get list of files in the directory
        files = os.listdir(sub_folder)
        #check if more than two dirs
        if len(files) > 2:
            #choose the most recently modified directory
            files.sort(key=lambda x: os.path.getmtime(sub_folder + x))
            #get the most recently modified directory
            savedir = sub_folder + files[-1]
        else:
            savedir = sub_folder + files[0]

        pairs_list = run_persistence_analysis(savedir)
        #save the pairs list
        # np.save(savedir + '/pairs_list.npy', pairs_list)










if __name__ == '__main__':
    main()
