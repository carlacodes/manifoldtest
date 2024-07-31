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

def reformat_persistence_diagrams(dgms):

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
    return dgm
def plot_barcode(diag, dim, **kwargs):
    """
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
    plt.title(kwargs.get('title', 'Persistence Barcode'))
    plt.xlabel(kwargs.get('xlabel', 'Filtration Value'))
    plt.yticks([])
    plt.tight_layout()
    plt.show()

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
            dgm_dict[i] = dgm
            #plot the betti curve using the giotto package

            betti_curve_transformer = BettiCurve(n_bins=100, n_jobs=20)  # n_bins controls the resolution of the Betti curve
            dgms = dgm['dgms']
            reformatted_dgms = reformat_persistence_diagrams(dgms)


            betti_curves = betti_curve_transformer.fit_transform(reformatted_dgms)
            betti_curve_transformer.plot(betti_curves, sample=0, plot_method='betti')
            #save the individual persistence diagrams
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
