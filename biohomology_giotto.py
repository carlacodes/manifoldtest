import pickle
import os
from gph import ripser_parallel
import numpy as np
print('finished importing')
import sys


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

def run_persistence_analysis(folder_str, dimension = 2):
    pairs_list = []
    dgms_dict = {}
    for i in range(5):

        print('at count ', i)
        reduced_data = np.load(folder_str + '/X_test_transformed_fold_' + str(i) + '.npy')
        dgms = ripser_parallel(reduced_data, maxdim=dimension, n_threads=-1, return_generators=True)

        #save the individual pairs with the count
        np.save(folder_str + f'/pairs_fold_h{dimension}_giotto_myriad' + str(i) + '.npy', dgms)
          # plot_barcode(diagrams, 1)
        # plt.show()
        # plt.close('all')
        dgms_dict[i] = dgms

    with open(folder_str + f'/pairs_list_h_{dimension}_giotto_myriad.pkl', 'wb') as f:
        pickle.dump(pairs_list, f)

    return pairs_list



def main():
    #load the already reduced data
    base_dir = '/home/zceccgr/Scratch/zceccgr/honeycomb_neural_data/'
    for dimension in [2]:
        dir = sys.argv[1]  # Get data_dir from command line argument
        print('at dir', dir)
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

        pairs_list = run_persistence_analysis(savedir, dimension=dimension)
            #save the pairs list
            # np.save(savedir + '/pairs_list.npy', pairs_list)





if __name__ == '__main__':
    main()
