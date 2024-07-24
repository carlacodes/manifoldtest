import copy
from datetime import datetime
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from pathlib import Path
from sklearn.metrics import r2_score
from manifold_neural.helpers.datahandling import DataHandler
import matplotlib.pyplot as plt
import gudhi as gd
from umap import UMAP
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
import sys
import os
from sklearn.base import BaseEstimator
from sklearn.model_selection import BaseCrossValidator
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.manifold import Isomap
import logging


def run_persistence_analysis(folder_str):
    for i in range(5):
        reduced_data = np.load(folder_str + 'X_test_transform_fold_' + str(i) + '.npy')
        #compute the persistence barcodes
        rips_complex = gd.RipsComplex(points=reduced_data, max_edge_length=2)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
        persistence = simplex_tree.persistence()

        # Optionally, you can also plot the persistence diagram
        print('plotting persistence')
        gd.plot_persistence_diagram(persistence)
        plt.savefig(f'{folder_str}/barcode_fold_{i}.png', dpi=300, bbox_inches='tight')
        plt.show()

        # plt.show()
        # plt.close('all')
    return



def main():
    #load the already reduced data
    base_dir = 'C:/neural_data/'
    for dir in [f'{base_dir}/rat_10/23-11-2021',f'{base_dir}/rat_8/15-10-2019', f'{base_dir}/rat_9/10-12-2021', f'{base_dir}/rat_3/25-3-2019', f'{base_dir}/rat_7/6-12-2019']:
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

        run_persistence_analysis(savedir)










if __name__ == '__main__':
    main()
