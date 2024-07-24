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
    return



def main():
    #load the already reduced data
    base_dir = 'C:/neural_data/'
    for dir in [f'{base_dir}/rat_10/23-11-2021',f'{base_dir}/rat_8/15-10-2019', f'{base_dir}/rat_9/10-12-2021', f'{base_dir}/rat_3/25-3-2019', f'{base_dir}/rat_7/6-12-2019']:
        run_persistence_analysis(dir)










if __name__ == '__main__':
    main()
