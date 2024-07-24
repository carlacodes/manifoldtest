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
        #comput the persistence barcodes



def main():
    #load the already reduced data
    base_folder = 'C:/neural_data/'
    for dir in os.listdir(base_folder):
        if os.path.isdir(base_folder + dir):
            print(dir)
            run_persistence_analysis(base_folder + dir + '/')









if __name__ == '__main__':
    main()
