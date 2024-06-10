
import os
import scipy
import pickle as pkl
from sklearn.base import BaseEstimator
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
'''load previous moser grid cell data'''


def load_data():
    #load the npz data
    data_dir = 'C:/Toroidal_topology_grid_cell_data/'
    data = np.load(Path(data_dir) / 'rat_q_grid_modules_1_2.npz', allow_pickle=True)
    #extract the data, get the keys
    data_keys = data.keys()
    #extract the data
    key_list = []
    for key in data_keys:
        print(key)
        key_list.append(key)

    spikes_mod1 = data[key_list[0]]
    #convert spikes_mod1 to an array
    spikes_mod1 = spikes_mod1.flatten()
    spikes_mod1_flattened = spikes_mod1[0]

    fig, ax = plt.subplots()
    ax.plot(spikes_mod1_flattened[4])
    ax.set_title('spikes_mod1')
    plt.show()




    return




if __name__ == '__main__':
        #
    load_data()
