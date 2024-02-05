import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import mat73
from pathlib import Path
import scipy



def load_data_from_paths(path):
    #load MATLAB spike data from the local path:
    spike_data = scipy.io.loadmat(path / 'units.mat')
    units = spike_data['units']
    fs = spike_data['sample_rate'][0][0]

    positional_data = scipy.io.loadmat(path / 'positionalDataByTrialType.mat')
    #load the positional data, pos key
    print(positional_data.keys())
    pos_cell = positional_data['pos']
    #access the hComb partition of the data
    hcomb_data = pos_cell[0][0][0][0]

    dlc_angle = hcomb_data['dlc_angle']
    print(spike_data.keys())



def main():
    load_data_from_paths(Path('C:/neural_data/'))



if __name__ == '__main__':
    main()