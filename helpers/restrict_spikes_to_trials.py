import os
import numpy as np
import pandas as pd
import pickle
from get_directories import get_data_dir
# from process_dlc_data import load_dlc_processed_pickle
from load_and_save_data import save_pickle, load_pickle


def restrict_times_to_interval(times_to_restrict, interval):
    restricted_times = times_to_restrict[(times_to_restrict >=
                                          interval[0]) & (times_to_restrict <= interval[1])]
    return restricted_times


def restrict_spikes_to_trials(units, dlc_data):
    restricted_units = {}

    for u in units.keys():
        # get the spike times for the unit

        spike_times = units[u]
        if not isinstance(spike_times, np.ndarray):
            spike_times = spike_times['samples']

        restricted_units[u] = {}

        for t in dlc_data.keys():
            # get the start and end times of the trial
            start_time = dlc_data[t]['video_samples'].iloc[0]
            end_time = dlc_data[t]['video_samples'].iloc[-1]

            # get the spike times that are in the trial
            spike_times_in_trial = restrict_times_to_interval(spike_times,
                                                              [start_time, end_time])

            # put spike times in restricted_units dictionary
            restricted_units[u][t] = spike_times_in_trial

    return restricted_units


if __name__ == "__main__":
    # animal = 'Rat65'
    # session = '10-11-2023'
    # data_dir = get_data_dir(animal, session)

    # data_dir = 'D:/analysis/og_honeycomb/rat7/6-12-2019'
    # data_dir = '/media/jake/DataStorage_6TB/DATA/neural_network/og_honeycomb/rat7/6-12-2019'
    big_dir = 'C:/neural_data/'

    for rat in [3, 8, 9, 10]:
        #get the list of folders directory that have dates
        dates = os.listdir(os.path.join(big_dir, f'rat_{rat}'))
        #check if the folder name is a date by checking if it contains a hyphen
        date = [d for d in dates if '-' in d][0]
        data_dir = os.path.join(big_dir, f'rat_{rat}', date)

    # load the dlc data, which contains the trial times
    # dlc_dir = os.path.join(data_dir, 'deeplabcut')
    # dlc_data = load_pickle('dlc_final', dlc_dir)

        pos_dir = os.path.join(data_dir, 'positional_data')
        dlc_data = load_pickle('dlc_data', pos_dir)

        # load the spike data
        # unit_dir = os.path.join(data_dir, 'spike_sorting')

        unit_dir = os.path.join(data_dir, 'physiology_data')
        units = load_pickle('unit_spike_times', unit_dir)
        units = units['pyramid']

        restricted_units = restrict_spikes_to_trials(units, dlc_data['hComb'])
        save_pickle(restricted_units, 'restricted_units', unit_dir)

    pass