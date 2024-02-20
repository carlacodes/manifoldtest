# code for generating the dataset for use with PyTorch
import os
import numpy as np
import pandas as pd

from get_directories import get_data_dir, get_robot_maze_directory
from load_and_save_data import load_pickle, save_pickle
from calculate_spike_pos_hd import interpolate_rads

sample_freq = 30000  # Hz


def smooth_positional_data(dlc_data, window_size=100):
    # THIS IS JUST A PLACEHOLDER FOR NOW
    pass


def create_positional_trains(dlc_data, window_size=100):  # we'll default to 100 ms windows for now
    # first, get the start and end time of the video
    window_in_samples = window_size * sample_freq / 1000  # convert window size to samples
    windowed_dlc = {}
    window_edges = {}
    for k in dlc_data.keys():
        start_time = dlc_data[k].video_samples.iloc[0]
        end_time = dlc_data[k].video_samples.iloc[-1]
        duration = end_time - start_time

        # calculate the number of windows. Windows overlap by 50%.
        num_windows = int(np.floor(duration / (window_in_samples / 2))) - 1

        # get the bin edges for the windows, these will be returned and used to bin the spikes
        window_edges[k] = np.int64([start_time + i * window_in_samples / 2 for i in range(num_windows + 2)])

        # calculate the window centres. Because of the 50% overlap, they are simply the
        # window edges with the first and last values excluded
        window_centres = window_edges[k][1:-1]

        # create a dataframe and make window_centres the video_samples column
        windowed_dlc[k] = pd.DataFrame(window_centres, columns=['video_samples'])

        # interpolate the x and y position, and goal distances data using window_centres
        cols_to_lin_interp = ['x', 'y']
        # find the cols that begin "distance_to_goal_", but don't end "platform"
        # and add them to the list
        cols_to_lin_interp.extend([col for col in dlc_data[k].columns if \
                                   col.startswith('distance_to_goal_') and not \
                                       col.endswith('platform')])

        for c in cols_to_lin_interp:
            # make the column c in windowed_dlc[k] equal to the linearly interpolated values
            # of the column c in dlc_data[k] at the window_centres
            windowed_dlc[k][c] = np.round(np.interp(window_centres, \
                                                    dlc_data[k].video_samples, dlc_data[k][c]), 1)

        # use interpolate_rads to interpolate the head direction data, and data in columns that
        # begin "relative_direction" but not including those that begin "relative_direction_to"
        cols_to_rad_interp = ['hd']
        cols_to_rad_interp.extend([col for col in dlc_data[k].columns if \
                                   col.startswith('relative_direction_') and not \
                                       col.startswith('relative_direction_to')])

        for c in cols_to_rad_interp:
            windowed_dlc[k][c] = np.round(interpolate_rads(dlc_data[k].video_samples, \
                                                           dlc_data[k][c], window_centres), 2)

    return windowed_dlc, window_edges, window_size


def create_spike_trains(units, window_edges, window_size):
    # windows overlap by 50%
    # window_size in ms - just hard coded, not checked, so be careful!!!!!

    # create a dictionary to hold the spike trains
    spike_trains = {}

    for i, k in enumerate(window_edges.keys()):

        # create the time bins. Starting times for each bin are from the first to the
        # third to last window edge, with the last window edge excluded. The end times
        # start from the third window edge and go to the last window edge, with the first
        # two edges exluded.

        for u in units.keys():

            if i == 0:
                spike_trains[u] = {}

            # get the spike times for the unit
            spike_times = units[u][k]

            if not isinstance(spike_times, np.ndarray):
                spike_times = spike_times['samples']

                # bin spike times into the windows
            binned_spikes = np.histogram(spike_times, window_edges[k])[0]
            # make a copy of binned spikes
            binned_spikes_copy = binned_spikes.copy()

            # the two copies are offset by half the window size
            # and added together to produce the ovelapping windows
            # (i.e. the new first bin is bin1 + bin2, the new second
            # bin is bin2 + bin3, etc.) without resorting to a slow
            # for loop
            # remove the last bin of binned spike
            binned_spikes = binned_spikes[:-1]
            # remove the first bin of the copy
            binned_spikes_copy = binned_spikes_copy[1:]
            # add the two together
            binned_spikes = binned_spikes + binned_spikes_copy

            spike_rate = binned_spikes / (window_size / 1000)
            spike_trains[u][k] = spike_rate

    return spike_trains


def cat_dlc(windowed_dlc):
    # concatenate data from all trials into np.arrays for training
    # we will keep columns x, y, and the 2 distance to goal columns.
    # the hd and relative_direction columns (but relative_direction_to columns)
    # will be converted to sin and cos and concatenated with the other data

    for i, k in enumerate(windowed_dlc):
        if i == 0:
            # get the column names
            columns = windowed_dlc[k].columns

            # find the distance to goal columns
            distance_cols = [c for c in columns if c.startswith('distance_to_goal_')]

            # find the relative direction columns (but not relative_direction_to columns)
            relative_direction_cols = [c for c in columns if c.startswith('relative_direction_') \
                                       and not c.startswith('relative_direction_screen')]

            column_names = ['x', 'y']
            column_names.extend(distance_cols)
            column_names.extend(['hd'])
            column_names.extend(relative_direction_cols)

            total_num_cols = 10  # x, y, distance_to_goal x2, hd x2, relative_direction x 4

        # get the number of rows in the dataframe
        num_rows = len(windowed_dlc[k])

        # create an empty np.array of the correct size
        temp_array = np.zeros((num_rows, total_num_cols))

        count = 0
        for c in column_names:
            # if c not hd or relative_direction, just add the column to the array
            if c not in ['hd'] and not c.startswith('relative_direction'):
                temp_array[:, count] = windowed_dlc[k][c].values
                count += 1

            else:
                # angular data needs to be converted to sin and cos
                temp_array[:, count] = np.sin(windowed_dlc[k][c].values)
                temp_array[:, count + 1] = np.cos(windowed_dlc[k][c].values)
                count += 2

        if i == 0:
            dlc_array = temp_array.copy()
        else:
            dlc_array = np.concatenate((dlc_array, temp_array), axis=0)

    # all columns need to be scaled to the range 0-1
    for i in range(dlc_array.shape[1]):
        dlc_array[:, i] = (dlc_array[:, i] - np.min(dlc_array[:, i])) / \
                          (np.max(dlc_array[:, i]) - np.min(dlc_array[:, i]))

    dlc_array = np.round(dlc_array, 3)

    return dlc_array


def cat_spike_trains(spike_trains):
    # get list of units
    unit_list = list(spike_trains.keys())
    n_units = len(unit_list)

    for i, k in enumerate(spike_trains[unit_list[0]].keys()):
        for j, u in enumerate(unit_list):
            if j == 0:
                # get the number of rows in the dataframe
                num_cols = len(spike_trains[u][k])

                # create an empty np.array of the correct size
                temp_array = np.zeros((n_units, num_cols))

            # add the spike trains to the array
            temp_array[j, :] = spike_trains[u][k]

        if i == 0:
            spike_array = temp_array.copy()

        else:
            spike_array = np.concatenate((spike_array, temp_array), axis=1)

    spike_array = np.round(spike_array, 3)
    spike_array = np.transpose(spike_array)

    return spike_array, unit_list


if __name__ == "__main__":
    # animal = 'Rat65'
    # session = '10-11-2023'
    # data_dir = get_data_dir(animal, session)

    data_dir = 'D:/analysis/og_honeycomb/rat7/6-12-2019'

    # data_dir = '/media/jake/DataStorage_6TB/DATA/neural_network/og_honeycomb/rat7/6-12-2019'

    # load spike data
    # spike_dir = os.path.join(data_dir, 'spike_sorting')
    # units = load_pickle('units_w_behav_correlates', spike_dir)

    spike_dir = os.path.join(data_dir, 'physiology_data')
    units = load_pickle('restricted_units', spike_dir)

    # load positional data
    # dlc_dir = os.path.join(data_dir, 'deeplabcut')
    # dlc_data = load_pickle('dlc_final', dlc_dir)

    dlc_dir = os.path.join(data_dir, 'positional_data')
    dlc_data = load_pickle('dlc_data', dlc_dir)
    dlc_data = dlc_data['hComb']

    # create positional and spike trains with overlapping windows
    # and save as a pickle file
    windowed_dlc, window_edges, window_size = \
        create_positional_trains(dlc_data, window_size=500)
    windowed_data = {'windowed_dlc': windowed_dlc, 'window_edges': window_edges}
    save_pickle(windowed_data, 'windowed_data', dlc_dir)

    windowed_data = load_pickle('windowed_data', dlc_dir)
    windowed_dlc = windowed_data['windowed_dlc']
    window_edges = windowed_data['window_edges']

    # create spike trains
    spike_trains = create_spike_trains(units, window_edges, window_size=window_size)
    save_pickle(spike_trains, 'spike_trains', spike_dir)

    spike_trains = load_pickle('spike_trains', spike_dir)

    # concatenate data from all trials into np.arrays for training
    labels = cat_dlc(windowed_dlc)
    # convert labels to float32
    labels = labels.astype(np.float32)
    np.save(f'{dlc_dir}/labels.npy', labels)

    # concatenate spike trains into np.arrays for training
    model_inputs, unit_list = cat_spike_trains(spike_trains)
    # convert model_inputs to float32
    model_inputs = model_inputs.astype(np.float32)
    np.save(f'{spike_dir}/inputs.npy', model_inputs)
    save_pickle(unit_list, 'unit_list', spike_dir)
    pass