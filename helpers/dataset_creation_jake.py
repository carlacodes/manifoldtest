# code for generating the dataset for use with PyTorch
import os
import numpy as np
import pandas as pd
from get_directories import get_data_dir, get_robot_maze_directory
from load_and_save_data import load_pickle, save_pickle
from calculate_spike_pos_hd import interpolate_rads
sample_freq = 30000  # Hz


import numpy as np
def create_positional_trains_no_overlap(dlc_data, window_size=100):  # we'll default to 100 ms windows for now
    # first, get the start and end time of the video
    window_in_samples = window_size * sample_freq / 1000  # convert window size to samples
    windowed_dlc = {}
    window_edges = {}
    for k in dlc_data.keys():
        start_time = dlc_data[k].video_samples.iloc[0]
        end_time = dlc_data[k].video_samples.iloc[-1]
        duration = end_time - start_time

        # calculate the number of windows. No overlap.
        num_windows = int(np.floor(duration / window_in_samples))

        # get the bin edges for the windows, these will be returned and used to bin the spikes
        window_edges[k] = np.int64([start_time + i * window_in_samples for i in range(num_windows + 1)])

        # calculate the window centres. No overlap, so they are simply the window edges with the last value excluded
        window_centres = window_edges[k][:-1]

        # create a dataframe and make window_centres the video_samples column
        windowed_dlc[k] = pd.DataFrame(window_centres, columns=['video_samples'])

        # interpolate the x and y position, and goal distances data using window_centres
        cols_to_lin_interp = ['x', 'y']
        # find the cols that begin "distance_to_goal_", but don't end "platform"
        # and add them to the list
        cols_to_lin_interp.extend([col for col in dlc_data[k].columns if \
                                   col.startswith('dist2') and not \
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
                                   col.startswith('dist2') and not \
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
    #remove the sample_rate from the units dictionary

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

def create_spike_trains_no_overlap(units, window_edges, window_size):
    # No overlap.
    # window_size in ms - just hard coded, not checked, so be careful!!!!!

    # create a dictionary to hold the spike trains
    spike_trains = {}

    for i, k in enumerate(window_edges.keys()):

        for u in units.keys():

            if i == 0:
                spike_trains[u] = {}

            # get the spike times for the unit
            spike_times = units[u][k]

            if not isinstance(spike_times, np.ndarray):
                spike_times = spike_times['samples']

            # bin spike times into the windows
            binned_spikes = np.histogram(spike_times, window_edges[k])[0]

            spike_rate = binned_spikes / (window_size / 1000)
            spike_trains[u][k] = spike_rate

    return spike_trains

# def create_spike_trains_with_array(units, window_size = 20, overlap = 10):
#     # window_size and overlap in ms - just hard coded, not checked, so be careful!!!!!
#     # Create a list to hold the spike trains
#     spike_trains = []
#
#     # Iterate over trials
#     for spike_times_dict in units.values():
#         trial_spike_trains = []
#
#         # Iterate over neurons
#         for spike_times in spike_times_dict.values():
#             # get the spike times for the unit
#
#             if not isinstance(spike_times, np.ndarray):
#                 spike_times = spike_times['samples']
#
#             # Calculate the number of windows with overlap
#             num_windows = int((len(spike_times) - window_size) / overlap) + 1
#
#             # Bin spike times into overlapping windows
#             binned_spikes = [np.histogram(spike_times[i:i+window_size], bins=window_size)[0]
#                              for i in range(0, len(spike_times) - window_size + 1, overlap)]
#
#             if len(binned_spikes) > 0:
#                 unit_spike_rate = np.vstack(binned_spikes) / (window_size / 1000)
#                 trial_spike_trains.append(unit_spike_rate)
#
#         # Ensure all arrays have the same shape before stacking
#         if len(trial_spike_trains) > 0:
#             max_shape = max(arr.shape for arr in trial_spike_trains)
#             trial_spike_trains_padded = [np.pad(arr, ((0, max_shape[0] - arr.shape[0]), (0, max_shape[1] - arr.shape[1])))
#                                          for arr in trial_spike_trains]
#             spike_trains.append(np.stack(trial_spike_trains_padded, axis=-1))
#
#     if len(spike_trains) == 0:
#         raise ValueError("No spike data available for the specified window size and overlap.")
#     result = np.stack(spike_trains, axis=0)
#     return np.stack(spike_trains, axis=0)



# def create_spike_trains_merge_into_trial(units, window_size):
#     #declare a 3D array of trials x time x neurons
#     #create a dictionary to hold the spike trains
#     #get the number of units for the thrid dimension
#     num_units = len(units)
#     cumulative_list = []
#     # for key in units:
#     #     example_unit = units[key]
#     #     for trial in example_unit:
#     #         spike_times = example_unit[trial]
#     #         if not isinstance(spike_times, np.ndarray):
#     #             spike_times = spike_times['samples']
#     #         # bin spike times into the windows
#     #         binned_spikes = np.histogram(spike_times, window_size)[0]
#     #         spike_rate = binned_spikes / (window_size / 1000)
#     #         cumulative_list.append(spike_rate)
#     #     # windows overlap by 50%
#     #     # window_size in ms - just hard coded, not checked, so be careful!!!!!
#
#         # create a dictionary to hold the spike trains
#     spike_trains = {}
#     # remove the sample_rate from the units dictionary
#
#     for i, k in enumerate(window_edges.keys()):
#
#         # create the time bins. Starting times for each bin are from the first to the
#         # third to last window edge, with the last window edge excluded. The end times
#         # start from the third window edge and go to the last window edge, with the first
#         # two edges exluded.
#
#         for u in units.keys():
#
#             if i == 0:
#                 spike_trains[u] = {}
#
#             # get the spike times for the unit
#             spike_times = units[u][k]
#
#             if not isinstance(spike_times, np.ndarray):
#                 spike_times = spike_times['samples']
#
#                 # bin spike times into the windows
#             binned_spikes = np.histogram(spike_times, window_edges[k])[0]
#             # make a copy of binned spikes
#             binned_spikes_copy = binned_spikes.copy()
#
#             # the two copies are offset by half the window size
#             # and added together to produce the ovelapping windows
#             # (i.e. the new first bin is bin1 + bin2, the new second
#             # bin is bin2 + bin3, etc.) without resorting to a slow
#             # for loop
#             # remove the last bin of binned spike
#             binned_spikes = binned_spikes[:-1]
#             # remove the first bin of the copy
#             binned_spikes_copy = binned_spikes_copy[1:]
#             # add the two together
#             binned_spikes = binned_spikes + binned_spikes_copy
#
#             spike_rate = binned_spikes / (window_size / 1000)
#             spike_trains[u][k] = spike_rate
#
#     return example_unit


def create_spike_trains_trial_binning(units, window_edges, window_size):
    # Overlap.
    # window_size in ms - just hard coded, not checked, so be careful!!!!!

    # create a dictionary to hold the spike trains
    spike_trains = {}

    for i, k in enumerate(window_edges.keys()):

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
            # and added together to produce the overlapping windows
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

def cat_dlc(windowed_dlc, include_raw_hd = True, scale_data = False, z_score_data = True):
    # concatenate data from all trials into np.arrays for training
    # we will keep columns x, y, and the 2 distance to goal columns.
    # the hd and relative_direction columns (but relative_direction_to columns)
    # will be converted to sin and cos and concatenated with the other data

    for i, k in enumerate(windowed_dlc):
        if i == 0:
            # get the column names
            columns = windowed_dlc[k].columns

            # find the distance to goal columns
            distance_cols = [c for c in columns if c.startswith('dist')]

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

            elif include_raw_hd:
                # angular data needs to be converted to sin and cos
                temp_array[:, count] = np.sin(windowed_dlc[k][c].values)
                temp_array[:, count + 1] = np.cos(windowed_dlc[k][c].values)
                temp_array[:, count + 2] = windowed_dlc[k][c].values
                count += 3
            else:
                temp_array[:, count] = np.sin(windowed_dlc[k][c].values)
                temp_array[:, count + 1] = np.cos(windowed_dlc[k][c].values)
                count += 2




        if i == 0:
            dlc_array = temp_array.copy()
        else:
            dlc_array = np.concatenate((dlc_array, temp_array), axis=0)

    # all columns need to be scaled to the range 0-1
    if scale_data:
        for i in range(dlc_array.shape[1]):
            dlc_array[:, i] = (dlc_array[:, i] - np.min(dlc_array[:, i])) / \
                              (np.max(dlc_array[:, i]) - np.min(dlc_array[:, i]))
    elif z_score_data:
        epsilon = 1e-10
        for i in range(dlc_array.shape[1]):
            dlc_array[:, i] = (dlc_array[:, i] - np.mean(dlc_array[:, i])) / \
                              (np.std(dlc_array[:, i]) + epsilon)

    dlc_array = np.round(dlc_array, 3)

    return dlc_array, column_names


def cat_dlc_modified(windowed_dlc, length_size, include_raw_hd=True, scale_data=False, z_score_data=True):
    # Concatenate data from all trials into np.arrays for training
    # Restructure data to have trial number x variable feature x time

    trial_arrays = []  # List to hold arrays for each trial
    trial_numbers = []  # List to hold trial numbers for each row in the trial_arrays

    for i, k in enumerate(windowed_dlc):
        # Get the column names
        columns = windowed_dlc[k].columns

        # Find the distance to goal columns
        distance_cols = [c for c in columns if c.startswith('dist')]

        # Find the relative direction columns (but not relative_direction_to columns)
        relative_direction_cols = [c for c in columns if c.startswith('relative_direction_') \
                                   and not c.startswith('relative_direction_screen')]

        column_names = ['x', 'y']
        column_names.extend(distance_cols)
        column_names.extend(['hd'])
        column_names.extend(relative_direction_cols)

        total_num_cols = 10  # x, y, distance_to_goal x2, hd x2, relative_direction x 4

        # Get the number of rows in the dataframe
        num_rows = len(windowed_dlc[k])

        # Create an empty np.array of the correct size
        temp_array = np.zeros((num_rows, total_num_cols))

        count = 0
        for c in column_names:
            # If c is not hd or relative_direction, just add the column to the array
            if c not in ['hd'] and not c.startswith('relative_direction'):
                temp_array[:, count] = windowed_dlc[k][c].values
                count += 1

            elif include_raw_hd:
                # Angular data needs to be converted to sin and cos
                temp_array[:, count] = np.sin(windowed_dlc[k][c].values)
                temp_array[:, count + 1] = np.cos(windowed_dlc[k][c].values)
                temp_array[:, count + 2] = windowed_dlc[k][c].values
                count += 3
            else:
                temp_array[:, count] = np.sin(windowed_dlc[k][c].values)
                temp_array[:, count + 1] = np.cos(windowed_dlc[k][c].values)
                count += 2

        # Split each trial into windows
        for start in range(0, num_rows, length_size):
            end = start + length_size
            trial_arrays.append(temp_array[start:min(end, num_rows), :])
            trial_numbers.append(np.full(min(end - start, num_rows), i))

    # Stack all trial arrays into a 3D array
    dlc_array = np.stack(trial_arrays, axis=0)
    trial_numbers = np.concatenate(trial_numbers)

    # All columns need to be scaled to the range 0-1
    if scale_data:
        for i in range(dlc_array.shape[2]):
            dlc_array[:, :, i] = (dlc_array[:, :, i] - np.min(dlc_array[:, :, i])) / \
                                  (np.max(dlc_array[:, :, i]) - np.min(dlc_array[:, :, i]))
    elif z_score_data:
        epsilon = 1e-10
        for i in range(dlc_array.shape[2]):
            dlc_array[:, :, i] = (dlc_array[:, :, i] - np.mean(dlc_array[:, :, i])) / \
                                  (np.std(dlc_array[:, :, i]) + epsilon)

    dlc_array = np.round(dlc_array, 3)

    return dlc_array, trial_numbers, column_names


def cat_dlc_rolling_window_shape(windowed_dlc, include_raw_hd=True, scale_data=False, z_score_data=True, length_size=100):
    # get list of keys
    key_list = list(windowed_dlc.keys())
    n_keys = len(key_list)

    trial_arrays = []  # list to hold arrays for each trial
    trial_numbers = []  # list to hold trial numbers for each data point

    # Flatten all trials into a single long array for each key
    flat_dlc = {k: np.concatenate([windowed_dlc[k][i].values for i in windowed_dlc[k]]) for k in key_list}
    flat_trial_numbers = {k: np.concatenate([np.full(len(windowed_dlc[k][i].values), i) for i in windowed_dlc[k]]) for k in key_list}

    # Determine the total length of the flattened data
    total_length = len(next(iter(flat_dlc.values())))

    # Create rolling windows for each key
    for start in range(0, total_length, length_size):
        end = start + length_size
        temp_array = np.zeros((n_keys, length_size))
        temp_trial_array = np.zeros((n_keys, length_size))
        for j, k in enumerate(


        ):
            # Create a temporary array filled with zeros
            temp_dlc = np.zeros(length_size)
            # Fill the temporary array with the data if it is not empty
            if len(flat_dlc[k][start:min(end, len(flat_dlc[k]))]) > 0:
                temp_dlc[:min(end, len(flat_dlc[k])) - start] = flat_dlc[k][start:min(end, len(flat_dlc[k]))]
            # Assign the temporary array to the temp_array
            temp_array[j, :] = temp_dlc

            temp_trial = np.zeros(length_size)
            # Ensure that flat_trial_numbers[k] only contains numeric values
            trial_numbers = [num for num in flat_trial_numbers[k] if isinstance(num, (int, float))]
            temp_trial[:min(end, len(trial_numbers)) - start] = trial_numbers[start:min(end, len(trial_numbers))]
            temp_trial_array[j, :] = temp_trial

        trial_arrays.append(temp_array)
        trial_numbers.append(temp_trial_array)

    # Stack all trial arrays into a 3D array
    dlc_array = np.stack(trial_arrays, axis=0)
    dlc_array = np.round(dlc_array, 3)

    # Transpose the array to get the dimensions trial x time x variable
    dlc_array = np.transpose(dlc_array, (0, 2, 1))

    trial_array = np.stack(trial_numbers, axis=0)

    return dlc_array, key_list, trial_array
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

def cat_spike_trains_3d(spike_trains):
    '''using padding to make the spike trains the same length'''
    # get list of units
    unit_list = list(spike_trains.keys())
    n_units = len(unit_list)

    trial_arrays = []  # list to hold arrays for each trial

    max_cols = max(len(spike_trains[u][k]) for u in unit_list for k in spike_trains[unit_list[0]].keys())

    for i, k in enumerate(spike_trains[unit_list[0]].keys()):
        # create an empty np.array of the correct size
        temp_array = np.zeros((n_units, max_cols))

        for j, u in enumerate(unit_list):
            # add the spike trains to the array
            temp_array[j, :len(spike_trains[u][k])] = spike_trains[u][k]

        # add the array for this trial to the list
        trial_arrays.append(temp_array)

    # concatenate along a new trial axis to get a 3D array
    spike_array = np.stack(trial_arrays, axis=0)

    spike_array = np.round(spike_array, 3)

    return spike_array, unit_list

def cat_spike_trains_3d(spike_trains):
    '''using padding to make the spike trains the same length'''
    # get list of units
    unit_list = list(spike_trains.keys())
    n_units = len(unit_list)

    trial_arrays = []  # list to hold arrays for each trial

    max_cols = max(len(spike_trains[u][k]) for u in unit_list for k in spike_trains[unit_list[0]].keys())

    for i, k in enumerate(spike_trains[unit_list[0]].keys()):
        # create an empty np.array of the correct size
        temp_array = np.zeros((n_units, max_cols))

        for j, u in enumerate(unit_list):
            # add the spike trains to the array
            temp_array[j, :len(spike_trains[u][k])] = spike_trains[u][k]

        # add the array for this trial to the list
        trial_arrays.append(temp_array)

    # concatenate along a new trial axis to get a 3D array
    spike_array = np.stack(trial_arrays, axis=0)

    spike_array = np.round(spike_array, 3)

    return spike_array, unit_list

def cat_dlc_3d(dlc_data):
    '''using padding to make the spike trains the same length'''
    # get list of units
    var_list = list(dlc_data.keys())
    n_units = len(var_list)

    trial_arrays = []  # list to hold arrays for each trial

    # max_cols = max(len(dlc_data[u][k]) for u in var_list for k in dlc_data[var_list[0]].keys())
    max_cols = max(
        len(dlc_data[u][k]) for u in var_list for d in dlc_data[var_list[0]] if len(dlc_data[var_list[0]]) > 0 for k in
        range(d.shape[0]))


    for i, k in enumerate(dlc_data[var_list[0]].keys()):
        # create an empty np.array of the correct size
        temp_array = np.zeros((n_units, max_cols))

        for j, u in enumerate(var_list):
            # add the spike trains to the array
            temp_array[j, :len(dlc_data[u][k])] = dlc_data[u][k]

        # add the array for this trial to the list
        trial_arrays.append(temp_array)

    # concatenate along a new trial axis to get a 3D array
    behav_array = np.stack(trial_arrays, axis=0)

    behav_array = np.round(behav_array, 3)

    return behav_array, var_list
def cat_spike_trains_3d_rolling_window(spike_trains, length_size = 100):
    # get list of units
    unit_list = list(spike_trains.keys())
    n_units = len(unit_list)

    trial_arrays = []  # list to hold arrays for each trial
    trial_numbers = []  # list to hold trial numbers for each spike bin

    # Flatten all trials into a single long array for each unit
    flat_spike_trains = {u: np.concatenate([spike_trains[u][k] for k in spike_trains[u]]) for u in unit_list}
    flat_trial_numbers = {u: np.concatenate([np.full(len(spike_trains[u][k]), i) for i, k in enumerate(spike_trains[u])]) for u in unit_list}

    # Determine the total length of the flattened spike trains
    total_length = len(next(iter(flat_spike_trains.values())))

    # Create rolling windows for each unit
    trial_number = 0  # Initialize trial number
    for start in range(0, total_length, length_size):
        end = start + length_size
        temp_array = np.zeros((n_units, length_size))
        temp_trial_array = np.zeros((n_units, length_size))
        for j, u in enumerate(unit_list):
            # Create a temporary array filled with zeros
            temp_spike_train = np.zeros(length_size)
            # Fill the temporary array with the spike train data
            temp_spike_train[:min(end, len(flat_spike_trains[u])) - start] = flat_spike_trains[u][start:min(end, len(flat_spike_trains[u]))]
            # Assign the temporary array to the temp_array
            temp_array[j, :] = temp_spike_train


            temp_trial_train = np.zeros(length_size)
            temp_trial_train[:min(end, len(flat_spike_trains[u])) - start] = flat_trial_numbers[u][start:min(end, len(flat_spike_trains[u]))]
            temp_trial_array[j, :] = temp_trial_train

        trial_arrays.append(temp_array)
        trial_numbers.append(temp_trial_array)


    # Stack all trial arrays into a 3D array
    spike_array = np.stack(trial_arrays, axis=0)

    spike_array = np.round(spike_array, 3)

    trial_array = np.stack(trial_numbers, axis=0)

    return spike_array, unit_list, trial_array

def cat_behav_data_3d_rolling_window(dlc_data, length_size = 100):
    # get list of units
    unit_list = list(dlc_data.keys())
    n_units = len(unit_list)

    trial_arrays = []  # list to hold arrays for each trial
    trial_numbers = []  # list to hold trial numbers for each spike bin

    # Flatten all trials into a single long array for each unit
    flat_spike_trains = {u: np.concatenate(dlc_data[u]) for u in unit_list}

    flat_trial_numbers = {u: np.concatenate([np.full(len(df), i) for i, df in enumerate(dlc_data[u])]) for u in
                          unit_list}

    # Determine the total length of the flattened spike trains
    total_length = len(next(iter(flat_spike_trains.values())))

    # Create rolling windows for each unit
    trial_number = 0  # Initialize trial number
    for start in range(0, total_length, length_size):
        end = start + length_size
        temp_array = np.zeros((n_units, length_size))
        temp_trial_array = np.zeros((n_units, length_size))
        for j, u in enumerate(unit_list):
            # Create a temporary array filled with zeros
            temp_spike_train = np.zeros(length_size)
            # Fill the temporary array with the spike train data
            temp_spike_train[:min(end, len(flat_spike_trains[u])) - start] = flat_spike_trains[u][start:min(end, len(flat_spike_trains[u]))]
            # Assign the temporary array to the temp_array
            temp_array[j, :] = temp_spike_train


            temp_trial_train = np.zeros(length_size)
            temp_trial_train[:min(end, len(flat_spike_trains[u])) - start] = flat_trial_numbers[u][start:min(end, len(flat_spike_trains[u]))]
            temp_trial_array[j, :] = temp_trial_train

        trial_arrays.append(temp_array)
        trial_numbers.append(temp_trial_array)


    # Stack all trial arrays into a 3D array
    spike_array = np.stack(trial_arrays, axis=0)

    spike_array = np.round(spike_array, 3)

    trial_array = np.stack(trial_numbers, axis=0)

    return spike_array, unit_list, trial_array
def reshape_model_inputs_and_labels(model_inputs, labels):
    labels = labels[:, 0:3]
    #reshape the the model input to be time interval x time bin x neuron
    labels = labels[:, 0:3]
    time_interval = 10  # Define your time interval
    time_bin = 1  # Define your time bin
    time_bins_per_interval = time_interval // time_bin
    total_time = model_inputs.shape[0]
    total_intervals = total_time // time_interval

    # Reshape model_inputs to be (time interval, time bin, 112)
    model_inputs_new = model_inputs[:total_intervals * time_interval].reshape(total_intervals, time_bins_per_interval, -1)
    return


if __name__ == "__main__":
    # animal = 'Rat65'
    # session = '10-11-2023'
    # data_dir = get_data_dir(animal, session)

    big_dir = 'C:/neural_data/'
    # 3, 8, 9, 10
    for rat in [7, 3, 8, 9, 10]:
        #get the list of folders directory that have dates
        print(f'now starting rat:{rat}')
        dates = os.listdir(os.path.join(big_dir, f'rat_{rat}'))
        #check if the folder name is a date by checking if it contains a hyphen
        date = [d for d in dates if '-' in d][0]
        data_dir = os.path.join(big_dir, f'rat_{rat}', date)

    # data_dir = '/media/jake/DataStorage_6TB/DATA/neural_network/og_honeycomb/rat7/6-12-2019'

    # load spike data
    # spike_dir = os.path.join(data_dir, 'spike_sorting')
    # units = load_pickle('units_w_behav_correlates', spike_dir)

        spike_dir = os.path.join(data_dir, 'physiology_data')
        units = load_pickle('restricted_units', spike_dir)
        use_overlap = False

        # load positional data
        # dlc_dir = os.path.join(data_dir, 'deeplabcut')
        # dlc_data = load_pickle('dlc_final', dlc_dir)

        dlc_dir = os.path.join(data_dir, 'positional_data')
        dlc_data = load_pickle('dlc_data', dlc_dir)
        dlc_data = dlc_data['hComb']

        # create positional and spike trains with overlapping windows
        # and save as a pickle file
        if use_overlap:
            windowed_dlc, window_edges, window_size = \
                create_positional_trains(dlc_data, window_size=25)
        else:
            windowed_dlc, window_edges, window_size = \
                create_positional_trains_no_overlap(dlc_data, window_size=25)
        windowed_data = {'windowed_dlc': windowed_dlc, 'window_edges': window_edges}
        save_pickle(windowed_data, 'windowed_data', dlc_dir)

        windowed_data = load_pickle('windowed_data', dlc_dir)
        windowed_dlc = windowed_data['windowed_dlc']
        window_edges = windowed_data['window_edges']

        # create spike trains
        if use_overlap:
            spike_trains = create_spike_trains(units, window_edges, window_size=window_size)
        else:
            spike_trains = create_spike_trains_no_overlap(units, window_edges, window_size=window_size)
            # spike_trains_binned = create_spike_trains_trial_binning(units, window_edges, window_size=window_size)
            # spike_trains_3d = create_spike_trains_merge_into_trial(units, window_size=window_size)
        save_pickle(spike_trains, f'spike_trains_overlap_{use_overlap}', spike_dir)
        spike_trains = load_pickle(f'spike_trains_overlap_{use_overlap}', spike_dir)

        # concatenate data from all trials into np.arrays for training
        norm_data = False
        zscore_option = False
        #rearrange windowed_dlc so it's index by variable number
        rearranged_dlc = {}
        for key, df in windowed_dlc.items():
            variable_name = f"trial_{key.split('_')[1]}"
            #get the columns of the dataframe
            columns = df.columns
            for col in columns:
                if col not in rearranged_dlc:
                    rearranged_dlc[col] = []
                rearranged_dlc[col].append(df[col].values)
        #convert the lists to np.arrays

        #save the rearranged dlc
        labels_rolling_window, var_list, trial_list = cat_behav_data_3d_rolling_window(rearranged_dlc, length_size=100)
        #take an example trial_list from the second dimension

        trial_list_example = trial_list[:,0,:]
        # behav_array, var_list_padded = cat_dlc_3d(rearranged_dlc)

        labels, column_names = cat_dlc(windowed_dlc, scale_data=norm_data, z_score_data=zscore_option)
        # convert labels to float32
        labels = labels.astype(np.float32)
        np.save(f'{dlc_dir}/labels_1103_with_dist2goal_scale_data_{norm_data}_zscore_data_{zscore_option}_overlap_{use_overlap}.npy', labels)

        # concatenate spike trains into np.arrays for training
        model_inputs_3d, unit_list = cat_spike_trains_3d(spike_trains)

        model_inputs_roving, unit_list_roving, trial_number_tracker = cat_spike_trains_3d_rolling_window(spike_trains, length_size=100)
        trial_number_tracker_example = trial_number_tracker[:,0,:]

        #check if the trial number tracker is the same as the trial_list
        assert np.all(trial_number_tracker_example == trial_list_example)

        model_inputs, unit_list = cat_spike_trains(spike_trains)

        # convert model_inputs to float32
        model_inputs = model_inputs.astype(np.float32)
        np.save(f'{spike_dir}/inputs_overlap_{use_overlap}.npy', model_inputs)
        save_pickle(unit_list, f'unit_list_overlap_{use_overlap}', spike_dir)

        reshape_model_inputs_and_labels(model_inputs, labels)
    pass