from scipy.io import loadmat
import numpy as np
import os
import pandas as pd
from load_and_save_data import save_pickle
from pathlib import Path

def round_matlab_data(data, n_decimals=0):
    """
    Round the data from matlab to the nearest integer.

    Parameters
    ----------
    data : array
        The data from matlab.

    Returns
    -------
    data : array
        The data rounded to the nearest integer.
    """
    data = [d.astype(float) for d in data]
    data = [np.round(d, n_decimals) for d in data]
    return data


def create_ps_dataframes(positional_data_dir):
    positional_path = os.path.join(positional_data_dir, 'positionalDataByTrialType.mat')
    positional_data = loadmat(positional_path)

    # get the goal coordinates
    goal_position = positional_data['goalPosition'][0][0][0]
    goal_id = positional_data['goalID'][
        0]  # note this gives 2 values, but for the single goal day, they should be identical
    save_pickle(goal_position, 'goal_position', positional_data_dir)

    # create the positional dataframes
    pos_structure = positional_data['pos']

    # access the hComb and openF partitions of the data in a loop
    pos_dataframes = {}
    for i in range(2):
        if i == 0:
            pos_data = pos_structure[0][0][0][0]  # honeycomb task
            task = 'hComb'
        else:
            pos_data = pos_structure[0][0][1][0]  # open field
            task = 'openF'

        # extract samples
        samples = pos_data['sample']
        samples = round_matlab_data(samples)
        samples = [np.int32(s) for s in samples]

        # extract ts
        ts = pos_data['ts']
        ts = round_matlab_data(ts, 1)

        # calculate the sample rate
        sample_rate = int((np.round(samples[0][0] / ts[0][0]) * 1000)[0])

        # extract the x and y coordinates
        # xy = pos_data['dlc_XYsmooth']
        xy = pos_data['dlc_XYsmooth']

        xy = round_matlab_data(xy, 2)

        # extract the head angle
        hd = pos_data['dlc_angle']
        hd = round_matlab_data(hd, 2)
        dist2goal = pos_data['dist2goal']



        # extract the video time
        video_time = pos_data['videoTime']
        video_time = round_matlab_data(video_time, 3)

        n_trials = len(samples)

        # create a dictionary to store the data
        pos_dataframes[task] = {}

        for t in range(n_trials):
            # make columns 1d
            samples[t] = samples[t].flatten()
            video_time[t] = video_time[t].flatten()
            x = xy[t][:, 0].flatten()
            y = xy[t][:, 1].flatten()
            hd[t] = hd[t].flatten()
            dist2goal[t] = dist2goal[t].flatten()

            # the hd data needs to be rotated by 90deg and converted to radians
            hdtemp = (hd[t] - 90)
            hdtemp = np.deg2rad(hdtemp)
            # convert from 0 to 2pi to -pi to pi
            hdtemp = np.arctan2(np.sin(hdtemp), np.cos(hdtemp))
            # round to 3 decimal places
            hdtemp = np.round(hdtemp, 3)

            # create a pandas dataframe to store the data
            trial = f'trial_{t + 1}'
            pos_dataframes[task][trial] = pd.DataFrame({'video_samples': samples[t],
                                                        'video_time': video_time[t], 'x': x, 'y': y,
                                                        'hd': hdtemp, 'dist2goal': dist2goal[t]})

    return pos_dataframes, goal_position, goal_id


def create_spike_arrays(spike_data_dir):
    spike_path = os.path.join(spike_data_dir, 'units.mat')
    spike_data = loadmat(spike_path)

    sample_rate = spike_data['sample_rate'][0][0]
    save_pickle(sample_rate, 'sample_rate', spike_data_dir)

    spike_data = spike_data['units']
    n_units = spike_data.shape[0]

    # create a dictionary to store the spike data
    spike_arrays = {}
    spike_arrays['sample_rate'] = sample_rate
    spike_arrays['pyramid'] = {}
    spike_arrays['interneuron'] = {}
    spike_arrays['unclassified'] = {}

    for i in range(n_units):
        unit_name = spike_data[i][0][0][0]
        unit_type = spike_data[i][0][3][0][0][0]
        if unit_type == 'u':
            unit_type = 'unclassified'
        elif unit_type == 'p':
            unit_type = 'pyramid'
        unit_cluster = spike_data[i][0][1][
            0]  # not really necessary, unless we plan to go back and look at the clusters.
        unit_samples = spike_data[i][0][2][:, 0]
        print(f'n_spikes = {len(unit_samples)}')

        spike_arrays[unit_type][unit_name] = unit_samples

    return spike_arrays, sample_rate

def create_lfp_arrays(lfp_data_dir, fs = None):
    theta_data = loadmat(lfp_data_dir / 'thetaAndRipplePower.mat')
    theta_power = theta_data['thetaPower']
    theta_signal_hcomb = theta_power['hComb'][0][0]['raw']
    power_sample_index = theta_data['powerSampleInd']['hComb'][0][0]
    #interporlate up from a sample rate of 1000 hz to the fs sample rate
    theta_signal_hcomb = np.interp(np.arange(0, len(theta_signal_hcomb), 1000/fs), np.arange(0, len(theta_signal_hcomb)), theta_signal_hcomb)
    return

if __name__ == '__main__':
    # data_dir = 'D:/analysis/og_honeycomb'
    data_dir = 'C:/neural_data/'

    rat = 7
    # date = '6-12-2019'

    # load the positional data
    for rat in [3, 8, 9, 10]:
        #get the list of folders directory that have dates
        dates = os.listdir(os.path.join(data_dir, f'rat_{rat}'))
        #check if the folder name is a date by checking if it contains a hyphen
        date = [d for d in dates if '-' in d][0]


        positional_data_dir = os.path.join(data_dir, f'rat_{rat}', date, 'positional_data')
        pos_dataframes, goal_position, goal_id = create_ps_dataframes(positional_data_dir)
        save_pickle(goal_position, 'goal_position', positional_data_dir)
        # save the data in positional_data_dir
        save_pickle(pos_dataframes, 'dlc_data', positional_data_dir)

        # load the spike data
        spike_data_dir = os.path.join(data_dir, f'rat_{rat}', date, 'physiology_data')
        spike_arrays, sample_rate = create_spike_arrays(spike_data_dir)
        lfp_arrays = create_lfp_arrays(Path(spike_data_dir), fs = sample_rate)
        save_pickle(spike_arrays, 'unit_spike_times', spike_data_dir)

    pass