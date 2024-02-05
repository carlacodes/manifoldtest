import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import mat73
from pathlib import Path
import scipy
import numpy as np


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

    time = hcomb_data['videoTime']
    ts = hcomb_data['ts']
    sample = hcomb_data['sample']

    dlc_angle = hcomb_data['dlc_angle']

    #create a new array that interpolates based on the video time
    #and the dlc angle to a 30,000 Hz sample rate
    #this will be used to compare to the spike data
    #to see if the spike data is aligned with the positional data
    len(units)

    df_all = pd.DataFrame()
    for j in range(0, len(units)):
        #extract the unit from the units array
        unit = units[j]
        #extract the spike times from the unit
        spike_times = unit['spikeSamples']
        #convert to float
        spike_times = spike_times[0].astype(float)

        spike_times_seconds = spike_times/fs
        head_angle_times = np.array([])
        dlc_angle_list = np.array([])
        head_angle_times_ms = np.array([])
        trial_number_array = np.array([])
        for i2 in range(0, len(dlc_angle)):
            trial_dlc = dlc_angle[i2]
            trial_ts = ts[i2]
            trial_sample = sample[i2]
            time_in_seconds = trial_sample/fs

            #ts -- this is the time in ms

            trial_time = time[i2]
            trial_number_array = np.append(trial_number_array, i2)
            head_angle_times = np.append(head_angle_times, time_in_seconds)
            head_angle_times_ms = np.append(head_angle_times_ms, trial_time)
            dlc_angle_list = np.append(dlc_angle_list, trial_dlc)

            if np.max(trial_time) > np.max(spike_times_seconds):
                print('Trial time is greater than spike time, aborting...')
                break

        #interpolate the dlc angle to the spike times
        #this will allow us to compare the spike times to the dlc angle
        #and see if the spike times are aligned with the dlc angle
        dlc_angle_list = np.array(dlc_angle_list)
        dlc_new = np.interp(spike_times_seconds, head_angle_times_ms, dlc_angle_list)
        #construct a dataframe with the spike times and the dlc angle
        unit_id = unit['name'][0].astype(str)
        flattened_spike_times_seconds = np.concatenate(spike_times_seconds).ravel()
        flattened_spike_times = np.concatenate(spike_times).ravel()
        flattened_dlc_new = np.concatenate(dlc_new).ravel()
        #make unit_id the same length as the spike times
        unit_id = np.full(len(flattened_spike_times), unit_id)
        phy_cluster = unit['phyCluster'][0].astype(str)
        phy_cluster = np.full(len(flattened_spike_times), phy_cluster)
        neuron_type = unit['neuronType'][0][0]
        neuron_type = np.full(len(flattened_spike_times), neuron_type)


        df = pd.DataFrame({'spike_times_seconds': flattened_spike_times_seconds, 'spike_times_samples': spike_times, 'dlc_angle': flattened_dlc_new, 'unit_id': unit_id, 'phy_cluster': phy_cluster, 'neuron_type': neuron_type, 'time (ms)': head_angle_times_ms, 'time (seconds)': head_angle_times, 'trial_number': trial_number_array})
        #append to a larger dataframe
        if j == 0:
            df_all = df
        else:
            df_all = pd.concat([df_all, df])



        #extract the trial type from the unit






    print(spike_data.keys())



def main():
    load_data_from_paths(Path('C:/neural_data/'))



if __name__ == '__main__':
    main()