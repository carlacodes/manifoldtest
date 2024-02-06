import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import mat73
from pathlib import Path
import scipy
import numpy as np
from scipy.signal import hilbert


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


            trial_number_full = np.full(len(trial_ts), i2)
            trial_number_array = np.append(trial_number_array, trial_number_full)

            head_angle_times = np.append(head_angle_times, time_in_seconds)
            head_angle_times_ms = np.append(head_angle_times_ms, trial_ts)
            dlc_angle_list = np.append(dlc_angle_list, trial_dlc)

            if np.max(time_in_seconds) > np.max(spike_times_seconds):
                print('Trial time is greater than spike time, aborting...')
                break

        #interpolate the dlc angle to the spike times
        #this will allow us to compare the spike times to the dlc angle
        #and see if the spike times are aligned with the dlc angle
        dlc_angle_list = np.array(dlc_angle_list)
        dlc_new = np.interp(spike_times_seconds, head_angle_times_ms, dlc_angle_list)
        trial_new = np.interp(spike_times_seconds, head_angle_times_ms, trial_number_array)

        #construct a dataframe with the spike times and the dlc angle
        unit_id = unit['name'][0].astype(str)
        flattened_spike_times_seconds = np.concatenate(spike_times_seconds).ravel()
        flattened_spike_times = np.concatenate(spike_times).ravel()
        flattened_dlc_new = np.concatenate(dlc_new).ravel()
        flattened_trial_new = np.concatenate(trial_new).ravel()

        #make unit_id the same length as the spike times
        unit_id = np.full(len(flattened_spike_times), unit_id)
        phy_cluster = unit['phyCluster'][0].astype(str)
        phy_cluster = np.full(len(flattened_spike_times), phy_cluster)
        try:
            neuron_type = unit['neuronType'][0][0][0][0].astype(str)
        except:
            neuron_type = 'unclassified'
        neuron_type = np.full(len(flattened_spike_times), neuron_type)


        df = pd.DataFrame({'spike_times_seconds': flattened_spike_times_seconds, 'spike_times_samples': flattened_spike_times, 'dlc_angle': flattened_dlc_new, 'unit_id': unit_id, 'phy_cluster': phy_cluster, 'neuron_type': neuron_type, 'trial_number': flattened_trial_new})
        #append to a larger dataframe
        if j == 0:
            df_all = df
        else:
            df_all = pd.concat([df_all, df])

    return


def load_theta_data(path, fs=1000, plot_figures = True):
    #   load the theta data from the local path
    theta_data = scipy.io.loadmat(path / 'thetaAndRipplePower.mat')
    theta_power = theta_data['thetaPower']
    theta_signal_hcomb = theta_power['hComb'][0][0]['raw']

    ripple_power = theta_data['ripplePower']
    #caluculate theta phase and amplitude
    phase_array = np.array([])
    trial_array = np.array([])
    for i in range(0, len(theta_signal_hcomb)):
        signal = theta_signal_hcomb[i][0]
        #flatten the data
        signal = signal.ravel()
        hilbert_transform = hilbert(signal)

        # Calculate the instantaneous phase
        instantaneous_phase = np.angle(hilbert_transform)

        # Calculate the instantaneous frequency
        t = np.arange(0, len(signal)) / fs
        instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi * np.diff(t))
        phase_array = np.append(phase_array, instantaneous_phase)
        trial_array = np.append(trial_array, np.full(len(instantaneous_phase), i))



        # Plot the results
        if plot_figures == True:
            plt.figure(figsize=(10, 10))

            plt.subplot(4, 1, 1)
            plt.plot(t, signal, label='Original Signal')
            plt.title('Original Signal, trial ' + str(i))

            plt.subplot(4, 1, 2)
            plt.plot(t, hilbert_transform.real, label='Hilbert Transform (Real)')
            plt.title('Hilbert Transform Real Part')

            plt.subplot(4, 1, 3)
            plt.plot(t, hilbert_transform.imag, label='Hilbert Transform (Imaginary)')
            plt.title('Hilbert Transform Imaginary Part')

            plt.subplot(4, 1, 4)
            plt.plot(t, instantaneous_phase, label='Instantaneous Phase')
            plt.title('Instantaneous Phase')

            plt.tight_layout()
            plt.show()
        #append the instantaneous phase



    return phase_array, trial_array








        #extract the trial type from the unit









def main():
    phase_array, trial_array = load_theta_data(Path('C:/neural_data/'))
    load_data_from_paths(Path('C:/neural_data/'))



if __name__ == '__main__':
    main()