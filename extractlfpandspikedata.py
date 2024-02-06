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
        dlc_angle_list = dlc_angle_list.ravel()
        dlc_new = np.interp(spike_times_seconds*1000, head_angle_times_ms, dlc_angle_list)
        trial_new = np.interp(spike_times_seconds*1000, head_angle_times_ms, trial_number_array)
        #round the trial number
        trial_new = np.round(trial_new)

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

    return df_all


def resample_by_interpolation(signal, input_fs, output_fs):

    scale = output_fs / input_fs
    # calculate new length of sample
    n = round(len(signal) * scale)

    # use linear interpolation
    # endpoint keyword means than linspace doesn't go all the way to 1.0
    # If it did, there are some off-by-one errors
    # e.g. scale=2.0, [1,2,3] should go to [1,1.5,2,2.5,3,3]
    # but with endpoint=True, we get [1,1.4,1.8,2.2,2.6,3]
    # Both are OK, but since resampling will often involve
    # exact ratios (i.e. for 44100 to 22050 or vice versa)
    # using endpoint=False gets less noise in the resampled sound
    resampled_signal = np.interp(
        np.linspace(0.0, 1.0, n, endpoint=False),  # where to interpret
        np.linspace(0.0, 1.0, len(signal), endpoint=False),  # known positions
        signal,  # known data points
    )
    return resampled_signal

def load_theta_data(path, fs=1000, spike_data = [], plot_figures = False):
    #   load the theta data from the local path
    theta_data = scipy.io.loadmat(path / 'thetaAndRipplePower.mat')
    theta_power = theta_data['thetaPower']
    theta_signal_hcomb = theta_power['hComb'][0][0]['raw']

    ripple_power = theta_data['ripplePower']
    #caluculate theta phase and amplitude
    phase_array = np.array([])
    trial_array = np.array([])
    theta_array = np.array([])
    for i in range(0, len(theta_signal_hcomb[0])):

        signal = theta_signal_hcomb[0][i]

        #flatten the data
        signal = signal.ravel()
        theta_array = np.append(theta_array, signal)
        hilbert_transform = hilbert(signal)

        # Calculate the instantaneous phase
        instantaneous_phase = np.angle(hilbert_transform)

        # Calculate the instantaneous frequency
        t = np.arange(0, len(signal)) / fs
        instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi * np.diff(t))
        phase_array = np.append(phase_array, instantaneous_phase)
        trial_array = np.append(trial_array, np.full(len(instantaneous_phase), i))

        #upsample the data to 30,000 Hz



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

    # trial_new = np.interp(spike_times_seconds, head_angle_times_ms, trial_number_array)
    t = np.arange(0, len(phase_array)) / fs
    phase_array_new = np.interp(spike_data['spike_times_seconds'], t, phase_array)

    trial_array_new = np.interp(spike_data['spike_times_seconds'], t, trial_array)
    #check if trial arrays are equivalent
    if np.array_equal(trial_array_new, spike_data['trial_number']):
        print('Trial arrays are equivalent')
    else:
        print('Trial arrays are not equivalent')

    return phase_array, trial_array, theta_array







        #extract the trial type from the unit
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')


def compare_spike_times_to_theta_phase(spike_data, phase_array,theta_array, trial_array, window_size = 100):
    #compare the spike times to the theta phase
    #for each spike time, find the corresponding theta phase
    #and trial number
    df_plv_all = pd.DataFrame()
    for i in spike_data['unit_id'].unique():
        #extract the spike times for the unit
        # unit_spike_times = spike_data[spike_data['unit_id'] == i]['spike_times_seconds']
        # unit_spike_times = unit_spike_times.to_numpy()
        unit_spike_data = spike_data[spike_data['unit_id'] == i]
        #extract the trial number for the unit
        plv_for_unit = np.array([])
        cross_corr_for_unit = np.array([])
        for j in unit_spike_data['trial_number'].unique():
            unit_spike_data_trial = unit_spike_data[unit_spike_data['trial_number'] == j]
            #calculate the phase locking value between the spike times, theta phase, and dlc angle
            #for the unit
            theta_in_trial = theta_array[trial_array == j]
            angle_in_trial = unit_spike_data_trial['dlc_angle']
            #downsample so that the length of the arrays are the same
            angle_in_trial = np.interp(np.linspace(0, len(angle_in_trial), len(theta_in_trial)), np.arange(0, len(angle_in_trial)), angle_in_trial)
            #calculate the gradient of the angle
            angle_in_trial_grad = np.gradient(angle_in_trial)


            # Detect non-stationary periods
            non_stationary_periods = angle_in_trial_grad > 0
            #filter for non stationary periods
            angle_in_trial = angle_in_trial[non_stationary_periods]
            theta_in_trial = theta_in_trial[non_stationary_periods]
            if angle_in_trial.isempty() or theta_in_trial.isempty():
                print('Angle or theta is empty, skipping...')
                continue


            theta_analytic = hilbert(theta_in_trial)
            head_analytic = hilbert(angle_in_trial)
            # Calculate the Phase Locking Value

            phase_difference = np.angle(theta_analytic/head_analytic)
            #calculate the cross correlation between the theta phase and the dlc angle
            cross_correlation = np.correlate(theta_in_trial, angle_in_trial, mode='full')

            # Calculate the Phase Locking Value
            plv = np.abs(np.mean(np.exp(1j * phase_difference)))
            print('Phase locking value: ' + str(plv))
            plv_for_unit = np.append(plv_for_unit, plv)
            #plot the phase difference
            if plv >=0.7:
                plt.figure()
                plt.plot(phase_difference)
                plt.title('Phase difference')
                plt.show()
                #plot the spike times and the theta phase
                # plt.figure()
                # plt.plot(unit_spike_data_trial['spike_times_seconds'], theta_in_trial, 'bo')
                # plt.title('Spike times and theta phase')
                # plt.show()

                # #plot the spike times and the dlc angle
                # plt.figure()
                # plt.plot(unit_spike_data_trial['spike_times_seconds'], angle_in_trial, 'ro')
                # plt.title('Spike times and dlc angle')
                # plt.show()

                #plot the theta, spike times, and dlc angle
                plt.figure()
                plt.plot(theta_in_trial, label = 'Theta phase')
                plt.plot(angle_in_trial, label = 'DLC angle')
                plt.legend()
                plt.title(f'theta phase for trial number {j} and dlc angle, \n  plv is {plv} and unit ID: {i}')

                plt.savefig(f'figures/theta_phase_dlc_angle_unit_{i}_{plv}.png')
                # plt.show()
        #plot the plv for the unit
        plt.figure()
        plt.plot(plv_for_unit)
        plt.title('PLV for unit ' + str(i))
        plt.show()
        mean_plv = np.mean(plv_for_unit)
        mean_plv = np.full(len(plv_for_unit), mean_plv)
        mean_cross_corr = np.mean(cross_correlation)
        mean_cross_corr = np.full(len(cross_correlation), mean_cross_corr)

        #add the plv to the dataframe
        df_plv = pd.DataFrame({'plv': plv_for_unit, 'unit_id': i, 'mean plv': mean_plv, 'cross correlation': cross_correlation, 'mean cross correlation': mean_cross_corr, 'trial_number': unit_spike_data['trial_number'].unique()})
        if i == 0:
            df_plv_all = df_plv
        else:
            df_plv_all = pd.concat([df_plv_all, df_plv])

        #extract the theta phase for the unit

        #for each spike time, find the corresponding theta phase
        # for j in unit_spike_times:
        #     #find the closest theta phase to the spike time
        #     closest_theta_phase = np.argmin(np.abs(unit_theta_phase - j))
        #     print('Closest theta phase to spike time: ' + str(closest_theta_phase))
        #     print('Spike time: ' + str(j))
        #     print('Theta phase: ' + str(unit_theta_phase[closest_theta_phase]))
        #     print('Trial number: ' + str(unit_trial_numbers[closest_theta_phase]))
        #
        #     #plot the spike time and the theta phase
        #     plt.figure()
        #     plt.plot(j, unit_theta_phase[closest_theta_phase], 'ro')
        #     plt.plot(unit_spike_times, unit_theta_phase, 'bo')
        #     plt.title('Spike time and theta phase')
        #     plt.show()
    return df_plv_all








def main():
    df_all = load_data_from_paths(Path('C:/neural_data/'))
    phase_array, trial_array, theta_array = load_theta_data(Path('C:/neural_data/'), spike_data = df_all)
    compare_spike_times_to_theta_phase(df_all, phase_array, theta_array, trial_array)





if __name__ == '__main__':
    main()