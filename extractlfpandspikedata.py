import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import mat73
import statsmodels.api as sm
from pathlib import Path
import scipy
import numpy as np
from scipy.signal import hilbert
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller  # Augmented Dickey-Fuller Test
from scipy.interpolate import interp1d
import pingouin as pg


def load_data_from_paths(path):
    '''Load the spike data and the positional data from the local path
    and return a dataframe with the spike times and the dlc angle
    for each unit. The dlc and trial numbers are interpolated to a 30,000 Hz
    :param path: the local path to the data
    :return: a dataframe with the spike times and the dlc angle for each unit'''

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

    fs = ((sample[0][0] / ts[0][0])*10000)[0]

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
        scipy.interpolate.interp1d(spike_times_seconds*1000, dlc_angle_list)

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



def load_theta_data(path, fs=1000, spike_data = [], plot_figures = False):
    ''' Load the theta data from the local path and calculate the instantaneous phase
    and frequency of the theta signal. Then, compare the spike times to the theta phase
    and amplitude to see if the spike times are aligned with the theta phase and amplitude
    :param path: the local path to the theta data
    :param fs: the sample rate of the theta data
    :param spike_data: the spike data
    :param plot_figures: whether to plot the figures
    :return: the theta phase, amplitude, and trial number for each spike time
    '''
    #   load the theta data from the local path
    theta_data = scipy.io.loadmat(path / 'thetaAndRipplePower.mat')
    theta_power = theta_data['thetaPower']
    theta_signal_hcomb = theta_power['hComb'][0][0]['raw']
    power_sample_index = theta_data['powerSampleInd']['hComb'][0][0]

    positional_data = scipy.io.loadmat(path / 'positionalDataByTrialType.mat')
    pos_cell = positional_data['pos']
    hcomb_data_pos = pos_cell[0][0][0][0]
    time = hcomb_data_pos['videoTime']
    ts = hcomb_data_pos['ts']
    dlc_angle = hcomb_data_pos['dlc_angle']
    sample = hcomb_data_pos['sample']

    #recreate ts to check -- ts is time in milliseconds
    test = sample/30000


    ripple_power = theta_data['ripplePower']
    #caluculate theta phase and amplitude

    #load the positional data
    positional_data = scipy.io.loadmat(path / 'positionalDataByTrialType.mat')
    for i in range(0, len(theta_signal_hcomb[0])):
        phase_array = np.array([])
        trial_array = np.array([])
        theta_array = np.array([])
        dlc_phase_array = np.array([])


        signal = theta_signal_hcomb[0][i]
        dlc_angle_trial = dlc_angle[i]
        ts_angle_trial = ts[i]

        power_sample_index_trial = power_sample_index[i,:]
        start_time = power_sample_index_trial[0] / 30000
        stop_time = power_sample_index_trial[1] / 30000

        # timestamp_array_theta = np.arange(start_time, stop_time, 1 / fs)
        #convert to milliseconds
        num_points = len(signal.ravel())
        timestamp_array_theta = np.linspace(start_time, stop_time, num_points)

        timestamp_array_theta = timestamp_array_theta * 1000
        #create time vector for the theta signal, the timepoimts are power_sample_index_trial
        signal = signal.ravel()
        theta_array = np.append(theta_array, signal)
        hilbert_transform = hilbert(signal)
        # Calculate the instantaneous phase
        ts_angle_trial = ts_angle_trial.ravel()
        dlc_angle_trial = dlc_angle_trial.ravel()

        interpolated_dlc_angle = np.interp(timestamp_array_theta, ts_angle_trial, dlc_angle_trial)
        if len(interpolated_dlc_angle) != len(signal):
            # Adjust the length if necessary
            interpolated_dlc_angle = np.resize(interpolated_dlc_angle, len(signal))
        instantaneous_phase = np.angle(hilbert_transform)

        hilbert_transform_angle = hilbert(interpolated_dlc_angle)
        instantaneous_phase_angle = np.angle(hilbert_transform_angle)


        # Calculate the instantaneous frequency
        t = np.arange(0, len(signal)) / fs
        instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi * np.diff(t))
        phase_array = np.append(phase_array, instantaneous_phase)
        dlc_phase_array = np.append(dlc_phase_array, instantaneous_phase_angle)
        trial_array = np.append(trial_array, np.full(len(instantaneous_phase), i))
        #
        #upsample the dlc_angle_trial to the same length as the theta signal

        #create a dataframe for this trial
        df_trial = pd.DataFrame({'theta_phase': instantaneous_phase, 'dlc_angle': interpolated_dlc_angle, 'dlc_angle_phase': dlc_phase_array, 'trial_number': np.full(len(instantaneous_phase), i), 'time_ms': timestamp_array_theta})
        if i == 0:
            df_theta_and_angle = df_trial
        else:
            df_theta_and_angle = pd.concat([df_theta_and_angle, df_trial])


        # Plot the results
        if plot_figures == True:
            plt.figure(figsize=(10, 10))
            # plt.plot(t, signal, label='Original Signal')
            # plt.plot(t, hilbert_transform.real, label='Hilbert Transform (Real)')
            # plt.plot(t, hilbert_transform.imag, label='Hilbert Transform (Imaginary)')
            # plt.plot(t, instantaneous_phase, label='Instantaneous Phase')


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
    #upsampl


    return phase_array, trial_array, theta_array, df_theta_and_angle



def compare_spike_times_to_theta_phase(spike_data, phase_array,theta_array, trial_array, export_to_csv = True):
    '''Compare the spike times to the theta phase and amplitude
    for each unit. Calculate the phase locking value between the spike times
    and the theta phase, and the cross correlation between the spike times
    IN PROGRESS AS OF 08/10/2024
    :param spike_data: the spike data
    :param phase_array: the theta phase
    :param theta_array: the theta amplitude
    :param trial_array: the trial number
    :param export_to_csv: whether to export the results to a csv
    :return: a dataframe with the phase locking value for each unit
    '''
    #compare the spike times to the theta phase
    #for each spike time, find the corresponding theta phase
    #and trial number
    df_plv_all = pd.DataFrame()
    granger_dict_all_acrossunits = np.array([])

    for count, i in enumerate(spike_data['unit_id'].unique()):
        #extract the spike times for the unit
        # unit_spike_times = spike_data[spike_data['unit_id'] == i]['spike_times_seconds']
        # unit_spike_times = unit_spike_times.to_numpy()
        unit_spike_data = spike_data[spike_data['unit_id'] == i]
        #extract the trial number for the unit
        plv_for_unit = np.array([])
        cross_corr_for_unit = np.array([])
        granger_dict_all = np.array([])
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
            non_stationary_periods = np.abs(angle_in_trial_grad) >= 0.1
            #get the indices of the non-stationary periods
            non_stationary_periods = np.where(non_stationary_periods == True)
            #only include the non-stationary periods
            angle_in_trial_grad = angle_in_trial_grad[non_stationary_periods]
            angle_in_trial = angle_in_trial[non_stationary_periods]
            theta_in_trial = theta_in_trial[non_stationary_periods]

            if len(angle_in_trial) == 0 or len(theta_in_trial) == 0:
                print('Angle or theta is empty, skipping...')
                continue


            theta_analytic = hilbert(theta_in_trial)
            head_analytic = hilbert(angle_in_trial)
            # Calculate the Phase Locking Value

            phase_difference = np.angle(theta_analytic/head_analytic)
            adf_result_angle = adfuller(angle_in_trial)
            adf_result_theta = adfuller(theta_in_trial)

            # Check the p-values to determine stationarity
            is_stationary_angle = adf_result_angle[1] <= 0.05
            is_stationary_theta = adf_result_theta[1] <= 0.05

            if not is_stationary_angle or not is_stationary_theta:
                print(f"Unit {i}, Trial {j}: Not stationary. Applying differencing...")
                # Apply differencing to make the time series stationary
                angle_in_trial = np.diff(angle_in_trial)
                theta_in_trial = np.diff(theta_in_trial)
                # Check the p-values again
                adf_result_angle = adfuller(angle_in_trial)
                adf_result_theta = adfuller(theta_in_trial)
                is_stationary_angle = adf_result_angle[1] <= 0.05
                is_stationary_theta = adf_result_theta[1] <= 0.05
                if not is_stationary_angle or not is_stationary_theta:
                    print(f"Unit {i}, Trial {j}: Still not stationary. Skipping...")
                    continue
            #calculate the ideal lag for granger causality

            #calculate the cross correlation between the theta phase and the dlc angle
            cross_correlation = np.correlate(theta_in_trial, angle_in_trial, mode='full')

            granger_test = grangercausalitytests(np.column_stack((angle_in_trial, theta_in_trial)), maxlag=150)
            for key in granger_test.keys():
                print('Granger test results: ' + str(granger_test[key][0]['ssr_ftest']))
                #add to a dataframe
                granger_test_for_indiv_lag = granger_test[key][0]['ssr_ftest']
                granger_test_lag_dataframe = pd.DataFrame({'F-statistic': granger_test_for_indiv_lag[0], 'p-value': granger_test_for_indiv_lag[1], 'df_denom': granger_test_for_indiv_lag[2], 'df_num': granger_test_for_indiv_lag[3]}, index=[0])
                granger_test_lag_dataframe['unit_id'] = i
                granger_test_lag_dataframe['trial_number'] = j
                granger_test_lag_dataframe['lag'] = key
                if key == 1:
                    granger_dataframe_all_lag = granger_test_lag_dataframe
                else:
                    granger_dataframe_all_lag = pd.concat([granger_dataframe_all_lag, granger_test_lag_dataframe])
            #append to a larger dictionary
            granger_dict = {'unit_id': i, 'trial_number': j, 'granger_test': granger_test}
            #extract the ssr_ftest value
            if j == 0:
                granger_dataframe_all_trial = granger_dataframe_all_lag
                granger_dict_all = granger_dict
            else:
                granger_dataframe_all_trial = pd.concat([granger_dataframe_all_trial, granger_dataframe_all_lag])
                granger_dict_all = np.append(granger_dict_all, granger_dict)












            # Calculate the Phase Locking Value
            plv = np.abs(np.mean(np.exp(1j * phase_difference)))
            print('Phase locking value: ' + str(plv))
            print(f'cross correlation at trial {j} is {cross_correlation}')
            plv_for_unit = np.append(plv_for_unit, plv)
            #plot the phase difference
            if plv >=0.7:
                # plt.figure()
                # plt.plot(phase_difference)
                # plt.title('Phase difference')
                # plt.show()
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

        df_plv = pd.DataFrame({'plv': plv_for_unit, 'unit_id': i, 'mean plv': mean_plv})
        if count == 0:
            df_plv_all = df_plv
            granger_dict_all_acrossunits = granger_dict_all
            granger_dataframe_all_unit = granger_dataframe_all_trial
        else:
            df_plv_all = pd.concat([df_plv_all, df_plv])
            granger_dict_all_acrossunits = np.append(granger_dict_all_acrossunits, granger_dict_all)
            granger_dataframe_all_unit = pd.concat([granger_dataframe_all_unit, granger_dataframe_all_trial])

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

    #for each unit, and each lag, calculate the average p value
    granger_dataframe_avg_all = pd.DataFrame()
    for i in granger_dataframe_all_unit['unit_id'].unique():
        for j in granger_dataframe_all_unit['trial_number'].unique():
            granger_dataframe_avg = granger_dataframe_all_unit[(granger_dataframe_all_unit['unit_id'] == i) & (granger_dataframe_all_unit['lag'] == j)]
            granger_dataframe_avg = granger_dataframe_avg.groupby('lag').mean()
            granger_dataframe_avg['unit_id'] = i
            granger_dataframe_avg_all = pd.concat([granger_dataframe_avg_all, granger_dataframe_avg])

    if export_to_csv:
        df_plv_all.to_csv('csvs/plv.csv')
        granger_dict_all_acrossunits.to_csv('csvs/granger.csv')
        # granger_dict_avg_acrossunits.to_csv('csvs/granger_avg.csv')

    return df_plv_all

def run_circular_correlation_test(df_theta_and_angle, export_to_csv=True):
    for trial in df_theta_and_angle['trial_number'].unique():
        df_trial = df_theta_and_angle[df_theta_and_angle['trial_number'] == trial]

        # Extract circular data
        angles1 = np.radians(df_trial['dlc_angle_phase'])
        angles2 = np.radians(df_trial['theta_phase'])

        # Calculate circular correlation coefficient
        circular_corr = pg.circ_corrcc(angles1, angles2)

        # Print the result
        print(f"Circular Correlation Coefficient for Trial {trial}: {circular_corr}")

        # Plotting
        plt.figure(figsize=(15, 6))
        plt.plot(df_trial['dlc_angle_phase'], label='[DLC] Head Angle Phase')
        plt.plot(df_trial['theta_phase'], label='Theta Phase')
        plt.ylabel('Phase')
        plt.xlabel('Time since start of trial (s)')
        plt.xticks(np.arange(0, len(df_trial['dlc_angle_phase']), 1000*50), labels=np.arange(0, len(df_trial['dlc_angle_phase'])/1000, 50))
        plt.legend()
        plt.title(f'DLC angle and theta phase for trial number {trial}')
        plt.savefig(f'figures/dlc_angle_theta_phase_trial_{trial}.png', dpi=300, bbox_inches='tight')
        #append to a dataframe
        circular_corr_df = pd.DataFrame({'circular_corr_value': circular_corr[0],'circ_corr_pvalue': circular_corr[1], 'trial_number': trial}, index=[0])
        if trial == 0:
            circular_corr_all_trials = circular_corr_df
        else:
            circular_corr_all_trials = pd.concat([circular_corr_all_trials, circular_corr_df])
    if export_to_csv:
        circular_corr_all_trials.to_csv('csvs/circular_corr.csv')
    return circular_corr_all_trials



def run_granger_cauality_test(df_theta_and_angle, export_to_csv = True, shuffle_data = False):
    #compare the granger causality between theta phase and dlc angle
    #for each trial
    for trial in df_theta_and_angle['trial_number'].unique():
        df_trial = df_theta_and_angle[df_theta_and_angle['trial_number'] == trial]
        #run the granger causality test
        adf_result_angle = adfuller(df_trial['dlc_angle_phase'])
        adf_result_theta = adfuller(df_trial['theta_phase'])

        # Check the p-values to determine stationarity
        is_stationary_angle = adf_result_angle[1] <= 0.05
        is_stationary_theta = adf_result_theta[1] <= 0.05

        if not is_stationary_angle or not is_stationary_theta:
            print(f"Trial {trial}: Not stationary. Applying differencing...")

            # Apply differencing to make the time series stationary
            dlc_angle_trial = np.diff(df_trial['dlc_angle_phase'])
            theta_phase_trial = np.diff(df_trial['theta_phase'])
            # Check the p-values again
            adf_result_angle = adfuller(df_trial['dlc_angle_phase'])
            adf_result_theta = adfuller(df_trial['theta_phase'])
            is_stationary_angle = adf_result_angle[1] <= 0.05
            is_stationary_theta = adf_result_theta[1] <= 0.05
            if not is_stationary_angle or not is_stationary_theta:
                print(f"Trial {trial}: Still not stationary. Skipping...")
                continue
            if shuffle_data == True:
                np.random.shuffle(dlc_angle_trial)
                np.random.shuffle(theta_phase_trial)

            granger_test = grangercausalitytests(np.column_stack((dlc_angle_trial, theta_phase_trial)), maxlag=20)
        else:
            #copy df_trial to dlc_angle_trial and theta_phase_trial
            dlc_angle_trial = df_trial['dlc_angle_phase'].copy()
            #convery to numpy
            dlc_angle_trial = dlc_angle_trial.to_numpy()
            theta_phase_trial = df_trial['theta_phase'].copy()
            theta_phase_trial = theta_phase_trial.to_numpy()
            if shuffle_data == True:
                np.random.shuffle(dlc_angle_trial)
                np.random.shuffle(theta_phase_trial)

            granger_test = grangercausalitytests(np.column_stack((dlc_angle_trial, theta_phase_trial)), maxlag=20)

        print(granger_test)
        #plot the dlc_angle and theta phase
        plt.figure(figsize=(40, 10))
        plt.plot(df_trial['dlc_angle_phase'], label = '[DLC] head angle phase')
        plt.plot(df_trial['theta_phase'], label = 'Theta phase')
        plt.ylabel('Phase')
        plt.xlabel('Time since start of trial (s)')
        plt.xticks(np.arange(0, len(df_trial['dlc_angle_phase']), 1000*50), labels=np.arange(0, len(df_trial['dlc_angle_phase'])/1000, 50))
        plt.legend()
        plt.title(f'DLC angle and theta phase for trial number {trial}')
        plt.savefig(f'figures/dlc_angle_theta_phase_trial_{trial}_shuffle_{shuffle_data}.png', dpi=300, bbox_inches='tight')

        for count, key in enumerate(granger_test.keys()):
            print('Granger test results: ' + str(granger_test[key][0]['ssr_ftest']))
            # add to a dataframe
            granger_test_for_indiv_lag = granger_test[key][0]['ssr_ftest']
            #apply a bonferroni correction for the p value
            #CHANGE BELOW LINE, 4 is the number of lags CURRENTLY
            corrected_bonferroni_p = granger_test_for_indiv_lag[1] * 4
            granger_test_lag_dataframe = pd.DataFrame(
                {'F-statistic': granger_test_for_indiv_lag[0], 'p-value': granger_test_for_indiv_lag[1], 'corrected p-value': corrected_bonferroni_p,
                 'df_denom': granger_test_for_indiv_lag[2], 'df_num': granger_test_for_indiv_lag[3]}, index=[0])
            granger_test_lag_dataframe['trial_number'] = trial
            granger_test_lag_dataframe['lag'] = key
            if count == 0:
                granger_dataframe_all_lag = granger_test_lag_dataframe
            else:
                granger_dataframe_all_lag = pd.concat([granger_dataframe_all_lag, granger_test_lag_dataframe])

        # append to a larger dataframe
        if trial == 0:
            granger_dataframe_all_trial = granger_dataframe_all_lag
        else:
            try:
                granger_dataframe_all_trial = pd.concat([granger_dataframe_all_trial, granger_dataframe_all_lag])
            except:
                granger_dataframe_all_trial = granger_dataframe_all_lag
    #get the mean for each lag
    granger_dataframe_all_trial['mean_p_value_for_lag'] = granger_dataframe_all_trial.groupby('lag')['p-value'].transform('mean')
    if export_to_csv:
        granger_dataframe_all_trial.to_csv(f'csvs/granger_trial_cumulative_shuffle_{shuffle_data}.csv')
    return granger_dataframe_all_trial





# # Function to create simulated time series data
# def simulate_data(n_samples, correlation_strength, lag_order):
#     # Generate independent random time series
#     np.random.seed(42)
#     x = np.random.randn(n_samples)
#
#     # Generate correlated time series (Granger causality)
#     y = np.zeros_like(x)
#     for t in range(lag_order, n_samples):
#         y[t] = correlation_strength * x[t - lag_order] + np.random.randn()
#
#     # Generate uncorrelated time series
#     z = np.random.randn(n_samples)
#
#     return x, y, z

def simulate_data(n_samples, correlation_strength, lag_order, sinusoid_frequency):
    # Generate sinusoidal time series for x
    t = np.arange(0, n_samples)
    x = np.sin(2 * np.pi * sinusoid_frequency * t / n_samples)

    # Generate correlated time series (Granger causality) for y
    y = np.zeros_like(x)
    for t in range(lag_order, n_samples):
        y[t] = correlation_strength * x[t - lag_order] + np.random.randn()

    # Generate uncorrelated sinusoidal time series for z
    z = np.sin(2 * np.pi * sinusoid_frequency * t / n_samples) + np.random.randn(n_samples)

    return x, y, z

def compare_simulated_data_to_granger_test(n_samples):
    '''create simulated time series data and compare the granger causality test when the time series are correlated and non correlated
    :param n_samples: the number of samples
    :return: the granger causality test results for the correlated and uncorrelated time series
    '''

    # Correlation strength for the Granger causality
    correlation_strength = 0.95
    print('running a granger causality test on simulated data, with a correlation strength of: ' + str(correlation_strength) + ' and ' + str(n_samples) + ' samples.')
    # Lag order for the Granger causality
    lag_order = 20

    # Simulate data
    x, y, z = simulate_data(n_samples, correlation_strength, lag_order, 5)

    # Plot the time series
    plt.figure(figsize=(40, 6))
    plt.plot(x, label='Independent Time Series (X)')
    plt.plot(y, label='Correlated Time Series (Y)')
    plt.plot(z, label='Uncorrelated Time Series (Z)')
    plt.legend()
    plt.title('Simulated Time Series Data')
    plt.xlabel('Time (in nonsense units)')
    plt.ylabel('Values')
    plt.savefig('figures/simulated_time_series.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Granger causality test
    #make sure it passes the adfuller test
    adf_result_x = adfuller(x)
    adf_result_y = adfuller(y)
    adf_result_z = adfuller(z)
    print('ADF test for x: ' + str(adf_result_x))
    print('ADF test for y: ' + str(adf_result_y))
    print('ADF test for z: ' + str(adf_result_z))


    result_xy = grangercausalitytests(np.column_stack((y, x)), maxlag=lag_order, verbose=True)
    result_xz = grangercausalitytests(np.column_stack((z, x)), maxlag=lag_order, verbose=True)
    #export to csv
    df_result_xy = pd.DataFrame(result_xy)
    df_result_xz = pd.DataFrame(result_xz)
    df_result_xy.to_csv('csvs/granger_simulated_xy_CORRELATED.csv')
    df_result_xz.to_csv('csvs/granger_simulated_xz_UNCORRELATED.csv')
    return result_xy, result_xz







def main():
    # result_correlated, result_uncorrelated = compare_simulated_data_to_granger_test(400*1000)
    phase_array, trial_array, theta_array, df_theta_and_angle = load_theta_data(Path('C:/neural_data/'), spike_data = [])
    # circ_corr_df = run_circular_correlation_test(df_theta_and_angle)
    granger_results = run_granger_cauality_test(df_theta_and_angle, shuffle_data=False)


    df_all = load_data_from_paths(Path('C:/neural_data/'))
    compare_spike_times_to_theta_phase(df_all, phase_array, theta_array, trial_array)





if __name__ == '__main__':
    main()