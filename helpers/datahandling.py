import scipy
import numpy as np
import pandas as pd

class DataHandler():
    '''A class to handle the data for the manifold neural project'''

    @staticmethod
    def load_data_from_paths(path):
        '''Load the spike data and the positional data from the local path
        and return a dataframe with the spike times and the dlc angle
        for each unit. The dlc and trial numbers are interpolated to a 30,000 Hz
        :param path: the local path to the data
        :return: a dataframe with the spike times and the dlc angle for each unit'''

        # Load MATLAB spike data from the local path:
        spike_data = scipy.io.loadmat(path / 'units.mat')
        units = spike_data['units']
        fs = spike_data['sample_rate'][0][0]

        positional_data = scipy.io.loadmat(path / 'positionalDataByTrialType.mat')
        pos_cell = positional_data['pos']
        hcomb_data = pos_cell[0][0][0][0]

        time = hcomb_data['videoTime']
        ts = hcomb_data['ts']
        sample = hcomb_data['sample']

        # Calculate sample rate
        fs = ((sample[0][0] / ts[0][0]) * 10000)[0]

        dlc_angle = hcomb_data['dlc_angle']

        df_all = pd.DataFrame()

        for j in range(len(units)):
            unit = units[j]
            spike_times = unit['spikeSamples'][0].astype(float) / fs

            head_angle_times = np.array([])
            dlc_angle_list = np.array([])
            head_angle_times_ms = np.array([])
            trial_number_array = np.array([])

            for i2 in range(len(dlc_angle)):
                trial_dlc, trial_ts, trial_sample = dlc_angle[i2], ts[i2], sample[i2]
                time_in_seconds = trial_sample / fs

                trial_number_full = np.full(len(trial_ts), i2)
                trial_number_array = np.append(trial_number_array, trial_number_full)

                head_angle_times = np.append(head_angle_times, time_in_seconds)
                head_angle_times_ms = np.append(head_angle_times_ms, trial_ts)
                dlc_angle_list = np.append(dlc_angle_list, trial_dlc)

                if np.max(time_in_seconds) > np.max(spike_times):
                    print('Trial time is greater than spike time, aborting...')
                    break

            # Interpolate spike times and dlc_angle to a common sample rate

            flattened_spike_times_seconds = np.concatenate(spike_times).ravel()
            flattened_spike_times = np.concatenate(unit['spikeSamples']).ravel()
            dlc_new = np.interp(flattened_spike_times_seconds*1000, head_angle_times_ms, dlc_angle_list)
            trial_new = np.interp(flattened_spike_times_seconds*1000, head_angle_times_ms, trial_number_array)

            # Create DataFrame for the current unit
            unit_id = np.full(len(flattened_spike_times), unit['name'][0].astype(str))
            phy_cluster = np.full(len(flattened_spike_times), unit['phyCluster'][0].astype(str))
            try:
                neuron_type = np.full(len(flattened_spike_times), unit['neuronType'][0][0][0][0].astype(str))
            except:
                neuron_type = np.full(len(flattened_spike_times), 'unclassified')

            df = pd.DataFrame({
                'spike_times_seconds': flattened_spike_times_seconds,
                'spike_times_samples': flattened_spike_times,
                'dlc_angle': dlc_new,
                'unit_id': unit_id,
                'phy_cluster': phy_cluster,
                'neuron_type': neuron_type,
                'trial_number': trial_new
            })

            # Append to the larger dataframe
            df_all = df_all.append(df, ignore_index=True)

        return df_all
