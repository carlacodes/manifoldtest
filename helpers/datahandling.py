import scipy
import numpy as np
import pandas as pd


class MathHelper():
    @staticmethod
    def multiInterp2(x, xp, fp):
        i = np.arange(len(x))
        j = np.searchsorted(xp, x) - 1

        # Clip indices to ensure they are within bounds
        j = np.clip(j, 0, len(xp) - 2)

        # Calculate the interpolation fraction 'd'
        d = (x - xp[j]) / (xp[j + 1] - xp[j])

        # Clip indices again to ensure they are within bounds
        j = np.clip(j, 0, len(xp) - 2)

        # Check if indices are valid
        valid_indices = (j >= 0) & (j < len(fp[0]))

        # Print debugging information
        print("x:", x)
        print("xp:", xp)
        print("fp.shape:", fp.shape)
        print("i:", i)
        print("j:", j)
        print("d:", d)

        # Interpolate only for valid indices
        # Interpolate only for valid indices
        result = np.zeros_like(x)
        result[valid_indices] = (1 - d[valid_indices]) * fp[i[valid_indices], j[valid_indices]] + d[valid_indices] * fp[
            i[valid_indices], np.clip(j[valid_indices] + 1, 0, len(fp[0]) - 1)]

        return result
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
        fs = ((sample[0][0] / ts[0][0]) * 1000)[0]

        dlc_angle = hcomb_data['dlc_angle']
        dlc_xy = hcomb_data['dlc_XYsmooth']

        df_all = pd.DataFrame()

        for j in range(len(units)):
            unit = units[j]
            #filter based on
            spike_times = unit['spikeSamples'][0].astype(float)

            spike_times = unit['spikeSamples'][0].astype(float) / fs

            head_angle_times = np.array([])
            dlc_angle_array = np.array([])
            head_angle_times_ms = np.array([])
            trial_number_array = np.array([])
            #initialise xy position as an empty 2d array
            xy_pos_array = np.array([])
            dlc_xy_array = np.empty((1, 2), dtype=float)

            #initial a trial identity array that is the same length as the spike times
            trial_number_full = np.full(len(spike_times), -1)

            for i2 in range(len(dlc_angle)):
                trial_dlc, trial_ts, trial_sample = dlc_angle[i2], ts[i2], sample[i2]
                time_in_seconds = trial_sample / 30000

                # trial_number_full = np.full(len(trial_ts), i2)
                # trial_number_array = np.append(trial_number_array, trial_number_full)
                #figure out which trials the spike times belong to
                mask = (spike_times > time_in_seconds[0]) & (spike_times < time_in_seconds[-1])
                mask = mask.ravel()
                trial_number_full[mask] = i2

                head_angle_times = np.append(head_angle_times, time_in_seconds)
                head_angle_times_ms = np.append(head_angle_times_ms, trial_ts)
                dlc_angle_array = np.append(dlc_angle_array, trial_dlc)
                dlc_xy_array = np.vstack((dlc_xy_array, dlc_xy[i2]))


                # if np.max(time_in_seconds) > np.max(spike_times):
                #     print('Trial time is greater than spike time, aborting...')
                #     break

            # Interpolate spike times and dlc_angle to a common sample rate
            #remove the first null row from the dlc_xy_array
            dlc_xy_array = dlc_xy_array[1:]
            flattened_spike_times_seconds = np.concatenate(spike_times).ravel()
            flattened_spike_times = np.concatenate(unit['spikeSamples']).ravel()
            dlc_new = np.interp(flattened_spike_times_seconds*1000, head_angle_times_ms, dlc_angle_array)
            # trial_new = np.interp(flattened_spike_times_seconds*1000, head_angle_times_ms, trial_number_array)
            xy_pos_new = MathHelper.multiInterp2(flattened_spike_times_seconds*1000, head_angle_times_ms, dlc_xy_array)
            #plot the trial_number_full
            # import matplotlib.pyplot as plt
            # fig, ax = plt.subplots()
            # ax.plot(flattened_spike_times_seconds, trial_number_full)
            # ax.set(xlabel='time (s)', ylabel='trial number',
            #        title='trial number')
            # ax.grid()
            # plt.show()

            # xy_pos_new = scipy.interpolate.griddata(flattened_spike_times_seconds * 1000, head_angle_times_ms, dlc_xy_array, method='linear')


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
                'dlc_xy': xy_pos_new,
                'unit_id': unit_id,
                'phy_cluster': phy_cluster,
                'neuron_type': neuron_type,
                'trial_number': trial_number_full

            })

            # Append to the larger dataframe
            df_all = pd.concat([df_all, df], ignore_index=True)
        #remove all rows where the trial number is -1, as this indicates the recording was not during a trial
        df_all = df_all[df_all['trial_number'] != -1]
        #get the max unqiue trial number
        max_trial_number = np.max(df_all['trial_number'])
        return df_all
