import scipy
import numpy as np
import pandas as pd
import os


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
    @staticmethod
    ###$$ GET_SPIKES_WITH_HISTORY #####
    def get_spikes_with_history(neural_data, bins_before, bins_after, bins_current=1):
        """
        Function that creates the covariate matrix of neural activity, taken from the Kording Lab at UPenn

        Parameters
        ----------
        neural_data: a matrix of size "number of time bins" x "number of neurons"
            the number of spikes in each time bin for each neuron
        bins_before: integer
            How many bins of neural data prior to the output are used for decoding
        bins_after: integer
            How many bins of neural data after the output are used for decoding
        bins_current: 0 or 1, optional, default=1
            Whether to use the concurrent time bin of neural data for decoding

        Returns
        -------
        X: a matrix of size "number of total time bins" x "number of surrounding time bins used for prediction" x "number of neurons"
            For every time bin, there are the firing rates of all neurons from the specified number of time bins before (and after)
        """

        num_examples = neural_data.shape[0]  # Number of total time bins we have neural data for
        num_neurons = neural_data.shape[1]  # Number of neurons
        surrounding_bins = bins_before + bins_after + bins_current  # Number of surrounding time bins used for prediction
        X = np.empty([num_examples, surrounding_bins, num_neurons])  # Initialize covariate matrix with NaNs
        X[:] = np.NaN
        # Loop through each time bin, and collect the spikes occurring in surrounding time bins
        # Note that the first "bins_before" and last "bins_after" rows of X will remain filled with NaNs, since they don't get filled in below.
        # This is because, for example, we cannot collect 10 time bins of spikes before time bin 8
        start_idx = 0
        for i in range(
                num_examples - bins_before - bins_after):  # The first bins_before and last bins_after bins don't get filled in
            end_idx = start_idx + surrounding_bins;  # The bins of neural data we will be including are between start_idx and end_idx (which will have length "surrounding_bins")
            X[i + bins_before, :, :] = neural_data[start_idx:end_idx,
                                       :]  # Put neural data from surrounding bins in X, starting at row "bins_before"
            start_idx = start_idx + 1
        return X
    @staticmethod
    def create_folds(n_timesteps, num_folds=5, num_windows=4):
        '''Create the folds for the cross validation that subsamples trials from overlapping segments, author: Jake Ormond 2024
        '''
        n_windows_total = num_folds * num_windows
        window_size = n_timesteps // n_windows_total
        window_start_ind = np.arange(0, n_timesteps, window_size)

        folds = []

        for i in range(num_folds):
            test_windows = np.arange(i, n_windows_total, num_folds)
            test_ind = []
            for j in test_windows:
                test_ind.extend(np.arange(window_start_ind[j], window_start_ind[j] + window_size))
            train_ind = list(set(range(n_timesteps)) - set(test_ind))

            folds.append((train_ind, test_ind))

        return folds
    @staticmethod
    def load_previous_results(directory_of_interest, window_size = 1000, bin_size = 250):
        param_dict = {}
        score_dict = {}
        # 'C:/neural_data/rat_3/25-3-2019'
        for rat_dir in ['C:/neural_data/rat_10/23-11-2021', 'C:/neural_data/rat_7/6-12-2019',
                        'C:/neural_data/rat_8/15-10-2019', 'C:/neural_data/rat_9/10-12-2021',
                        'C:/neural_data/rat_3/25-3-2019']:
            rat_id = rat_dir.split('/')[-2]
            #get all directories which contain the directory of interest
            list_of_directories = os.listdir(rat_dir)
            #get the directory of interest
            #check if keyword matches any of the directories
            for directory in list_of_directories:
                if directory.__contains__(directory_of_interest):
                    directory_of_interest = directory
                    break
            if directory_of_interest == None:
                print(f'{rat_id} does not have the directory of interest')
                continue
                # else:
                #     print(f'{rat_id} does not have the directory of interest')
                #     continue

            # if list_of_directories.__contains__(directory_of_interest):
            #     #get the index
            #     index = list_of_directories.index(directory_of_interest)
            #     #if index is multiple numbers, then choose the first one
            #     if type(index) == list:
            #         index = index[0]
            #     param_directory = f'{rat_dir}/{list_of_directories[index]}'
            # else:
            #     print(f'{rat_id} does not have the directory of interest')
            #     continue

            param_directory = f'{rat_dir}/{directory_of_interest}'

            # param_directory = f'{rat_dir}/{directory_of_interest}'
            # find all the files in the directory
            try:
                files = os.listdir(param_directory)
            except Exception as e:
                print(f'Error: {e}')
                continue

            for window in [window_size]:
                for bin_size in [bin_size]:
                    # find the file names
                    for file in files:
                        if file.__contains__(f'{window}windows') or file.__contains__(f'numwindows{window}'):
                            if file.__contains__('mean_score'):
                                score_dict[rat_id] = np.load(f'{param_directory}/{file}')
                            elif file.__contains__('params'):
                                with open(f'{param_directory}/{file}', 'rb') as f:
                                    param_dict[rat_id] = np.load(f'{param_directory}/{file}', allow_pickle=True)
        return param_dict, score_dict


