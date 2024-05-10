# code for generating the dataset for use with PyTorch
import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io

import sys
sys.path.append('C:/Users/Jake/Documents/python_code/robot_maze_analysis_code')
# from utilities.get_directories import get_data_dir, get_robot_maze_directory
# from utilities.load_and_save_data import load_pickle, save_pickle
# from spikes.calculate_spike_pos_hd import interpolate_rads
# from behaviour.load_behaviour import split_dictionary_by_goal, get_goals

# code for generating the dataset for use with PyTorch
import os
import numpy as np
import pandas as pd
from get_directories import get_data_dir, get_robot_maze_directory
from load_and_save_data import load_pickle, save_pickle
from calculate_spike_pos_hd import interpolate_rads

def smooth_positional_data(dlc_data, window_size=100):  
    # THIS IS JUST A PLACEHOLDER FOR NOW   
    pass


def create_positional_trains(dlc_data, window_size=100, sample_freq=30000): # we'll default to 100 ms windows for now

    # first, get the start and end time of the video
    window_in_samples = window_size * sample_freq / 1000 # convert window size to samples
    windowed_dlc = {}
    window_edges = {}
    for k in dlc_data.keys():
        start_time = dlc_data[k].video_samples.iloc[0]
        end_time = dlc_data[k].video_samples.iloc[-1]

        # get the bin edges for the windows, these will be returned and used to bin the spikes
        window_edges[k] = np.arange(start_time, end_time, window_in_samples)
        # calculate the window centres. Because of the 50% overlap, they are simply the 
        # window edges with the first and last values excluded
        # window_centres = window_edges[k][1:-1]
        # window_centres are the midpoints of the windows
        window_centres = window_edges[k][:-1] + window_in_samples / 2

        # create a dataframe and make window_centres the video_samples column
        windowed_dlc[k] = pd.DataFrame(window_centres, columns=['video_samples'])

        # interpolate the x and y position, and goal distances data using window_centres
        cols_to_lin_interp = ['x', 'y', 'dist2goal']
        # find the cols that begin "distance_to_goal_", but don't end "platform"
        # and add them to the list
        # cols_to_lin_interp.extend([col for col in dlc_data[k].columns if \
        #                col.startswith('distance_to_goal_') and not \
        #                col.endswith('platform')])
        
        for c in cols_to_lin_interp:
            # make the column c in windowed_dlc[k] equal to the linearly interpolated values
            # of the column c in dlc_data[k] at the window_centres
            windowed_dlc[k][c] = np.round(np.interp(window_centres, \
                                dlc_data[k].video_samples, dlc_data[k][c]),1)
      
        # use interpolate_rads to interpolate the head direction data, and data in columns that 
        # begin "relative_direction" but not including those that begin "relative_direction_to" 
        cols_to_rad_interp = ['hd', 'relative_direction_to_goal']
        # cols_to_rad_interp.extend([col for col in dlc_data[k].columns if \
        #                col.startswith('relative_direction_') and not \
        #                col.startswith('relative_direction_to')])
        
        for c in cols_to_rad_interp:
            windowed_dlc[k][c] = np.round(interpolate_rads(dlc_data[k].video_samples, \
                                    dlc_data[k][c], window_centres), 2)
           
    return windowed_dlc, window_edges, window_size


def create_spike_trains(units, window_edges, window_size, overlap=None): 

          
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
           
            spike_rate = binned_spikes / (window_size / 1000)
            spike_trains[u][k] = spike_rate
            
    return spike_trains


def cat_dlc(windowed_dlc):
    # concatenate data from all trials into np.arrays for training
    # we will keep columns x, y, and the 2 distance to goal columns.
    # the hd and relative_direction columns 
    # will be converted to sin and cos and concatenated with the other data
    
    for i, k in enumerate(windowed_dlc):
        if i == 0:
            # get the column names
            # columns = windowed_dlc[k].columns

            # find the distance to goal columns
            # distance_cols = [c for c in columns if c.startswith('distance_to_goal_')]

            # find the relative direction columns (but not relative_direction_to columns)
            # relative_direction_cols = [c for c in columns if c.startswith('relative_direction_') \
            #                            and not c.startswith('relative_direction_screen')]

            # column_names = ['x', 'y'] 
            # column_names.extend(distance_cols)
            # column_names.extend(['hd']) 
            # column_names.extend(relative_direction_cols)  
    
            # total_num_cols = 10 # x, y, distance_to_goal x2, hd x2, relative_direction x 4
            column_names = ['x', 'y', 'dist2goal', 'hd', 'relative_direction_to_goal']
            total_num_cols = 7 # x, y, dist2goal, sin(hd), cos(hd), sin(relative_direction), cos(relative_direction)

    
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
                temp_array[:, count+1] = np.cos(windowed_dlc[k][c].values)
                count += 2
        
        if i == 0:
            dlc_array = temp_array.copy()
        else:
            dlc_array = np.concatenate((dlc_array, temp_array), axis=0)
        
    # # all columns need to be scaled to the range 0-1
    # for i, col_name in zip(dlc_array.shape[1], column_names) :
    #     # dlc_array[:, i] = (dlc_array[:, i] - np.min(dlc_array[:, i])) / \
    #     #                     (np.max(dlc_array[:, i]) - np.min(dlc_array[:, i]))
    #
    #     # z-score scaling
    #     print(f'z-scoring column {col_name}')
    #     dlc_array[:, i] = (dlc_array[:, i] - np.mean(dlc_array[:, i])) / np.std(dlc_array[:, i])

    for i, col_name in zip(range(dlc_array.shape[1]), column_names):
        # z-score scaling
        if col_name not in ['hd', 'relative_direction_to_goal']:
            print(f'z-scoring column {col_name}')
            dlc_array[:, i] = (dlc_array[:, i] - np.mean(dlc_array[:, i])) / np.std(dlc_array[:, i])

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
            temp_array[j,:] = spike_trains[u][k]

        if i == 0:
            spike_array = temp_array.copy()

        else:
            spike_array = np.concatenate((spike_array, temp_array), axis=1)

    spike_array = np.round(spike_array, 3)
    spike_array = np.transpose(spike_array)

    # zscore the columns of spike_array
    # spike_array = (spike_array - np.mean(spike_array, axis=0)) / np.std(spike_array, axis=0)
    
    return spike_array, unit_list

def cat_spike_trains_by_goal(spike_trains, data_dir):
    # get list of units
    unit_list = list(spike_trains.keys())

    goals = get_goals(data_dir)
    spike_array_dict = {}
    spike_trains_by_goal = {}
    for g in goals:
        spike_array_dict[g] = {}
        spike_trains_by_goal[g] = {}

    for u in unit_list:
        spike_train = spike_trains[u]
        spike_train_by_goal = split_dictionary_by_goal(spike_train, data_dir)

        for g in goals:
            spike_trains_by_goal[g][u] = spike_train_by_goal[g]

    for g in goals:
        spike_array_dict[g], unit_list = cat_spike_trains(spike_trains_by_goal[g])
    
    return spike_array_dict, unit_list


def create_datasets_from_robot_maze():
    animal = 'Rat46'
    session = '19-02-2024'
    data_dir = get_data_dir(animal, session)
 
   #  data_dir = 'D:/analysis/og_honeycomb/rat7/6-12-2019'
    # data_dir = '/media/jake/DataStorage_6TB/DATA/neural_network/og_honeycomb/rat7/6-12-2019'

    # load spike data
    spike_dir = os.path.join(data_dir, 'spike_sorting')
    units = load_pickle('units_w_behav_correlates', spike_dir)
    # units = load_pickle('restricted_units', spike_dir)

    # load positional data
    dlc_dir = os.path.join(data_dir, 'deeplabcut')
    dlc_data  = load_pickle('dlc_final', dlc_dir)
    # dlc_data = dlc_data['hComb']


    window_sizes = [100, 250]

    for window_size in window_sizes:
        # create positional and spike trains 
        # and save as a pickle file
        windowed_dlc, window_edges, window_size = \
            create_positional_trains(dlc_data, window_size=window_size)    
        windowed_data = {'windowed_dlc': windowed_dlc, 'window_edges': window_edges}
        dlc_train_file_name = f'windowed_dlc_{window_size}'
        save_pickle(windowed_data, dlc_train_file_name, dlc_dir)

        windowed_data = load_pickle(dlc_train_file_name, dlc_dir)
        windowed_dlc = windowed_data['windowed_dlc']
        window_edges = windowed_data['window_edges']

        # plot the windowed data against the original data (plotted, looked fine so commented out)
        # for k in windowed_dlc.keys():
        #     # plot the x and y position
        #     plt.figure()
        #     # plot the original data in blue
        #     # plt.plot(dlc_data[k].video_samples, dlc_data[k].x, 'b')
        #     # plt.plot(dlc_data[k].video_samples, dlc_data[k].y, 'b')
        #     plt.plot(dlc_data[k].video_samples, dlc_data[k].hd, 'b')

        #     # plot the windowed data in red
        #     # plt.plot(windowed_dlc[k].video_samples, windowed_dlc[k].x + 10, 'r')
        #     # plt.plot(windowed_dlc[k].video_samples, windowed_dlc[k].y + 10, 'r')
        #     plt.plot(windowed_dlc[k].video_samples, windowed_dlc[k].hd + 1, 'r')
        #     plt.title(k)
        #     plt.show()
        #     plt.close()

        # create spike trains
        spike_trains = create_spike_trains(units, window_edges, window_size=window_size)
        spike_train_file_name = f'spike_trains_{window_size}'
        save_pickle(spike_trains, spike_train_file_name, spike_dir)

        spike_trains = load_pickle(spike_train_file_name, spike_dir)

        # split the dlc data by goal
        dlc_by_goal = split_dictionary_by_goal(windowed_dlc, data_dir)
        # concatenate data from all trials into np.arrays for training
        goals = list(dlc_by_goal.keys())
        for g in goals:
            labels = cat_dlc(dlc_by_goal[g])
            labels = labels.astype(np.float32)
            labels_file_name = f'labels_goal{g}_ws{window_size}'
            np.save(f'{dlc_dir}/{labels_file_name}.npy', labels)

        labels = cat_dlc(windowed_dlc)
        labels = labels.astype(np.float32)
        labels_file_name = f'labels_{window_size}'
        np.save(f'{dlc_dir}/{labels_file_name}.npy', labels)

        # concatenate spike trains into np.arrays for training
        
        model_inputs_by_goal, unit_list = cat_spike_trains_by_goal(spike_trains, data_dir)  
        for g in goals:
            # convert model_inputs to float32
            model_inputs = model_inputs_by_goal[g].astype(np.float32)
            inputs_file_name = f'inputs_goal{g}_ws{window_size}'
            np.save(f'{spike_dir}/{inputs_file_name}', model_inputs)
                
        model_inputs, unit_list = cat_spike_trains(spike_trains)
        # convert model_inputs to float32
        model_inputs = model_inputs.astype(np.float32)
        inputs_file_name = f'inputs_{window_size}'
        np.save(f'{spike_dir}/{inputs_file_name}', model_inputs)
        save_pickle(unit_list, 'unit_list', spike_dir)
        pass 


def get_goal_coordinates(data_dir):
    
    goal_position_path = os.path.join(data_dir, 'goalCoordinates.mat')
    goal_position = scipy.io.loadmat(goal_position_path)
    goal_position = goal_position['goalCoor']
    #un nest the array
    goal_position = goal_position[0][0]
    #convert the goal position to an array, convert from void type to float
    goal_position = np.array(goal_position.tolist())
    #remove the first dimension
    goal_position = goal_position[0]

    center_x = np.mean(goal_position[:, 0])
    center_y = np.mean(goal_position[:, 1])
    goal_coordinates = np.array([center_x, center_y])

    return goal_coordinates


def calculate_rel_direction_to_goal(dlc_data, goal_coordinates):
    # calculate the direction to the goal from the x and y position
    x = dlc_data['x']
    y = dlc_data['y']

    dx = goal_coordinates[0] - x
    dy = y - goal_coordinates[1]

    angle = np.arctan2(dy, dx)
    angle = np.round(angle, 2)

    hd = dlc_data['hd']
    relative_direction = hd - angle
    relative_direction = np.round(relative_direction, 2)
    relative_direction = np.where(relative_direction < -np.pi, relative_direction + 2*np.pi, relative_direction)
    relative_direction = np.where(relative_direction > np.pi, relative_direction - 2*np.pi, relative_direction)

    return relative_direction


def calculate_rel_direction_across_trials(dlc_data, goal_coordinates):

    for k in dlc_data.keys():
        relative_direction = calculate_rel_direction_to_goal(dlc_data[k], goal_coordinates)
        dlc_data[k]['relative_direction_to_goal'] = relative_direction

    return dlc_data





def main():
    # big_dir = 'C:/neural_data/'

    data_dir = 'C:/neural_data/'
    rat_and_session = ['rat_3/25-3-2019', 'rat_7/6-12-2019', 'rat_8/15-10-2019', 'rat_9/10-12-2021', 'rat_10/23-11-2021']
    sample_freqs = {'rat_3': 20000, 'rat_7': 30000, 'rat_8': 30000, 'rat_9': 30000, 'rat_10': 30000}

    window_sizes = [20, 50, 100, 250, 500]

    for rs in rat_and_session:

        rat = rs.split('/')[0]
        sample_freq = sample_freqs[rat]

        # load spike data
        spike_dir = os.path.join(data_dir, rs, 'physiology_data')
        # units = load_pickle('units_w_behav_correlates', spike_dir)
        units = load_pickle('restricted_units', spike_dir)

        # load positional data
        dlc_dir = os.path.join(data_dir, rs, 'positional_data')
        # dlc_data  = load_pickle('dlc_final', dlc_dir)
        dlc_data = load_pickle('dlc_data', dlc_dir)
        dlc_data = dlc_data['hComb']

        # get goal coordinates
        goal_coordinates = get_goal_coordinates(dlc_dir)

        # calculate relative direction to goal
        dlc_data = calculate_rel_direction_across_trials(dlc_data, goal_coordinates)

        for window_size in window_sizes:
            windowed_dlc, window_edges, window_size = \
                create_positional_trains(dlc_data, window_size=window_size, sample_freq=sample_freq)    
            windowed_data = {'windowed_dlc': windowed_dlc, 'window_edges': window_edges}
            dlc_train_file_name = f'windowed_dlc_{window_size}'
            save_pickle(windowed_data, dlc_train_file_name, dlc_dir)

            windowed_data = load_pickle(dlc_train_file_name, dlc_dir)
            windowed_dlc = windowed_data['windowed_dlc']
            window_edges = windowed_data['window_edges']

            labels = cat_dlc(windowed_dlc)
            labels = labels.astype(np.float32)
            labels_file_name = f'labels_{window_size}'
            np.save(f'{dlc_dir}/{labels_file_name}.npy', labels)

            # create spike trains
            spike_trains = create_spike_trains(units, window_edges, window_size=window_size)
            spike_train_file_name = f'spike_trains_{window_size}'
            save_pickle(spike_trains, spike_train_file_name, spike_dir)

            spike_trains = load_pickle(spike_train_file_name, spike_dir)

            model_inputs, unit_list = cat_spike_trains(spike_trains)
            # convert model_inputs to float32
            model_inputs = model_inputs.astype(np.float32)
            inputs_file_name = f'inputs_10052024_{window_size}'
            np.save(f'{spike_dir}/{inputs_file_name}', model_inputs)
            save_pickle(unit_list, 'unit_list', spike_dir)
            print('..finished saving for...', rs, window_size)

        

if __name__ == "__main__":
    main()