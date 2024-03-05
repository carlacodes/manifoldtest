import copy
from pathlib import Path
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor
# from mpl_toolkits import mplot3d
import os
from sklearn.model_selection import permutation_test_score

from tqdm import tqdm
from joblib import Parallel, delayed
# from extractlfpandspikedata import load_theta_data
from helpers.load_and_save_data import load_pickle, save_pickle
from helpers.datahandling import DataHandler
from sklearn.svm import SVR
from umap import UMAP
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, TimeSeriesSplit, permutation_test_score, \
    GridSearchCV
import matplotlib.cm as cm
import numpy as np
import torch
from torch import nn


# Define LSTM model
class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_prob=0.2):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out


def run_lstm(X, y):
    # Define the TimeSeries Cross Validator
    tscv = TimeSeriesSplit(n_splits=5)
    # TimeSeries Cross Validation model evaluation
    fold_no = 1
    score_df = pd.DataFrame()
    for train, test in tscv.split(X):
        input_dim = X[train].shape[2]  # number of features
        hidden_dim = 400  # you can change this
        output_dim = 2  # regression problem, changing output dimension to 2
        model = LSTMNet(input_dim, hidden_dim, output_dim)

        # Define loss function and optimizer
        criterion = nn.MSELoss()  # for regression problem
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # you can change the learning rate

        # Convert numpy arrays to PyTorch tensors
        X_train_torch = torch.from_numpy(X[train]).float()
        y_train_torch = torch.from_numpy(y[train]).float()

        # Train the model
        num_epochs = 100  # you can change this
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_torch)
            loss = criterion(outputs, y_train_torch)
            loss.backward()
            optimizer.step()

        # Convert test data to PyTorch tensor
        X_test_torch = torch.from_numpy(X[test]).float()
        y_test = y[test]

        # Make predictions on the test set
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test_torch)

        # Increase fold number
        fold_no = fold_no + 1

        # Manual permutation test
        n_permutations = 100
        permutation_scores = np.zeros(n_permutations)
        for i in range(n_permutations):
            y_test_permuted = np.random.permutation(y_test)
            y_pred_permuted = model(torch.from_numpy(X_test_torch.numpy()).float())
            permutation_scores[i] = mean_squared_error(y_test_permuted, y_pred_permuted.detach().numpy(),
                                                       multioutput='raw_values').mean()  # calculate mean squared error for each output and then take the mean

        score = mean_squared_error(y_test, y_pred.detach().numpy(), multioutput='raw_values').mean()
        pvalue = (np.sum(permutation_scores >= score) + 1.0) / (n_permutations + 1.0)

        print(f"True score: {score}")
        # print(f"Permutation scores: {permutation_scores}")
        print(f'Mean permutation score: {np.mean(permutation_scores)}')
        print(f"Permutation test p-value: {pvalue}")

        # append the scores to a list
        current_score_df = pd.DataFrame(
            {'score': [score], 'pvalue': [pvalue], 'permutation_scores': [permutation_scores],
             'mean_perm_score': [np.mean(permutation_scores)]})

        # Append the current scores to the main DataFrame
        score_df = pd.concat([score_df, current_score_df], ignore_index=True)
    return score_df


def run_lstm_with_history(data_dir):
    spike_dir = os.path.join(data_dir, 'physiology_data')
    # spike_trains = load_pickle('spike_trains', spike_dir)
    dlc_dir = os.path.join(data_dir, 'positional_data')

    # load labels
    labels = np.load(f'{dlc_dir}/labels_2902.npy')
    spike_data = np.load(f'{spike_dir}/inputs.npy')
    # bin into 256 positions 16 x 16

    bins_before = 6  # How many bins of neural data prior to the output are used for decoding
    bins_current = 1  # Whether to use concurrent time bin of neural data
    bins_after = 6  # How many bins of neural data after the output are used for decoding
    frame_size = [496, 442]
    diff_xy = frame_size[0] - frame_size[1]
    x_edges = np.linspace(diff_xy/2 +1, frame_size[0] - diff_xy/2, 16+1)
    y_edges = np.linspace(0, frame_size[1], 16+1)

    x_data = labels[:, 0]
    y_data = labels[:, 1]
    # Bin the x and y position data
    # Bin the x and y position data
    # x_bins = np.digitize(x_data, x_edges) - 1  # subtract 1 to make the bins start from 0
    # y_bins = np.digitize(y_data, y_edges) - 1
    #
    # # Combine the x and y bin indices to create a 2D bin index
    # bin_indices = x_bins * 16 + y_bins  # assuming the size of y dimension is 16
    #
    # # Create an empty array of lists to store the spike data for each bin
    # binned_spike_data = np.empty((256, 1), dtype=object)
    # for i in range(256):
    #     binned_spike_data[i, 0] = []
    #
    # # Iterate over the spike data and the 2D bin indices
    # for spike, bin_index in zip(spike_data, bin_indices):
    #     # Append the spike data to the list in the corresponding bin
    #     binned_spike_data[bin_index, 0].append(spike)

    # Define the number of neurons
    num_neurons = 112

    # Bin the x and y position data
    x_bins = np.digitize(x_data, x_edges) - 1  # subtract 1 to make the bins start from 0
    y_bins = np.digitize(y_data, y_edges) - 1

    # Combine the x and y bin indices to create a 2D bin index
    bin_indices = x_bins * 16 + y_bins  # assuming the size of y dimension is 16

    # Create an empty array of lists to store the spike data for each bin and each neuron
    binned_spike_data = np.empty((256, num_neurons), dtype=object)
    for i in range(256):
        for j in range(num_neurons):
            binned_spike_data[i, j] = []

    # Iterate over the spike data and the 2D bin indices
    for spike, bin_index in zip(spike_data, bin_indices):
        # Append the spike data to the list in the corresponding bin for each neuron
        for neuron_index in range(num_neurons):
            binned_spike_data[bin_index, neuron_index].append(spike[neuron_index])
    max_time_points = max(len(spike_list) for spike_list in binned_spike_data.flatten())

    # Create a new 3D numpy array with shape (256, max_time_points, 112)
    reshaped_spike_data = np.zeros((256, max_time_points, 112))

    # Iterate over the binned_spike_data array
    for position_index in range(256):
        for neuron_index in range(112):
            # Get the spike data for the current position and neuron
            spike_data = binned_spike_data[position_index, neuron_index]

            # Convert the spike data to a numpy array and copy it into the reshaped_spike_data array
            reshaped_spike_data[position_index, :len(spike_data), neuron_index] = np.array(spike_data)
    # make binned

    # Flatten the binned_spike_data into a 256 x 1 array
    flattened_spike_data = binned_spike_data.reshape(-1, 1)



    X = DataHandler.get_spikes_with_history(spike_data, bins_before, bins_after, bins_current)
    # remove the first six and last six bins
    X_for_lstm = X[6:-6]
    labels_for_umap = labels[6:-6]
    labels_for_umap = labels_for_umap[:, 0:5]
    # labels_for_umap = labels[:, 0:3]
    label_df = pd.DataFrame(labels_for_umap, columns=['x', 'y', 'angle_sin', 'angle_cos', 'dlc_angle_raw'])
    label_df['time_index'] = np.arange(0, label_df.shape[0])
    target_label = 'xy'
    target_x = label_df['x'].values
    target_y = label_df['y'].values

    # Stack 'x' and 'y' to form a 2D target array
    target = np.column_stack((target_x, target_y))

    # big dataframe
    big_score_df = pd.DataFrame()
    # isolate each of the 112 neurons and run the lstm on each of them
    for i in range(0, X_for_lstm.shape[2]):
        X_of_neuron = X_for_lstm[:, :, i]
        target_reshaped = target.reshape(-1, 2)  # reshape to (-1, 2)

        # add back the third dimension
        X_of_neuron = X_of_neuron.reshape(X_of_neuron.shape[0], X_of_neuron.shape[1], 1)
        score_df_neuron = run_lstm(X_of_neuron, target_reshaped)
        score_df_neuron['neuron_index'] = i
        big_score_df = pd.concat([big_score_df, score_df_neuron], ignore_index=True)

    # save big_score_df to csv
    big_score_df.to_csv(f'{data_dir}/lstm_scores_{target_label}.csv')
    print('done')


def main():
    data_dir = 'C:/neural_data/rat_7/6-12-2019/'
    run_lstm_with_history(data_dir)
    return


if __name__ == '__main__':
    main()