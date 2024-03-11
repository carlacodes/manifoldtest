
#add sys.path.append to the top of the file
import sys
import copy
from pathlib import Path
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor
import os
from sklearn.model_selection import permutation_test_score
from tqdm import tqdm
from joblib import Parallel, delayed
# from extractlfpandspikedata import load_theta_data
from manifold_neural.helpers.datahandling import DataHandler
from sklearn.svm import SVR
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, TimeSeriesSplit, permutation_test_score, GridSearchCV
import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR

# Define LSTM model
# class LSTMNet(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, dropout_prob=0.25):
#         super(LSTMNet, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
#         self.dropout = nn.Dropout(dropout_prob)
#         self.fc = nn.Linear(hidden_dim, output_dim)
#
#     def forward(self, x):
#         lstm_out, _ = self.lstm(x)
#         out = self.fc(lstm_out[:, -1, :])
#         return out
import torch.nn.functional as F

import torch.nn.functional as F

class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout_prob=0.25):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout_prob)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)

        # Additional hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)
        ])

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)

        # Apply Batch Normalization only to the last hidden layer output
        lstm_out = self.batch_norm(lstm_out[:, -1, :])

        # Apply additional hidden layers
        for layer in self.hidden_layers:
            lstm_out = F.relu(layer(lstm_out))
            lstm_out = self.dropout(lstm_out)

        out = self.fc(lstm_out)
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
        criterion = nn.SmoothL1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Adjust the weight decay value
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)  # Adjust the step_size and gamma

        # Convert numpy arrays to PyTorch tensors
        X_train_torch = torch.from_numpy(X[train]).float()
        y_train_torch = torch.from_numpy(y[train]).float()

        # Train the model
        num_epochs = 500  # you can change this
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_torch)
            loss = criterion(outputs, y_train_torch)
            #print the loss for each epoch
            print(f'Epoch {epoch} has loss: {loss}')
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

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
            y_test_permuted = copy.deepcopy(y_pred)
            X_test_torch_permuted = copy.deepcopy(X_test_torch)
            X_test_torch_permuted = np.roll(X_test_torch_permuted, np.random.randint(0, X_test_torch_permuted.shape[0]), axis=0)

            model.eval()
            with torch.no_grad():
                y_pred_permuted = model(torch.from_numpy(X_test_torch_permuted.numpy()).float())
            #check if y_pred_permuted has nan
            if torch.isnan(y_pred_permuted).any():
                print('y_pred_permuted has nan')
            #check if X_test_torch has nan
            if torch.isnan(X_test_torch).any():
                print('X_test_torch has nan')

            permutation_scores[i] = mean_squared_error(y_test_permuted, y_pred_permuted.detach().numpy(),
                                                       multioutput='raw_values').mean()  # calculate mean squared error for each output and then take the mean

        score_mse = mean_squared_error(y_test, y_pred.detach().numpy(), multioutput='raw_values').mean()
        pvalue = (np.sum(permutation_scores >= score_mse) + 1.0) / (n_permutations + 1.0)

        print(f"True score: {score_mse}")
        # print(f"Permutation scores: {permutation_scores}")
        print(f'Mean permutation score: {np.mean(permutation_scores)}')
        print(f"Permutation test p-value: {pvalue}")

        #append the scores to a list
        current_score_df = pd.DataFrame(
            {'score': [score_mse], 'pvalue': [pvalue], 'permutation_scores': [permutation_scores], 'mean_perm_score': [np.mean(permutation_scores)]})

        # Append the current scores to the main DataFrame
        score_df = pd.concat([score_df, current_score_df], ignore_index=True)
    return score_df





def run_lstm_with_history(data_dir, rat_id = 'unknown_rat'):

    spike_dir = os.path.join(data_dir, 'physiology_data')
    # spike_trains = load_pickle('spike_trains', spike_dir)
    dlc_dir = os.path.join(data_dir, 'positional_data')

    # load labels
    labels = np.load(f'{dlc_dir}/labels_0503_with_dist2goal_scale_data_False_zscore_data_True.npy')
    spike_data = np.load(f'{spike_dir}/inputs.npy')

    bins_before = 6  # How many bins of neural data prior to the output are used for decoding
    bins_current = 1  # Whether to use concurrent time bin of neural data
    bins_after = 6  # How many bins of neural data after the output are used for decoding
    X = DataHandler.get_spikes_with_history(spike_data, bins_before, bins_after, bins_current)
    # remove the first six and last six bins
    X_for_lstm = X[6:-6]
    labels_for_umap = labels[6:-6]
    labels_for_umap = labels_for_umap[:, 0:6]
    # labels_for_umap = labels[:, 0:3]
    label_df = pd.DataFrame(labels_for_umap, columns=['x', 'y', 'dist2goal', 'angle_sin', 'angle_cos', 'dlc_angle_zscore'])
    label_df['time_index'] = np.arange(0, label_df.shape[0])
    target_label = 'angle_sincos_zscore'
    target_sin = label_df['angle_sin'].values
    target_cos = label_df['angle_cos'].values

    # Stack 'x' and 'y' to form a 2D target array
    target = np.column_stack((target_sin, target_cos))

    #big dataframe
    big_score_df = pd.DataFrame()
    #isolate each of the 112 neurons and run the lstm on each of them
    for i in range(0, X_for_lstm.shape[2]):
        X_of_neuron = X_for_lstm[:, :, i]
        #z-score the input data
        X_of_neuron = (X_of_neuron - np.mean(X_of_neuron, axis=0)) / np.std(X_of_neuron, axis=0)
        target_reshaped = target.reshape(-1, 2)  # reshape to (-1, 2)

        # add back the third dimension
        X_of_neuron = X_of_neuron.reshape(X_of_neuron.shape[0], X_of_neuron.shape[1], 1)
        score_df_neuron = run_lstm(X_of_neuron, target_reshaped)
        score_df_neuron['neuron_index'] = i
        big_score_df = pd.concat([big_score_df, score_df_neuron], ignore_index=True)

    #save big_score_df to csv
    big_score_df.to_csv(f'{data_dir}/csvs_0603/lstm_scores_{target_label}_rat_{rat_id}.csv')
    print('done')




def main():
    # data_dir = '/home/zceccgr/Scratch/zceccgr/jake/rat_7/6-12-2019/'
    # run_lstm_with_history(data_dir)
    big_dir = 'C:/neural_data/'

    for rat in [3, 8, 9, 10, 7]:
        #get the list of folders directory that have dates
        print(f'now starting rat:{rat}')
        dates = os.listdir(os.path.join(big_dir, f'rat_{rat}'))
        #check if the folder name is a date by checking if it contains a hyphen
        date = [d for d in dates if '-' in d][0]
        data_dir = os.path.join(big_dir, f'rat_{rat}', date)
        if Path(f'{data_dir}/csvs_0603/lstm_scores_angle_sincos_zscore_rat_{rat}.csv').is_file():
            print(f'lstm scores for rat {rat} already computed')
            continue
        run_lstm_with_history(data_dir, rat_id = rat)
    return




if __name__ == '__main__':
    main()