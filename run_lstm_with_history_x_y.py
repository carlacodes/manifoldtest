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
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, TimeSeriesSplit, permutation_test_score, GridSearchCV
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
        output_dim = 1  # regression problem, so output dimension is 1
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
            permutation_scores[i] = mean_squared_error(y_test_permuted, y_pred_permuted.detach().numpy())

        score = mean_squared_error(y_test, y_pred.detach().numpy())
        pvalue = (np.sum(permutation_scores >= score) + 1.0) / (n_permutations + 1.0)

        print(f"True score: {score}")
        # print(f"Permutation scores: {permutation_scores}")
        print(f'Mean permutation score: {np.mean(permutation_scores)}')
        print(f"Permutation test p-value: {pvalue}")

        #append the scores to a list
        current_score_df = pd.DataFrame(
            {'score': [score], 'pvalue': [pvalue], 'permutation_scores': [permutation_scores], 'mean_perm_score': [np.mean(permutation_scores)]})

        # Append the current scores to the main DataFrame
        score_df = pd.concat([score_df, current_score_df], ignore_index=True)
    return score_df





def run_lstm_with_history(data_dir):

    spike_dir = os.path.join(data_dir, 'physiology_data')
    # spike_trains = load_pickle('spike_trains', spike_dir)
    dlc_dir = os.path.join(data_dir, 'positional_data')

    # load labels
    labels = np.load(f'{dlc_dir}/labels.npy')
    spike_data = np.load(f'{spike_dir}/inputs.npy')

    bins_before = 6  # How many bins of neural data prior to the output are used for decoding
    bins_current = 1  # Whether to use concurrent time bin of neural data
    bins_after = 6  # How many bins of neural data after the output are used for decoding
    X = DataHandler.get_spikes_with_history(spike_data, bins_before, bins_after, bins_current)
    # remove the first six and last six bins
    X_for_lstm = X[6:-6]
    labels_for_umap = labels[6:-6]
    labels_for_umap = labels_for_umap[:, 0:5]
    # labels_for_umap = labels[:, 0:3]
    label_df = pd.DataFrame(labels_for_umap, columns=['x', 'y', 'angle_sin', 'angle_cos', 'dlc_angle_raw'])
    label_df['time_index'] = np.arange(0, label_df.shape[0])
    target_label = 'x'
    target = label_df[target_label].values

    #big dataframe
    big_score_df = pd.DataFrame()
    #isolate each of the 112 neurons and run the lstm on each of them
    for i in range(0, X_for_lstm.shape[2]):
        X_of_neuron = X_for_lstm[:, :, i]
        target_reshaped = target.reshape(-1, 1)

        #add back the third dimension
        X_of_neuron = X_of_neuron.reshape(X_of_neuron.shape[0], X_of_neuron.shape[1], 1)
        score_df_neuron = run_lstm(X_of_neuron, target_reshaped)
        score_df_neuron['neuron_index'] = i
        big_score_df = pd.concat([big_score_df, score_df_neuron], ignore_index=True)
    #save big_score_df to csv
    big_score_df.to_csv(f'{data_dir}/lstm_scores_{target_label}.csv')
    print('done')




def main():
    dir = 'C:/neural_data/rat_7/6-12-2019/'
    run_lstm_with_history(dir)
    return



if __name__ == '__main__':
    main()