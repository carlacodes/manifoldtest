import copy
from pathlib import Path
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor
# from mpl_toolkits import mplot3d
import os
from tqdm import tqdm
from joblib import Parallel, delayed
# from extractlfpandspikedata import load_theta_data
from helpers.load_and_save_data import load_pickle, save_pickle
from helpers.datahandling import DataHandler
from sklearn.svm import SVR
from umap import UMAP
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
from torch import nn

# Define LSTM model
class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

# Instantiate the model

def run_lstm():
    #split by timeseries
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    input_dim = X_train.shape[2]  # number of features
    hidden_dim = 50  # you can change this
    output_dim = 1  # regression problem, so output dimension is 1
    model = LSTMNet(input_dim, hidden_dim, output_dim)

    # Define loss function and optimizer
    criterion = nn.MSELoss()  # for regression problem
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # you can change the learning rate

    # Convert numpy arrays to PyTorch tensors
    X_train_torch = torch.from_numpy(X_train).float()
    y_train_torch = torch.from_numpy(y_train).float()

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
    X_test_torch = torch.from_numpy(X_test).float()

    # Make predictions on the test set
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_torch)




def run_umap_with_history(data_dir):

    spike_dir = os.path.join(data_dir, 'physiology_data')
    # spike_trains = load_pickle('spike_trains', spike_dir)
    dlc_dir = os.path.join(data_dir, 'positional_data')

    # load labels
    labels = np.load(f'{dlc_dir}/labels_2902.npy')
    spike_data = np.load(f'{spike_dir}/inputs.npy')

    bins_before = 6  # How many bins of neural data prior to the output are used for decoding
    bins_current = 1  # Whether to use concurrent time bin of neural data
    bins_after = 6  # How many bins of neural data after the output are used for decoding
    X = DataHandler.get_spikes_with_history(spike_data, bins_before, bins_after, bins_current)
    # remove the first six and last six bins
    X_for_umap = X[6:-6]
    labels_for_umap = labels[6:-6]
    labels_for_umap = labels_for_umap[:, 0:5]
    # labels_for_umap = labels[:, 0:3]
    label_df = pd.DataFrame(labels_for_umap, columns=['x', 'y', 'angle_sin', 'angle_cos', 'dlc_angle_norm'])
    label_df['time_index'] = np.arange(0, label_df.shape[0])
    # unsupervised_umap(X_for_umap, label_df, remove_low_variance_neurons=False, n_components=3)

    bin_width = 0.5
    window_for_decoding = 6  # in s
    window_size = int(window_for_decoding / bin_width)  # in bins

    n_runs = 1

    regressor = SVR
    regressor_kwargs = {'kernel': 'linear', 'C': 1}

    reducer = UMAP

    reducer_kwargs = {
        'n_components': 3,
        'n_neighbors': 70,
        'min_dist': 0.3,
        'metric': 'euclidean',
        'n_jobs': 1,
    }

    # space_ref = ['No Noise', 'Noise']
    # temporarily remove the space_ref variable, I don't want to incorporate separate data yet
    regress = 'dlc_angle_norm'  # Assuming 'head_angle' is the column in your DataFrame for regression

    # Use KFold for regression
    # kf = KFold(n_splits=5, shuffle=True)

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f'results_{now}.npy'
    results_between = {}
    results_within = {}
    results_w_perm_reduced = {}
    n_permutations = 5
    for run in range(n_runs):
        results_between[run] = {}
        results_within[run] = {}
        # for space in space_ref:

        # results_between[run] = train_ref_classify_rest(
        #     X_for_umap,
        #     label_df,
        #     regress,
        #     regressor,
        #     regressor_kwargs,
        #     reducer,
        #     reducer_kwargs,
        #     window_size,
        #     n_permutations=n_permutations,
        # )
        results_w_perm_reduced[run] = train_and_test_on_reduced(
            X_for_umap,
            label_df,
            regress,
            regressor,
            regressor_kwargs,
            reducer,
            reducer_kwargs,
            window_size,
            n_permutations=n_permutations, n_jobs_parallel=5
        )

        # Save results
    results = {'between': results_between, 'within': results_w_perm_reduced}
    save_path = Path('C:/neural_data/rat_7/6-12-2019')
    save_path.mkdir(exist_ok=True)
    np.save(save_path / filename, results)
def main():
    dir = 'C:/neural_data/rat_7/6-12-2019/'
    run_umap_with_history(dir)
    return



if __name__ == '__main__':
    main()