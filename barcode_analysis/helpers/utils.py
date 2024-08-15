import numpy as np
import numpy as np
from scipy.optimize import curve_fit

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
def generate_white_noise_control(df, noise_level=1.0, iterations=1000, savedir = ''):
    r_squared_values = []
    for _ in range(iterations):
        noise = np.random.normal(0, noise_level, len(df))
        noise_df = df.copy()
        noise_df['death_minus_birth'] += noise
        _, r_squared_df = fit_sinusoid_data_filtered(noise_df, savedir , threshold=0)
        r_squared_values.append(r_squared_df['R-squared'].mean())
    return r_squared_values

def fit_sinusoid_data_filtered(df, save_dir, cumulative_param=False, trial_number=None, shuffled_control=False,
                               threshold=0):
    """
    Fit a sinusoidal function to the filtered dataset where death_minus_birth is above a threshold.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing the raw data to fit.
    save_dir: str
        Directory to save the plot.
    threshold: float
        Threshold for filtering death_minus_birth values.
    """

    def sinusoidal(x, A, B, C, D):
        return A * np.sin(B * x + C) + D

    def calculate_goodness_of_fit(x_data, y_data, y_fitted):
        residuals = y_data - y_fitted
        ssr = np.sum(residuals ** 2)
        sst = np.sum((y_data - np.mean(y_data)) ** 2)
        r_squared = 1 - (ssr / sst)
        return r_squared

    fit_params = {}
    r_squared_df = pd.DataFrame(columns=['Dimension', 'R-squared'])

    for dim in df['Dimension'].unique():
        dim_data = df[df['Dimension'] == dim]

        # Filter data based on the threshold
        filtered_data = dim_data[dim_data['death_minus_birth'] > threshold]
        if filtered_data.empty:
            print(f"No data points above threshold {threshold} for dimension {dim} or less than 4 data points.")
            continue

        x_data = filtered_data['Interval'].values
        y_data = filtered_data['death_minus_birth'].values
        unique_x = np.unique(x_data)
        y_data_new = []
        for i in unique_x:
            y_data_new.append(np.mean(y_data[x_data == i]))
        if len(y_data_new) < 5:
            print('not enough points, less than 10, risk of overfitting')
            continue

        initial_guess = [1, 1, 0, np.mean(y_data_new)]
        try:
            params, _ = curve_fit(sinusoidal, unique_x, y_data_new, p0=initial_guess)
        except Exception as e:
            print(f'Failed to fit sinusoidal function for dimension {dim}, error: {e}')
            continue

        fit_params[dim] = params
        r_squared = calculate_goodness_of_fit(unique_x, y_data_new, sinusoidal(unique_x, *params))
        r_squared_df_individual = pd.DataFrame([{'Dimension': dim, 'R-squared': r_squared}])

        # Concatenate with the existing DataFrame
        r_squared_df = pd.concat([r_squared_df, r_squared_df_individual], ignore_index=True)
        print(f'R-squared for dimension {dim}: {r_squared}')

        plt.figure(figsize=(10, 6))
        plt.plot(unique_x, y_data_new, 'bo', label='Filtered Data')
        plt.plot(unique_x, sinusoidal(unique_x, *params), 'r-', label='Fitted Function')
        plt.title(f'Sinusoidal Fit for Homology Dimension {dim}, r2 score: {r_squared:.2f}')
        plt.xlabel('Interval (j)')
        plt.ylabel('Death - Birth')
        plt.legend()
        plt.savefig(
            f'{save_dir}/sinusoidal_fit_dimension_{dim}_filtered_cumulative_{cumulative_param}_trial_{trial_number}_shuffled_{shuffled_control}.png',
            dpi=300, bbox_inches='tight')
        plt.show()
        plt.close('all')

    r_squared_df.to_csv(
        f'{save_dir}/r_squared_values_filtered_cumulative_{cumulative_param}_shuffle_{shuffled_control}.csv',
        index=False)

    return fit_params, r_squared_df


def fit_sinusoid_data_whole(df, save_dir, cumulative_param = False, trial_number = None, shuffled_control = False):
    """
    Fit a sinusoidal function to the entire dataset and plot the results.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing the data to fit.
    save_dir: str
        Directory to save the plot.
    """
    def sinusoidal(x, A, B, C, D):
        return A * np.sin(B * x + C) + D

    def calculate_goodness_of_fit(x_data, y_data, y_fitted):
        residuals = y_data - y_fitted
        ssr = np.sum(residuals ** 2)
        sst = np.sum((y_data - np.mean(y_data)) ** 2)
        r_squared = 1 - (ssr / sst)
        return r_squared

    heatmap_data = df.pivot_table(index='Interval', columns='Dimension', values='death_minus_birth', aggfunc='mean')

    fit_params = {}
    r_squared_df = pd.DataFrame(columns=['Dimension', 'R-squared'])

    for dim in heatmap_data.columns:
        x_data = heatmap_data.index.values
        y_data = heatmap_data[dim].values
        initial_guess = [1, 1, 0, np.mean(y_data)]
        try:
            params, _ = curve_fit(sinusoidal, x_data, y_data, p0=initial_guess)
        except Exception as e:
            print(f'Failed to fit sinusoidal function for dimension {dim}, error: {e}')
            continue

        fit_params[dim] = params
        r_squared = calculate_goodness_of_fit(x_data, y_data, sinusoidal(x_data, *params))
        r_squared_df_indiv = pd.DataFrame({'Dimension': dim, 'R-squared': r_squared}, index=[0])
        r_squared_df = pd.concat([r_squared_df, r_squared_df_indiv], ignore_index=True)
        print(f'R-squared for dimension {dim}: {r_squared}')

        plt.figure(figsize=(10, 6))
        plt.plot(x_data, y_data, 'bo', label='Original Data')
        plt.plot(x_data, sinusoidal(x_data, *params), 'r-', label='Fitted Function')
        plt.title(f'Sinusoidal Fit for Homology Dimension {dim}, r2 score: {r_squared:.2f}')
        plt.xlabel('Interval (j)')
        plt.ylabel('Mean Death - Birth')
        plt.legend()
        plt.savefig(f'{save_dir}/sinusoidal_fit_dimension_{dim}_cumulative_{cumulative_param}_trial_{trial_number}_shuffled_{shuffled_control}_1308.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close('all')

    for dim, params in fit_params.items():
        print(f'Dimension {dim}: A={params[0]}, B={params[1]}, C={params[2]}, D={params[3]}')

    r_squared_df.to_csv(f'{save_dir}/r_squared_values_whole_cumulative_{cumulative_param}_shuffle_{shuffled_control}.csv', index=False)

    return fit_params, r_squared_df
def create_folds(n_timesteps, num_folds=5, num_windows=10):
    n_windows_total = num_folds * num_windows
    window_size = n_timesteps / n_windows_total

    # window_start_ind = np.arange(0, n_windows_total) * window_size
    window_start_ind = np.round(np.arange(0, n_windows_total) * window_size)

    folds = []

    for i in range(num_folds):
        test_windows = np.arange(i, n_windows_total, num_folds)
        test_ind = []
        for j in test_windows:
            test_ind.extend(np.arange(window_start_ind[j], window_start_ind[j] + np.round(window_size)))

        train_ind = list(set(range(n_timesteps)) - set(test_ind))
        # convert test_ind to int
        test_ind = [int(i) for i in test_ind]

        folds.append((train_ind, test_ind))
        # print the ratio
        ratio = len(train_ind) / len(test_ind)
        print(f'Ratio of train to test indices is {ratio}')

    return folds

def fit_sinusoid_data_per_interval(df, save_dir, start_indices, end_indices, cumulative_param = False):
    """
    Fit a sinusoidal function to the data in each interval and plot the results.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing the data to fit.
    save_dir: str
        Directory to save the plot.
    start_indices: list
        List of start indices for each segment.
    end_indices: list
        List of end indices for each segment.
    """
    def sinusoidal(x, A, B, C, D):
        return A * np.sin(B * x + C) + D

    def calculate_goodness_of_fit(x_data, y_data, y_fitted):
        residuals = y_data - y_fitted
        ssr = np.sum(residuals ** 2)
        sst = np.sum((y_data - np.mean(y_data)) ** 2)
        r_squared = 1 - (ssr / sst)
        return r_squared
    r_squared_df = pd.DataFrame(columns=['Interval', 'Dimension', 'R-squared'])
    for idx, (start, end) in enumerate(zip(start_indices, end_indices)):
        segment_df = df[(df['Interval'] >= start) & (df['Interval'] <= end)]
        heatmap_data = segment_df.pivot_table(index='Interval', columns='Dimension', values='death_minus_birth', aggfunc='mean')

        fit_params = {}
        for dim in heatmap_data.columns:
            x_data = heatmap_data.index.values
            y_data = heatmap_data[dim].values
            initial_guess = [1, 1, 0, np.mean(y_data)]
            try:
                params, _ = curve_fit(sinusoidal, x_data, y_data, p0=initial_guess)
            except Exception as e:
                print(f'Failed to fit sinusoidal function for dimension {dim} in interval {start}-{end}, error: {e}')
                continue

            fit_params[dim] = params
            r_squared = calculate_goodness_of_fit(x_data, y_data, sinusoidal(x_data, *params))
            ##append to r_squared_df
            r_squared_df_indiv = pd.DataFrame({'Interval': idx, 'Dimension': dim, 'R-squared': r_squared}, index=[0])
            r_squared_df = pd.concat([r_squared_df, r_squared_df_indiv], ignore_index=True)
            print(f'R-squared for dimension {dim} in interval {start}-{end}: {r_squared}')

            plt.figure(figsize=(10, 6))
            plt.plot(x_data, y_data, 'bo', label='Original Data')
            plt.plot(x_data, sinusoidal(x_data, *params), 'r-', label='Fitted Function')
            plt.title(f'Sinusoidal Fit for Homology Dimension {dim} in Interval {start}-{end}, r2 score: {r_squared:.2f}')
            plt.xlabel('Interval (j)')
            plt.ylabel('Mean Death - Birth')
            plt.legend()
            plt.savefig(f'{save_dir}/sinusoidal_fit_dimension_{dim}_interval_{start}_{end}_cumulative_{cumulative_param}.png', dpi=300, bbox_inches='tight')
            plt.show()
            plt.close('all')

        for dim, params in fit_params.items():
            print(f'Dimension {dim} in interval {start}-{end}: A={params[0]}, B={params[1]}, C={params[2]}, D={params[3]}')
    r_squared_df['mean_r_squared'] = r_squared_df.groupby('Dimension')['R-squared'].transform('mean')

    r_squared_df.to_csv(f'{save_dir}/r_squared_values_cumulative_{cumulative_param}.csv', index=False)
    return fit_params