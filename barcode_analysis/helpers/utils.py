import numpy as np
import numpy as np
from scipy.optimize import curve_fit

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from gph import ripser_parallel
from gtda.homology._utils import _postprocess_diagrams

def visualize_simplex_ripser(dgm_gtda, filtration_radii, save_dir=None):
    """
    Visualize the simplex at varying filtration radius lengths using ripser\_parallel.

    Parameters
    ----------
    points: np.array
        Array of points in the space.
    filtration_radii: list
        List of filtration radii to visualize.
    save_dir: str, optional
        Directory to save the plots.
    """
    for radius in filtration_radii:
        # Compute the Vietoris-Rips persistence for the given points

        # Plot the persistence diagram
        plt.figure(figsize=(8, 6))
        for dim in range(3):
            plt.scatter(dgm_gtda[0][dgm_gtda[0][:, 2] == dim][:, 0], dgm_gtda[0][dgm_gtda[0][:, 2] == dim][:, 1], label=f'H{dim}')
        plt.title(f'Simplex at Filtration Radius {radius}')
        plt.xlabel('Birth')
        plt.ylabel('Death')
        plt.legend()
        plt.tight_layout()

        if save_dir:
            plt.savefig(f'{save_dir}/simplex_filtration_radius_{radius}.png', dpi=300, bbox_inches='tight')

        plt.show()
        plt.close()

# Example usage
points = np.random.rand(10, 2)  # Random points in 2D
filtration_radii = [0.1, 0.2, 0.3, 0.4, 0.5]
visualize_simplex_ripser(points, filtration_radii, save_dir='C:/path_to_save_dir')

def plot_barcode_mosaic(diag, save_dir=None, count=0, **kwargs):
    """
    Plot the barcode for a persistence diagram using matplotlib in a mosaic layout.
    ----------
    diag: np.array: of shape (num_features, 3), i.e. each feature is
           a triplet of (birth, death, dim) as returned by e.g.
           VietorisRipsPersistence
    **kwargs
    Returns
    -------
    None.
    """
    unique_dims = np.unique(diag[:, 2])
    num_plots = len(unique_dims)
    cols = kwargs.get('cols', 3)
    rows = (num_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=kwargs.get('figsize', (15, 10)))
    axes = axes.flatten()
    color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    letter_list = ['a)', 'b)', 'c)', 'd)', 'e', 'f', 'g']
    for i, dim in enumerate(unique_dims):
        ax = axes[i]
        diag_dim = diag[diag[:, 2] == dim]
        birth = diag_dim[:, 0]
        death = diag_dim[:, 1]
        finite_bars = death[death != np.inf]
        if len(finite_bars) > 0:
            inf_end = 2 * max(finite_bars)
        else:
            inf_end = 2
        death[death == np.inf] = inf_end

        if len(np.unique(birth)) > 1:
            sorted_indices = np.argsort(birth)
            birth = birth[sorted_indices]
            death = death[sorted_indices]

        for j, (b, d) in enumerate(zip(birth, death)):
            if d == inf_end:
                ax.plot([b, d], [j, j], color='k', lw=kwargs.get('linewidth', 2))
            else:
                ax.plot([b, d], [j, j], color=kwargs.get('color', color_list[i]), lw=kwargs.get('linewidth', 2))

        ax.set_title(f'Dimension {dim}')
        ax.set_xlabel(kwargs.get('xlabel', 'Filtration Value'))
        ax.set_ylabel(kwargs.get('ylabel', 'Birth'))
        min_birth = np.min(birth)
        min_birth = np.round(min_birth, 3)
        max_birth = np.max(birth)
        max_birth = np.round(max_birth, 3)
        ax.set_yticks([0, len(birth) - 1])
        ax.set_yticklabels([min_birth, max_birth])
        ax.annotate(letter_list[i], xy=(-0.1, 1), xycoords='axes fraction', fontsize=14, ha='center', va='center', fontweight='bold')


    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    rat_id = str(save_dir.split('/')[-4])
    plt.suptitle(kwargs.get('title', 'Persistence barcodes') + f', trial {count}, {rat_id}')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_dir is not None:
        plt.savefig(save_dir + f'/barcode_mosaic_trialid_{count}.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close('all')


def generate_white_noise_control(df, noise_level=1.0, iterations=1000, savedir = ''):
    r_squared_values = []
    for _ in range(iterations):
        noise = np.random.normal(0, noise_level, len(df))
        noise_df = df.copy()
        noise_df['death_minus_birth'] += noise
        _, r_squared_df = fit_sinusoid_data_filtered(noise_df, savedir , threshold=0)
        r_squared_values.append(r_squared_df['R-squared'].mean())
    return r_squared_values


def read_distance_matrix(file_path):
    """
    Read a distance matrix from a file.

    Parameters
    ----------
    file_path: str
        Path to the file containing the distance matrix.

    Returns
    -------
    distance_matrix: np.ndarray
        Distance matrix read from the file.
    """
    #load from pkl file

    distance_matrix = pd.read_pickle(file_path)
    std_dev_distance_list = []
    mean_distance_list = []
    #calculate the mean
    for dim in distance_matrix.keys():
        dim_matrix = distance_matrix[dim]
        #take the upper triangle, remove the nans
        dim_matrix_df = pd.DataFrame(dim_matrix)
        dim_matrix_upper = dim_matrix_df.where(np.triu(np.ones(dim_matrix_df.shape), k=1).astype(bool))
        distance_matrix[dim] = dim_matrix_upper.values#]
        mean_distance = dim_matrix_upper.stack().mean()
        mean_distance_list.append(mean_distance)
        print(f'Mean distance for group {dim}: {mean_distance}')
        #calculate the std dev
        std_dev_distance = dim_matrix_upper.stack().std()
        std_dev_distance_list.append(std_dev_distance)
        print(f'Standard deviation of distance for group {dim}: ', std_dev_distance)
    return mean_distance_list, std_dev_distance_list, distance_matrix

def fit_sinusoid_data_filtered(df, save_dir, cumulative_param=False, trial_number=None, shuffled_control=False, threshold=0):
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
            print('Not enough points, less than 10, risk of overfitting')
            continue

        # Initial guess and bounds
        # initial_guess = [1, 2 * np.pi / np.ptp(unique_x), 0, np.mean(y_data_new)]
        # bounds = ([0.1, 0.01, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf])  # Example bounds
        initial_guess = [np.std(y_data_new), 2 * np.pi / np.ptp(unique_x), 0, np.mean(y_data_new)]
        bounds = ([0, 0, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf])

        try:
            params, _ = curve_fit(sinusoidal, unique_x, y_data_new, p0=initial_guess, bounds=bounds)
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