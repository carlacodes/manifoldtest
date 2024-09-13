import copy
from sklearn.neighbors import KNeighborsRegressor
from manifold_neural.helpers.datahandling import DataHandler
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from sklearn.manifold import Isomap
from kneed import KneeLocator

def find_elbow_point(x, y):
    kneedle = KneeLocator(x, y, curve='concave', direction='increasing')
    return kneedle.elbow




def process_isomap_results():
    base_dir = 'C:/neural_data/'

    big_componentresult_df = pd.DataFrame()

    for data_dir in [f'{base_dir}/rat_8/15-10-2019', f'{base_dir}/rat_9/10-12-2021', f'{base_dir}/rat_3/25-3-2019', f'{base_dir}/rat_7/6-12-2019', f'{base_dir}/rat_10/23-11-2021',]:
        print(f'Processing {data_dir}')
        rat_id = data_dir.split('/')[-2]
        sub_folder = data_dir + '/plot_results/'
        # get list of files in the directory
        files = os.listdir(sub_folder)
        # check if more than two dirs
        if len(files) >= 2:
            # choose the most recently modified directory
            files.sort(key=lambda x: os.path.getmtime(sub_folder + x))
            # remove files that have 'test' in them
            files = [x for x in files if 'test' in x]
            # get the most recently modified directory
            savedir = sub_folder + files[-1]
        else:
            savedir = sub_folder + files[0]

        #find the csv with isomap_components in the csv
        isomap_df = pd.read_csv(f'{savedir}/isomap_components_evaluation.csv')
        #append to a larger dataframe
        isomap_df['rat_id'] = rat_id
        big_componentresult_df = pd.concat([big_componentresult_df, isomap_df], axis=0)

    #calculate the mean by rat_id
    #group by regress variables
    # Split the big_componentresult_df into separate DataFrames based on the 'regress' variable
    df_xy = big_componentresult_df[big_componentresult_df['regress'].isin(['[\'x\', \'y\']'])]
    df_cos_sin = big_componentresult_df[big_componentresult_df['regress'].isin(['[\'cos_hd\', \'sin_hd\']'])]
    #calculate the mean per component
    numeric_columns = df_xy.select_dtypes(include=[np.number]).columns

    # Perform the groupby operation and calculate the mean for numeric columns
    std_error_xy = df_xy.groupby('n_components')[numeric_columns].std() / np.sqrt(df_xy.groupby('n_components')[numeric_columns].count())
    df_xy_mean = df_xy.groupby('n_components')[numeric_columns].mean()

    numeric_columns = df_cos_sin.select_dtypes(include=[np.number]).columns
    std_error = df_cos_sin.groupby('n_components')[numeric_columns].std() / np.sqrt(df_cos_sin.groupby('n_components')[numeric_columns].count())
    df_cos_sin_mean = df_cos_sin.groupby('n_components')[numeric_columns].mean()


    x_values_cos_sin = df_cos_sin_mean.index.values

    y_values_cos_sin = df_cos_sin_mean['mean_test_score'].values

    x_values_xy = df_xy_mean.index.values
    y_values_xy = df_xy_mean['mean_test_score'].values

    # Find the elbow points
    elbow_point_xy = find_elbow_point(x_values_xy, y_values_xy)
    #find the point where the accuracy is over 80% of the final mean accuracy
    eighty_percent_point = None
    eighty_percent_point_train = None
    eighty_percent_point_train_cos_sin = None
    eighty_percent_point_cos_sin = None
    for i, y in enumerate(y_values_xy):
        if y > 0.8 * y_values_xy[-1]:
            eighty_percent_point = i
            break
    for i, y in enumerate(df_xy_mean['mean_train_score'].values):
        if y > 0.8 * df_xy_mean['mean_train_score'].values[-1]:
            eighty_percent_point_train = i
            break

    for i, y in enumerate(y_values_cos_sin):
        if y > 0.8 * y_values_cos_sin[-1]:
            eighty_percent_point_cos_sin = i
            break

    for i, y in enumerate(df_cos_sin_mean['mean_train_score'].values):
        if y > 0.8 * df_cos_sin_mean['mean_train_score'].values[-1]:
            eighty_percent_point_train_cos_sin = i
            break

    print(f'Eighty percent point for test cos_sin: {eighty_percent_point_cos_sin}')
    print(f'Eighty percent point for train cos_sin: {eighty_percent_point_train_cos_sin}')


    print(f'Eighty percent point for test xy: {eighty_percent_point}')
    print(f'Eighty percent point for train xy: {eighty_percent_point_train}')

    elbow_point_xy_train = find_elbow_point(x_values_xy, df_xy_mean['mean_train_score'].values)
    elbow_point_cos_sin = find_elbow_point(x_values_cos_sin, y_values_cos_sin)
    elbow_point_cos_sin_train = find_elbow_point(x_values_cos_sin, df_cos_sin_mean['mean_train_score'].values)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    #don't plot the n_components
    df_xy_mean.drop(columns='n_components', inplace=True)
    df_cos_sin_mean.drop(columns='n_components', inplace=True)
    #plot the error bars
    df_xy_mean.plot(ax=ax[0], yerr=std_error_xy, linestyle='None', marker='o')
    df_cos_sin_mean.plot(ax=ax[1], yerr=std_error, linestyle='None', marker='o')
    #find the elbow point
    ax[0].set_title('x, y')
    ax[0].set_ylim(0, 0.6)
    ax[1].set_title('cos(theta), sin(theta)')
    ax[0].set_xlabel('n_components')
    ax[0].set_ylabel('mean_r2_score')
    ax[1].set_ylim(0, 0.6)
    #remove the legend from the first suplot
    ax[0].get_legend().remove()
    plt.savefig(f'{base_dir}/isomap_component_results_plot.png', dpi=300, bbox_inches='tight')
    plt.show()



    big_componentresult_df_mean = big_componentresult_df.groupby('rat_id').mean()




if __name__ == '__main__':
    #
    process_isomap_results()






