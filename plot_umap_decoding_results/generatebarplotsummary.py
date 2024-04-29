import copy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

def main():
    big_df_savedir = 'C:/neural_data/r2_decoding_figures/umap'
    #read all the csv files in the directory
    all_files = glob.glob(big_df_savedir + "/*.csv")
    df_across_var = pd.DataFrame()

    for var in ['dist2goal', 'anglesincos_allocentric', 'sincosrelativetogoal', 'x_y_zscore']:
        #load the data
        var_df = pd.read_csv('C:/neural_data/r2_decoding_figures/umap/umap_decomposition_results_{}.csv'.format(var))
        #add the variable name
        if var == 'x_y_zscore':
            var_text = 'xy_position'
        elif var == 'sincosrelativetogoal':
            var_text = 'angle_relative_to_goal'
        elif var == 'anglesincos_allocentric':
            var_text = 'angle_allocentric'
        elif var == 'dist2goal':
            var_text = 'distance_to_goal'
        var_df['variable'] = var_text
        #remove best_params
        var_df = var_df.drop(columns=['best_params'])
        var_df = var_df.drop(columns = ['Unnamed: 0'])
        #calculate the standard deviation
        var_df['std_score_max'] = np.std(var_df['mean_score_max'])
        var_df['std_score_max_train'] = np.std(var_df['mean_score_max_train'])
        var_df['std_score_max_shuffled'] = np.std(var_df['mean_score_max_shuffled'])
        var_df['std_score_max_train_shuffled'] = np.std(var_df['mean_score_max_train_shuffled'])


        #append to the big dataframe
        df_across_var = pd.concat([df_across_var, var_df])
    #create an individual bar plot for each rat_id
    for rat_id in df_across_var['rat_id'].unique():
        fig, ax = plt.subplots()
        df_across_var_rat = df_across_var[df_across_var['rat_id'] == rat_id]
        df_across_var_rat = df_across_var_rat.drop(columns=['rat_id'])
        df_across_var_rat = df_across_var_rat.set_index('variable')
        #plot the data
        # std_devs = df_across_var_rat[['std_score_max', 'std_score_max_train', 'std_score_max_shuffled', 'std_score_max_train_shuffled']].values.T
        df_across_var_rat[['mean_score_max', 'mean_score_max_train', 'mean_score_max_shuffled', 'mean_score_max_train_shuffled']].plot(kind='bar', ax=ax,  capsize=5)
        ax.set_ylabel('r2')
        ax.set_xlabel('Variable')
        #rotate the xticks by 45 degrees
        plt.xticks(rotation=45)
        plt.title('R2 decoding results for rat {}'.format(rat_id))
        plt.savefig('C:/neural_data/r2_decoding_figures/umap/umap_decoding_results_summary_rat_{}.png'.format(rat_id), dpi=300, bbox_inches='tight')
        plt.show()

    #now take the mean scores for each variable


    df_across_var_mean = copy.deepcopy(df_across_var)
    df_across_var_mean = df_across_var_mean.rename(columns={'mean_score_max': 'test R2', 'mean_score_max_train': 'train R2', 'mean_score_max_shuffled': 'shuffled test R2', 'mean_score_max_train_shuffled': 'shuffled train R2'})

    df_across_var_mean = df_across_var_mean.drop(columns=['rat_id'])
    df_across_var_mean = df_across_var_mean.groupby('variable').mean()
    #rank the variables
    df_across_var_mean = df_across_var_mean.sort_values(by='test R2', ascending=False)
    #plot the data
    fig, ax = plt.subplots()
    # df_across_var_mean.plot(kind='bar', ax=ax, capsize=5)
    std_devs = df_across_var_mean[
        ['std_score_max', 'std_score_max_train', 'std_score_max_shuffled', 'std_score_max_train_shuffled']].values.T

    df_across_var_mean[['test R2', 'train R2', 'shuffled test R2', 'shuffled train R2']].plot(kind='bar', ax=ax,
                                                                                              yerr=std_devs, capsize=5)

    #plot the standard deviation

    ax.set_ylabel('r2')
    ax.set_xlabel('Variable')
    #rotate the xticks by 45 degrees
    plt.xticks(rotation=45)
    plt.title('Mean R2 across rats ')
    plt.savefig('C:/neural_data/r2_decoding_figures/umap/umap_decoding_results_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    fig.close('all')
    return




if __name__ == '__main__':
    #
    main()