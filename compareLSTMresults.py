import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

def load_csvs(csv_dir):
    #load the csv files
    #get the list of csv files from the directory
    #get the list of csv files from the directory
    file_list = os.listdir(csv_dir)
    csv_list = {}
    #load the csv files
    for i in range(0, len(file_list)):
        csv_file_to_read = 'C:/neural_data/rat_7\lstm_csvs/' + file_list[i]
        #add to a dictionary
        if 'angle' in file_list[i]:
            csv_list['angle'] = pd.read_csv(csv_file_to_read)
        elif 'xy' in file_list[i]:
            csv_list['xy'] = pd.read_csv(csv_file_to_read)
        elif 'dist2goal' in file_list[i]:
            csv_list['dist2goal'] = pd.read_csv(csv_file_to_read)
    return csv_list



def run_analysis_on_csvs(csv_dict):
    #get the number of neurons that have scores that are above the mean_perm_score
    big_df_by_independent_variable = pd.DataFrame()
    big_df_by_independent_variable_diff = pd.DataFrame()
    for key in csv_dict.keys():
        #get the specific dataframe
        df = csv_dict[key]
        #get the number of unique neuron idices
        unique_neuron_indices = df['neuron_index'].unique()
        for i in unique_neuron_indices:
            #get the scores of the current neuron
            current_neuron_scores = df[df['neuron_index'] == i]
            #get the number of scores that are above the mean_perm_score
            difference_in_scores = current_neuron_scores['score'] - current_neuron_scores['mean_perm_score']
            difference_in_scores = np.array(difference_in_scores)
            neuron_index = np.array([i]*len(difference_in_scores))
            key_index = np.array([key]*len(difference_in_scores))
            scores_above_mean_perm_score = current_neuron_scores[current_neuron_scores['score'] > current_neuron_scores['mean_perm_score']]
            print(f'Neuron {i} has {len(scores_above_mean_perm_score)} scores that are above the mean_perm_score')
            #get the fraction of scores that are above the mean_perm_score
            fraction_above_mean_perm_score = len(scores_above_mean_perm_score)/len(current_neuron_scores)
            print(f'Neuron {i} has {fraction_above_mean_perm_score} fraction of scores that are above the mean_perm_score')
            #add to a separate dataframe
            if i == 0:
                fraction_df = pd.DataFrame({'neuron_index': [i], 'fraction_above_mean_perm_score': [fraction_above_mean_perm_score], 'independent_variable': [key]})
                diff_df = pd.DataFrame({'neuron_index': neuron_index, 'difference_in_scores': difference_in_scores, 'independent_variable': key_index})
            else:
                current_fraction_df = pd.DataFrame({'neuron_index': [i], 'fraction_above_mean_perm_score': [fraction_above_mean_perm_score], 'independent_variable': [key]})
                current_diff_df = pd.DataFrame({'neuron_index': neuron_index, 'difference_in_scores': difference_in_scores, 'independent_variable': key_index})



                fraction_df = pd.concat([fraction_df, current_fraction_df], ignore_index=True)
                diff_df = pd.concat([diff_df, current_diff_df], ignore_index=True)
        #add to the big dataframe
        big_df_by_independent_variable = pd.concat([big_df_by_independent_variable, fraction_df], ignore_index=True)
        big_df_by_independent_variable_diff = pd.concat([big_df_by_independent_variable_diff, diff_df], ignore_index=True)
    return big_df_by_independent_variable, big_df_by_independent_variable_diff

def plot_histograms(big_df_by_independent_variable, rat_id):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    #plot the histograms
    for i in range(0, len(big_df_by_independent_variable['independent_variable'].unique())):
        current_independent_variable = big_df_by_independent_variable['independent_variable'].unique()[i]
        current_df = big_df_by_independent_variable[big_df_by_independent_variable['independent_variable'] == current_independent_variable]
        ax[i].hist(current_df['fraction_above_mean_perm_score'])
        ax[i].set_ylim(0, 40)
        ax[i].set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax[i].set_title(f'{current_independent_variable}')
        ax[i].set_xlabel('Fraction of scores above mean_perm_score')
        ax[i].set_ylabel('Number of neurons')
    plt.suptitle(f'Fraction of scores above mean_perm_score for:  {rat_id}')
    plt.savefig('C:/neural_data/rat_7/lstm_csvs/histograms.png')
    plt.show()

def plot_difference_in_scores(dataframe, rat_id):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    #plot the histograms
    for i in range(0, len(dataframe['independent_variable'].unique())):
        current_independent_variable = dataframe['independent_variable'].unique()[i]
        current_df = dataframe[dataframe['independent_variable'] == current_independent_variable]
        ax[i].hist(current_df['difference_in_scores'])
        ax[i].set_ylim(0, 40)
        ax[i].set_xlim(-1.25, 0.3)
        # ax[i].set_xticks([-0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3])
        ax[i].set_title(f'{current_independent_variable}')
        ax[i].set_xlabel('Difference in scores')
        ax[i].set_ylabel('Number of neurons')
    plt.suptitle(f'Difference in scores between actual and permuted scores for: {rat_id}')
    plt.savefig('C:/neural_data/rat_7/lstm_csvs/histograms_diff.png', dpi = 300, bbox_inches = 'tight')
    plt.show()

    return


def main():

    # csv_dict = load_csvs('C:/neural_data/rat_7/lstm_csvs/')

    #plot the distribution of scores - mean perm score for each neuron
    big_dir = 'C:/myriad_results/jake/'

    for rat in [3, 8, 9, 10, 7]:
        #get the list of folders directory that have dates
        print(f'now starting rat:{rat}')
        dates = os.listdir(os.path.join(big_dir, f'rat_{rat}'))
        #check if the folder name is a date by checking if it contains a hyphen
        date = [d for d in dates if '-' in d][0]
        data_dir = os.path.join(big_dir, f'rat_{rat}', date)
        csv_dir = os.path.join(data_dir, 'csvs_0603')
        #load the list of files in the csv_dir
        csv_dict = load_csvs(csv_dir)
        big_df, diff_df = run_analysis_on_csvs(csv_dict)
        plot_histograms(big_df, rat)
        plot_difference_in_scores(diff_df, rat)


    return





















if __name__ == '__main__':
    main()