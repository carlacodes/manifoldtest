import pandas as pd
import os

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
    for i in csv_dict.keys():
        #get the specific dataframe
        df = csv_dict[i]
        #get the number of unique neuron idices
        unique_neuron_indices = df['neuron_index'].unique()
        for i in unique_neuron_indices:
            #get the scores of the current neuron
            current_neuron_scores = df[df['neuron_index'] == i]
            #get the number of scores that are above the mean_perm_score
            scores_above_mean_perm_score = current_neuron_scores[current_neuron_scores['score'] > current_neuron_scores['mean_perm_score']]
            print(f'Neuron {i} has {len(scores_above_mean_perm_score)} scores that are above the mean_perm_score')
            #get the fraction of scores that are above the mean_perm_score
            fraction_above_mean_perm_score = len(scores_above_mean_perm_score)/len(current_neuron_scores)
            print(f'Neuron {i} has {fraction_above_mean_perm_score} fraction of scores that are above the mean_perm_score')
            #add to a separate dataframe
            if i == 0:
                fraction_df = pd.DataFrame({'neuron_index': [i], 'fraction_above_mean_perm_score': [fraction_above_mean_perm_score]})
            else:
                current_fraction_df = pd.DataFrame({'neuron_index': [i], 'fraction_above_mean_perm_score': [fraction_above_mean_perm_score]})
                fraction_df = pd.concat([fraction_df, current_fraction_df], ignore_index=True)
        #add to the big dataframe
        fraction_df['independent_variable'] = i
        big_df_by_independent_variable = pd.concat([big_df_by_independent_variable, fraction_df], ignore_index=True)
    return big_df_by_independent_variable

def main():
    csv_dict = load_csvs('C:/neural_data/rat_7/lstm_csvs/')



















if __name__ == '__main__':
    main()