import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import numpy as np


def Experiment2_plot(filename, folder, save):
    palette = sns.color_palette('colorblind')

    # Opening results
    dict_res = pickle.load(open(filename, 'rb'))

    # Figure storage
    fig1 = folder + 'Experiment2_d' + str(dict_res['d']) + '_fig1.png'
    fig2 = folder + '/Experiment2_d' + str(dict_res['d']) + '_fig2.png'

    # Averaging over experiments
    mean_scores = np.mean(dict_res['scores'], axis=2)
    std_scores = np.std(dict_res['scores'], axis=2) / np.sqrt(dict_res['number_experiments'])
    mean_scores_feature_space = np.mean(dict_res['scores_feature_space'], axis=2)
    std_scores_feature_space = np.std(dict_res['scores_feature_space'], axis=2) / np.sqrt(
        dict_res['number_experiments'])
    mean_scores_noise = np.mean(dict_res['scores_noise'], axis=1)
    std_score_noise = np.std(dict_res['scores_noise'], axis=1) / np.sqrt(dict_res['number_experiments'])

    plt.figure(figsize=(8, 6))
    sns.set(font_scale=2, style='whitegrid')
    color = 0
    i = 0
    markers = ['o', 's', 'd']
    for little_n in dict_res['range_n']:
        plt.errorbar(dict_res['range_m'], mean_scores[i, :], yerr=std_scores[i, :], label='n=' + str(little_n),
                     color=palette[color],
                     linewidth=1.5, marker=markers[i], markersize=8)
        color += 1
        i += 1

    plt.axhline(0, color='black')
    plt.axhline(1, color='black')
    # plt.errorbar(dict_res['range_m'], mean_scores_noise, yerr=std_score_noise,
    #            label='Best possible score (Noise Level)', color=palette[3], linewidth=2, marker='o')
    # plt.title('', fontsize=0)
    plt.xlabel('Number of random features m')
    plt.ylabel('R2 score')
    plt.legend()
    plt.subplots_adjust(bottom=0.15, left=0.16)
    if save:
        plt.savefig(fig1, dpi=100)
    plt.show()
    print('Figure 1 over')

    plt.figure(figsize=(8, 6))
    sns.set(font_scale=2, style='whitegrid')
    color = 0
    i = 0
    for little_n in dict_res['range_n']:
        plt.errorbar(dict_res['range_m'], mean_scores_feature_space[i, :], yerr=std_scores_feature_space[i, :],
                     label='n=' + str(little_n),
                     color=palette[color],
                     linewidth=1.5, marker=markers[i], markersize=8)
        color += 1
        i += 1
    plt.axhline(0, color='black')
    plt.axhline(1, color='black')
    # plt.title('Feature learning performance: d=' + str(dict_res['d']) + ', s=' + str(dict_res['s']))
    plt.xlabel('Number of random features m')
    plt.ylabel('Feature learning score')
    plt.legend()
    plt.subplots_adjust(bottom=0.15, left=0.16)
    if save:
        plt.savefig(fig2, dpi=100)
    plt.show()
    print('Figure 2 over')
