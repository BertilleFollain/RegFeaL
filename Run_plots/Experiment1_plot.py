import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import numpy as np


def Experiment1_plot(filename, folder, save):
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette('colorblind')

    # Opening results
    dict_res = pickle.load(open(filename, 'rb'))

    # Figure storage
    if 'polynomial' in filename:
        fig1 = folder + 'Experiment1_feature' + str(dict_res['feature']) + '_d' + str(
            dict_res['d']) + 'polynomial_fig1.png'
        fig2 = folder + 'Experiment1_feature' + str(dict_res['feature']) + '_d' + str(
            dict_res['d']) + 'polynomial_fig2.png'
        fig3 = folder + 'Experiment1_feature' + str(dict_res['feature']) + '_d' + str(
            dict_res['d']) + 'polynomial_fig3.png'

    else:
        fig1 = folder + 'Experiment1_feature' + str(dict_res['feature']) + '_d' + str(dict_res['d']) + '_fig1.png'
        fig2 = folder + 'Experiment1_feature' + str(dict_res['feature']) + '_d' + str(dict_res['d']) + '_fig2.png'
        fig3 = folder + 'Experiment1_feature' + str(dict_res['feature']) + '_d' + str(dict_res['d']) + '_fig3.png'

    # Averaging over experiments
    mean_scores = np.mean(dict_res['scores'], axis=1)
    std_scores = np.std(dict_res['scores'], axis=1) / np.sqrt(dict_res['number_experiments'])
    mean_scores_dim = np.mean(dict_res['scores_dim'], axis=1)
    std_scores_dim = np.std(dict_res['scores_dim'], axis=1) / np.sqrt(dict_res['number_experiments'])
    mean_scores_mave = np.mean(dict_res['scores_mave'], axis=1)
    std_scores_mave = np.std(dict_res['scores_mave'], axis=1) / np.sqrt(dict_res['number_experiments'])
    mean_scores_dim_mave = np.mean(dict_res['scores_dim_mave'], axis=1)
    std_scores_dim_mave = np.std(dict_res['scores_dim_mave'], axis=1) / np.sqrt(dict_res['number_experiments'])
    mean_scores_ridge = np.mean(dict_res['scores_ridge'], axis=1)
    std_scores_ridge = np.std(dict_res['scores_ridge'], axis=1) / np.sqrt(dict_res['number_experiments'])
    mean_scores_noise = np.mean(dict_res['scores_noise'], axis=1)
    std_score_noise = np.std(dict_res['scores_noise'], axis=1) / np.sqrt(dict_res['number_experiments'])
    mean_scores_feature_space = np.mean(dict_res['scores_feature_space'], axis=1)
    std_scores_feature_space = np.std(dict_res['scores_feature_space'], axis=1) / np.sqrt(
        dict_res['number_experiments'])
    mean_scores_feature_space_mave = np.mean(dict_res['scores_feature_space_mave'], axis=1)
    std_scores_feature_space_mave = np.std(dict_res['scores_feature_space_mave'], axis=1) / np.sqrt(
        dict_res['number_experiments'])
    # print(dict_res['run_times'])

    markers = ['o', 's', 'd', 'v']

    # Plot prediction scores
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=2, style='whitegrid')
    plt.errorbar(dict_res['range_n'], mean_scores_mave, yerr=std_scores_mave, label='MAVE', color=palette[1],
                 linewidth=1.5,
                 marker=markers[1], markersize=8)
    plt.errorbar(dict_res['range_n'], mean_scores_ridge, yerr=std_scores_ridge, label='Kernel Ridge',
                 color=palette[4], linewidth=1.5,
                 marker=markers[2], markersize=8)
    plt.axhline(0, color='black')
    plt.axhline(1, color='black')
    plt.errorbar(dict_res['range_n'], mean_scores_noise, yerr=std_score_noise,
                 label='Noise Level', color=palette[3], linewidth=1.5,
                 marker=markers[3], markersize=8)
    plt.errorbar(dict_res['range_n'], mean_scores, yerr=std_scores, label='RegFeaL', color=palette[0], linewidth=1.5,
                 marker=markers[0], markersize=8)
    # plt.title('Prediction performance: d=' + str(dict_res['d']) + ', s=' + str(dict_res['s']))
    plt.xlabel('Number of samples n')
    plt.ylabel('R2 score')
    if 'polynomial' in filename:
        plt.ylim(-0.6, 1.1)  # todo change that
        plt.legend()
    else:     plt.legend(loc='lower right')
    plt.subplots_adjust(bottom=0.15, left=0.16)
    if save:
        plt.savefig(fig1, dpi=100)
    plt.show()
    print('Figure 1 over')

    # Plot feature learning scores
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=2, style='whitegrid')
    plt.errorbar(dict_res['range_n'], mean_scores_feature_space_mave, yerr=std_scores_feature_space_mave, label='MAVE',
                 color=palette[1], linewidth=1.5, marker=markers[1], markersize=8)
    plt.errorbar(dict_res['range_n'], mean_scores_feature_space, yerr=std_scores_feature_space, label='RegFeaL',
                 color=palette[0], linewidth=1.5, marker=markers[0], markersize=8)
    plt.axhline(0, color='black')
    plt.axhline(1, color='black')
    # plt.title('Feature learning performance: d=' + str(dict_res['d']) + ', s=' + str(dict_res['s']))
    plt.xlabel('Number of samples n')
    plt.ylabel('Feature learning score')
    plt.legend()
    plt.subplots_adjust(bottom=0.15, left=0.16)
    if save:
        plt.savefig(fig2, dpi=100)
    plt.show()
    print('Figure 2 over')

    # Plot dimension learning scores
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=2, style='whitegrid')
    plt.errorbar(dict_res['range_n'], mean_scores_dim_mave, yerr=std_scores_dim_mave, label='MAVE',
                 color=palette[1], linewidth=1.5, marker=markers[1], markersize=8)
    plt.errorbar(dict_res['range_n'], mean_scores_dim, yerr=std_scores_dim, label='RegFeaL',
                 color=palette[0], linewidth=1.5, marker=markers[0], markersize=8)
    plt.axhline(0, color='black')
    plt.axhline(1, color='black')
    # plt.title('Dimension learning performance: d=' + str(dict_res['d']) + ', s=' + str(dict_res['s']))
    plt.xlabel('Number of samples n')
    plt.ylabel('Dimension score')
    plt.legend()
    plt.subplots_adjust(bottom=0.15, left=0.16)
    if save:
        plt.savefig(fig3, dpi=100)
    plt.show()
    print('Figure 3 over')
