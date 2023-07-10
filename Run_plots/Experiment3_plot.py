import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import numpy as np


def Experiment3_plot(filename, folder, save):
    palette = sns.color_palette('colorblind')

    # Opening results
    dict_res = pickle.load(open(filename, 'rb'))
    # Figure storage
    fig1 = folder + 'Experiment3_d' + str(dict_res['d']) + '_fig1.png'
    fig2 = folder + 'Experiment3_d' + str(dict_res['d']) + '_fig2.png'
    fig3 = folder + 'Experiment3_d' + str(dict_res['d']) + '_fig3.png'
    fig4 = folder + 'Experiment3_d' + str(dict_res['d']) + '_fig4.png'

    # Averaging over experiments
    scores_train = dict_res['scores_train']
    scores_test = dict_res['scores_test']
    scores_noise = dict_res['scores_noise']
    scores_feature_space = dict_res['scores_feature_space']
    etas = dict_res['etas']
    scaled_etas = etas ** (dict_res['r'] / (2 - dict_res['r']))
    alphas = dict_res['alphas']
    range_iter = []
    for i in range(dict_res['n_iter']):
        # In the following assignment statement every value in the datax
        # list will be set as a string, this solves the floating point issue
        range_iter += [str(1 + i)]

    # Plot prediction scores
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=2, style='whitegrid')
    plt.plot(range_iter, scores_test, label='Test set', color=palette[0], linewidth=1.5, marker='o', markersize=8)
    plt.plot(range_iter, scores_train, label='Train set', color=palette[1], linewidth=1.5, marker='s', markersize=8)
    plt.axhline(0, color='black')
    plt.axhline(1, color='black')
    # plt.plot(range_iter, scores_noise,
    #         label='Best possible score (Noise Level)', color=palette[3], linewidth=2, marker='o')
    # plt.title('Prediction performance: d=' + str(dict_res['d']) + ', s=' + str(dict_res['s']))
    plt.xlabel('Iteration')
    plt.ylabel('R2 score')
    plt.legend()
    plt.subplots_adjust(bottom=0.15, left=0.16)
    if save:
        plt.savefig(fig1, dpi=100)
    plt.show()
    print('Figure 1 over')

    # Plot feature learning scores
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.5, style='whitegrid')
    plt.plot(range_iter, scores_feature_space, color=palette[0], linewidth=1.5, marker='o')
    plt.axhline(0, color='black')
    plt.axhline(1, color='black')
    # plt.title('Feature learning performance: d=' + str(dict_res['d']) + ', s=' + str(dict_res['s']))
    plt.xlabel('Iteration')
    plt.ylabel('Feature learning score')
    plt.subplots_adjust(bottom=0.15, left=0.16)
    if save:
        plt.savefig(fig2, dpi=100)
    plt.show()
    print('Figure 2 over')

    # Plot scaled etas
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.5, style='whitegrid')
    plt.plot(range_iter, scaled_etas, linewidth=1.5, label=np.arange(dict_res['d']),
             marker='o')
    # plt.title('Feature importance: d=' + str(dict_res['d']) + ', s=' + str(dict_res['s']))
    plt.xlabel('Iteration')
    plt.ylabel(r"$\eta_a^{r/(2-r)}$")
    plt.subplots_adjust(bottom=0.15, left=0.16)
    if save:
        plt.savefig(fig3, dpi=100)
    plt.show()
    print('Figure 3 over')

    # Plot histograms alphas
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.5, style='whitegrid')
    alphas_bis = np.zeros((4, dict_res['m']))
    alphas_bis[0, :] = alphas[dict_res['n_iter'] - 1, :, 0]
    alphas_bis[1, :] = alphas[dict_res['n_iter'] - 1, :, dict_res['d'] - 1]
    alphas_bis[2, :] = alphas[0, :, 0]
    alphas_bis[3, :] = alphas[0, :, dict_res['d'] - 1]
    bins = np.arange(np.max(alphas_bis) + 1) - 0.5 * np.ones(int(np.max(alphas_bis)) + 1)
    plt.hist(alphas_bis.T, density=True, bins=bins, log=True,
             label=[r'$\eta_a$' + ' small, last iter', r'$\eta_a$' + ' large, last iter',
                    r'$\eta_a$' + ' small, first iter', r'$\eta_a$' + ' large, first iter'])
    # plt.title('Empirical density of ' + r"$\alpha$" + ' : d=' + str(dict_res['d']) + ', s=' + str(
    #   dict_res['s']))
    plt.xlabel('Value of ' + r'$\alpha_a$')
    plt.ylabel('Density')
    # plt.xticks(np.arange(np.max(alphas_bis) + 1), np.arange(np.max(alphas_bis) + 1, dtype=int))
    plt.legend()
    plt.subplots_adjust(bottom=0.15, left=0.16)
    if save:
        plt.savefig(fig4, dpi=100)
    plt.show()
    print('Figure 4 over')
