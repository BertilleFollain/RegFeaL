import numpy as np
from sklearn.model_selection import GridSearchCV
from Regressors.BasicRegFeaL import BasicRegFeaL
from Experiments.Data_generation import data_generation
import pickle


def Experiment2(range_n, range_m, filename, number_experiments=5, seed=35,
                save=False, r=0.33, d=5, s=2, n_test=5000, std_noise=1.5, easy=False):
    """
    Experiment2 studies the dependency of  prediction performance and feature learning performance on the number of
    random features.

    :param range_n: range of number of training data considered
    :param range_m: range of number of random features considered
    :param filename: name of file where results (scores and parameters) are stored in the form of a dictionary
    :param number_experiments: number of times each experiment is run, for mean and standard deviation computation
    :param seed: seed for randomness
    :param save: whether to save the results or not
    :param r: regularisation parameter
    :param d: dimension of data
    :param s: dimension of hidden feature space
    :param n_test: number of test data
    :param std_noise: standard deviation of noise added to training and testing data
    :param easy: if True, the regression function is polynomial in the projected data, else it is a combination of sinus
    """

    # Cross val param
    rhos = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    mus = np.array([100, 1, 0.1, 0.01, 0.001]) * (1 / (d ** ((2 - r) / r)))
    lambs = mus

    # Setting up cross val
    parameters = {'rho': rhos, 'mu': mus}

    # Score storage
    scores = np.zeros((len(range_n), len(range_m), number_experiments))
    scores_feature_space = np.zeros((len(range_n), len(range_m), number_experiments))
    scores_noise = np.zeros((len(range_m), number_experiments))

    for exp in range(number_experiments):
        seed += 1
        X, y, X_test, y_test, p = data_generation(d, np.max(range_n), n_test, s, easy, std_noise, seed, True)
        j = 0
        for little_n in range_n:
            k = 0

            # cross val RegFeaL for largest possible m
            regfeal = BasicRegFeaL(m=np.max(range_m), feature=True, n_iter=3)
            clf = GridSearchCV(regfeal, parameters, n_jobs=-1, pre_dispatch=5)
            clf.fit(X[:little_n, :], y[:little_n])
            mu = clf.best_estimator_.mu
            rho = clf.best_estimator_.rho
            print('RegFeaL ran with selected parameters rho and mu:')
            print(clf.best_estimator_.rho, clf.best_estimator_.mu / (1 / (d ** ((2 - r) / r))))

            for little_m in range_m:
                print('experiment number', exp, 'little_n', little_n, 'little_m', little_m)

                # train RegFeaL with smaller m, but rho and mu from cross val
                regfeal = BasicRegFeaL(m=little_m, rho=rho, mu=mu, feature=True)
                regfeal.fit(X[:little_n, :], y[:little_n])
                scores[j, k, exp] = regfeal.score(X_test, y_test)
                scores_feature_space[j, k, exp] = regfeal.feature_learning_score(p)

                # Best possible score due to noise level
                scores_noise[k, exp] = 1 - n_test * (std_noise ** 2) / ((y_test - y_test.mean()) ** 2).sum()
                k += 1
            j += 1

    results = {'d': d, 's': s, 'n_test': n_test, 'std_noise': std_noise, 'easy': easy,
               'range_m': range_m, 'r': r, 'range_n': range_n, 'number_experiments': number_experiments, 'seed': seed,
               'rhos': rhos, 'mus': mus, 'lambs': lambs,
               'scores': scores, 'scores_feature_space': scores_feature_space,
               'scores_noise': scores_noise}
    if save:
        pickle.dump(results, open(filename, 'wb'))
    print('Experiment2 over')
    return filename
