import numpy as np
from sklearn.model_selection import GridSearchCV
from Regressors.RegFeaL import RegFeaL
from Regressors.PyMave import PyMave
from Experiments.Data_generation import data_generation
from Regressors.BasicRegFeaL import BasicRegFeaL
import pickle
import time


def Experiment1(range_n, filename, number_experiments=5, seed=35, save=False, m=5000, r=0.33, d=5, s=2, n_test=5000,
                std_noise=1.5, easy=False, feature=True):
    """
    Experiment1 studies the dependency of  prediction performance and feature learning performance on the number of
    training data.

    :param range_n: range of number of training data considered
    :param filename: name of file where results (scores and parameters) are stored in the form of a dictionary
    :param number_experiments: number of times each experiment is run, for mean and standard deviation computation
    :param seed: seed for randomness
    :param save: whether to save the results or not
    :param m: number of random features
    :param r: regularisation parameter
    :param d: dimension of data
    :param s: dimension of hidden feature space
    :param n_test: number of test data
    :param std_noise: standard deviation of noise added to training and testing data
    :param easy: if True, the regression function is polynomial in the projected data, else it is a combination of sinus
    :param feature: whether to use RegFeaL (True) of the variable selection version (False)
    """

    # Cross val param
    rhos = np.array([0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8])
    mus = np.array([1000, 100, 10, 1, 0.1, 0.01, 0.001]) * (1 / (d ** ((2 - r) / r)))
    lambs = mus

    # Setting up cross val
    parameters = {'rho': rhos, 'mu': mus}
    ridge_parameters = {'rho': rhos, 'lamb': lambs}

    # Score storage
    run_times = np.zeros((2, len(range_n), number_experiments))
    scores = np.zeros((len(range_n), number_experiments))
    scores_dim = np.zeros((len(range_n), number_experiments))
    scores_ridge = np.zeros((len(range_n), number_experiments))
    scores_feature_space = np.zeros((len(range_n), number_experiments))
    scores_mave = np.zeros((len(range_n), number_experiments))
    scores_feature_space_mave = np.zeros((len(range_n), number_experiments))
    scores_dim_mave = np.zeros((len(range_n), number_experiments))
    scores_noise = np.zeros((len(range_n), number_experiments))

    for exp in range(number_experiments):
        seed += 1
        X, y, X_test, y_test, p = data_generation(d, np.max(range_n), n_test, s, easy, std_noise, seed, feature)
        j = 0
        for little_n in range_n:
            print('exp number', exp, 'little_n', little_n)

            # PyMave training and scoring
            start = time.time()
            pymave = PyMave()
            pymave.fit(X[:little_n, :], y[:little_n])
            scores_dim_mave[j, exp] = pymave.dimension_score(s)
            scores_mave[j, exp] = pymave.score(X_test, y_test)
            scores_feature_space_mave[j, exp] = pymave.feature_learning_score(p)
            end = time.time()
            run_times[0, j, exp] = end - start
            print('Mave ran')

            # RegFeal with feature learning training and scoring
            start = time.time()
            regfeal = RegFeaL(m=m, feature=feature, r=r)
            clf1 = GridSearchCV(regfeal, parameters, n_jobs=-1, pre_dispatch=8)
            clf1.fit(X[:little_n, :], y[:little_n])
            scores_dim[j, exp] = clf1.best_estimator_.dimension_score(s)
            scores[j, exp] = clf1.best_estimator_.score(X_test, y_test)
            scores_feature_space[j, exp] = clf1.best_estimator_.feature_learning_score(p)
            end = time.time()
            run_times[1, j, exp] = end - start
            print('RegFeaL ran with selected parameters rho and mu:')
            print(clf1.best_estimator_.rho, clf1.best_estimator_.mu / (1 / (d ** ((2 - r) / r))))

            # Kernel ridge with custom kernel training and scoring
            ridge = BasicRegFeaL(m=m, feature=False, n_iter=1, mu=0.0)
            clf2 = GridSearchCV(ridge, ridge_parameters, n_jobs=-1, pre_dispatch=8)
            clf2.fit(X[:little_n, :], y[:little_n])
            scores_ridge[j, exp] = clf2.best_estimator_.score(X_test, y_test)
            print('Ridge ran with selected parameters rho and lambda:')
            print(clf2.best_estimator_.rho, clf2.best_estimator_.lamb_ / (1 / (d ** ((2 - r) / r))))

            # Best possible score due to noise level
            scores_noise[j, exp] = 1 - n_test * (std_noise ** 2) / ((y_test - y_test.mean()) ** 2).sum()

            j += 1

    results = {'d': d, 's': s, 'n_test': n_test, 'std_noise': std_noise, 'easy': easy,
               'm': m, 'r': r, 'range_n': range_n, 'number_experiments': number_experiments, 'seed': seed,
               'rhos': rhos, 'mus': mus, 'lambs': lambs, 'scores_dim': scores_dim, 'scores_dim_mave': scores_dim_mave,
               'scores_mave': scores_mave, 'scores_feature_space_mave': scores_feature_space_mave,
               'scores': scores, 'scores_feature_space': scores_feature_space, 'feature': feature,
               'scores_ridge': scores_ridge, 'scores_noise': scores_noise, 'run_times': run_times}
    if save:
        pickle.dump(results, open(filename, 'wb'))
    print('Experiment1 over')
    return filename
