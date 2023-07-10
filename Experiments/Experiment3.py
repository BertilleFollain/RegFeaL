import numpy as np
from sklearn.model_selection import GridSearchCV
from Regressors.BasicRegFeaL import BasicRegFeaL
from Experiments.Data_generation import data_generation
import pickle


def Experiment3(filename, seed=35, n=200, m=5000, n_iter=5,
                save=False, r=0.33, d=5, s=2, n_test=5000, std_noise=1.5, easy=False):
    """
    Experiment3 studies the training behavior of RegFeaL.

    :param filename: name of file where results (scores and parameters) are stored in the form of a dictionary
    :param n: number of training data
    :param m: number of random features to sample
    :param n_iter: number of iterations for the optimisation of the method
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
    degrees = [1, 2, 3, 4, 6, 8, 10]

    # Setting up cross val
    parameters = {'rho': rhos, 'mu': mus}

    # Score storage
    scores_train = np.zeros(n_iter)
    scores_test = np.zeros(n_iter)
    scores_feature_space = np.zeros(n_iter)
    scores_noise = np.zeros(n_iter)
    etas = np.zeros((n_iter, d))
    alphas = np.zeros((n_iter, m, d))

    X, y, X_test, y_test, p = data_generation(d, n, n_test, s, easy, std_noise, seed, True)

    # cross val RegFeaL
    regfeal = BasicRegFeaL(m=m, feature=True)
    clf = GridSearchCV(regfeal, parameters, n_jobs=-1)
    clf.fit(X, y)
    mu = clf.best_estimator_.mu
    rho = clf.best_estimator_.rho
    print('RegFeaL ran with selected parameters rho and mu:')
    print(clf.best_estimator_.rho, clf.best_estimator_.mu / (1 / (d ** ((2 - r) / r))))

    for i in range(n_iter):
        print('iter', i)

        # train RegFeaL with smaller m, but rho and mu from cross val
        regfeal = BasicRegFeaL(m=m, rho=rho, mu=mu, feature=True, n_iter=i + 1)
        regfeal.fit(X, y)
        scores_test[i] = regfeal.score(X_test, y_test)
        scores_train[i] = regfeal.score(X, y)
        scores_feature_space[i] = regfeal.feature_learning_score(p)
        etas[i, :] = regfeal.eta_
        alphas[i, :] = regfeal.uncollapsed_alphas_

        # Best possible score due to noise level
        scores_noise[i] = 1 - n_test * (std_noise ** 2) / ((y_test - y_test.mean()) ** 2).sum()

    results = {'d': d, 's': s, 'n_test': n_test, 'std_noise': std_noise, 'easy': easy, 'n': n,
               'n_iter': n_iter, 'seed': seed, 'r': r, 'm': m,
               'rhos': rhos, 'mus': mus, 'lambs': lambs, 'degrees': degrees,
               'scores_test': scores_test, 'scores_train': scores_train, 'scores_feature_space': scores_feature_space,
               'scores_noise': scores_noise, 'etas': etas, 'alphas': alphas}
    if save:
        pickle.dump(results, open(filename, 'wb'))
    print('Experiment3 over')
    return filename
