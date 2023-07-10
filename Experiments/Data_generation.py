import numpy as np
import scipy.stats


def data_generation(d, n, n_test, s, easy, std_noise, seed, feature):
    """
    data_generation generates training and testing data for regression with a hidden linear feature space, but a
    non-linear response/cofactors dependency, which is more non-linear when easy is False.

    :param d: dimension of data
    :param n: number of training data
    :param n_test: number of test data
    :param s: dimension of hidden feature space
    :param easy: if True, the regression function is polynomial in the projected data, else it is a combination of sinus
    :param std_noise: standard deviation of noise added to training and testing data
    :param seed: seed for randomness
    :param feature: whether to use a random linear feature space or a random set of variables
    """

    np.random.seed(seed)

    # generate cofactors, uniform on the hypercube with unit variance
    X = np.sqrt(3) * (2 * np.random.uniform(size=(n, d)) - 1)
    X_test = np.sqrt(3) * (2 * np.random.uniform(size=(n_test, d)) - 1)

    if feature:
        # generate projection matrix of dim s
        p = scipy.stats.ortho_group.rvs(d)
        p = p[:, 0:s]
    else:
        p = np.identity(d)[:, 0:s]

    # generate response, which only depends on the first s coordinates of the projected data
    if easy:
        y = np.sum(np.dot(X, p), axis=1) - np.sum(np.dot(X, p) ** 2, axis=1) \
            + 2 * np.dot(X, p)[:, 0] * (np.dot(X, p)[:, 1] ** 3) - 4 + std_noise * np.random.normal(0, 1, n)
        y_test = np.sum(np.dot(X_test, p), axis=1) - np.sum(np.dot(X_test, p) ** 2,
                                                            axis=1) + 2 * np.dot(X_test, p)[:, 0] * (
                         np.dot(X_test, p)[:, 1] ** 3) - 4 + std_noise * np.random.normal(0, 1, n_test)
    else:
        y = np.sum(np.sin(2 * np.dot(X, p)), axis=1) + std_noise * np.random.normal(0, 1, n)
        y_test = np.sum(np.sin(2 * np.dot(X_test, p)), axis=1) + std_noise * np.random.normal(0, 1, n_test)
    return X, y, X_test, y_test, p
