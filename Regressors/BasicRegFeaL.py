from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from scipy.special import comb
from sklearn.utils import check_random_state
import numpy as np
import numba


class BasicRegFeaL(RegressorMixin, BaseEstimator):
    """
    RegFeal implements the regressor proposed in Follain, B., Simsekli, U., and Bach F. (2023), Nonparametric Linear Feature Learning
    in Regression Through Regularisation and integrates it into the Scikit-learn framework.
    """

    def __init__(self, rho=0.5, mu=0.1, m=500, feature=False, epsilon=1e-8, r=0.33, n_iter=5, lamb=None):
        """"
        :param rho: # parameter to define the penalty, advice is to crossvalidate it
        :param mu: regularisation parameter, advice is to crossvalidate it
        :param m: number of random features used to approximate the kernel
        :param feature: whether to use the feature learning version (True) or the variable section version (False)
        :param epsilon: computational stability parameter
        :param r: parameter to change the penalty to a convex or concave one
        :param n_iter: number of optimisation iterations during training
        :param lamb: regularisation parameter, if not specified a default value is used
        """
        self.rho = rho
        self.mu = mu
        self.m = m
        self.feature = feature
        self.epsilon = epsilon
        self.r = r
        self.n_iter = n_iter
        self.lamb = lamb

    def fit(self, X, y, random_state=0):
        if y is None:
            raise ValueError('requires y to be passed, but the target y is None')
        X, y = check_X_y(X, y, y_numeric=True)
        self.X_ = X
        self.y_ = y
        self.n_features_in_ = self.X_.shape[1]
        self.n_ = self.X_.shape[0]
        if self.n_ == 1:
            raise ValueError("1 sample")
        if self.n_features_in_ == 1:
            raise ValueError("n_features = 1")

        if self.lamb is None:
            self.lamb_ = 1e-8 / self.n_features_in_ ** ((2 - self.r) / self.r)
        else:
            self.lamb_ = self.lamb

        random_state = check_random_state(random_state)

        for i in range(self.n_iter):
            # initialize eta (lambda) or obtain new eta (lambda) from fixed function represented by theta
            if i == 0:
                self.eta_ = np.ones(self.n_features_in_) / (self.n_features_in_ ** ((2 - self.r) / self.r))
                self.U_ = np.identity(self.n_features_in_)
            else:
                if self.feature:
                    self.eta_, self.U_ = self.update_lambda(self.n_features_in_, self.r, self.theta_,
                                                            self.weight_feature_, self.alphas_, self.m_prime_, self.U_,
                                                            self.epsilon)
                else:
                    self.update_eta()

            # update approximation of kernel by sampling new features
            self.update_features(i, random_state)

            # obtain new function represented by theta from fixed eta (or lambda)
            self.update_theta()
        self.select_dim()
        self.is_fitted_ = True
        return self

    def update_features(self, i, random_state):
        """
        update_features approximates the kernel at step i by using importance sampling and then computes the new
        features for this kernel approximation on the training set.
        """
        x = np.dot(self.X_, self.U_)
        if i == 0:
            self.initial_exact_sampling(random_state)
        else:
            self.group_sampling(random_state)
        self.phi_ = self.hermite_polynomials_features(x, self.alphas_)
        return self

    def initial_exact_sampling(self, random_state):
        """
        initial_exact_sampling samples tuples to approximate the kernel and is used during the initial iteration of the
        training. It is almost exact as the only difference from the theory is that the value of max_k is bounded for
        computational purposes.
        """
        max_k = 40
        probabilities = np.zeros(max_k)
        values = np.arange(1, max_k + 1)

        # filing the array of probabilities that the sampled alpha has sum k+1
        for k in range(0, max_k):
            probabilities[k] = comb(k + 1 + self.n_features_in_ - 1, self.n_features_in_ - 1) * (
                    self.rho ** (k + 1)) / (self.lamb_ + self.mu * (k + 1) / self.eta_[0])

        # sampling the np.sum(tuple) according to the previously defined distribution
        k = random_state.choice(values, self.m, True, probabilities / np.sum(probabilities))

        # computes the constance for importance sampling
        c_eta = np.sum(probabilities)

        # samples the alpha uniformly over those summing to k
        alphas = self.uniform_tuple_sampling(k, self.n_features_in_, random_state)
        self.uncollapsed_alphas_ = alphas
        self.alphas_, weight_sample = np.unique(alphas, axis=0, return_counts=True)  # collapses the non-unique alphas
        self.m_prime_ = len(self.alphas_)  # number of unique alphas
        self.weight_feature_ = np.diag(
            np.sqrt(weight_sample / self.m * c_eta))  # collapses equal alphas, prints statistics
        return self

    def uniform_tuple_sampling(self, k_vals, d, random_state):
        """
        uniform _tuple_sampling samples d dimensional tuples which sum to the value in k_vals uniformly over the set of
        all such tuples
        """
        # if the dimension is one, return k_vals; because there are no other tuples fulfilling the constraint
        if d == 1:
            return k_vals.reshape((-1, 1))

        temp = np.zeros((self.m, d - 1))

        # samples uniformly over the (d-1)-tuples of ints with each value in [1,k_vals[little_m] + d] and no repeated
        # values, and sorts them
        for little_m in range(self.m):
            temp[little_m, :] = np.sort(
                random_state.choice(k_vals[little_m] + d - 1, d - 1, replace=False) + 1)

        # computes the difference between consecutive sorted values (with borders added), yielding a d-dimensional tuple
        # of strictly positive ints summing to k+d
        alphas = np.diff(temp, axis=1, prepend=0, append=k_vals[:, None] + d)

        # removes one to every value, so that the possible values include 0 and that the sum is k
        alphas = alphas - 1
        assert (np.all(np.sum(alphas, axis=1) == k_vals))  # checks that the constraint is verified
        return alphas.astype(int)

    def group_sampling(self, random_state):
        """
        group_sampling samples from a distribution close to the theoretical one by splitting the dimensions in two
        groups. Define the gaps between the rescaled, sorted, consecutive values of eta and take the largest one and its
        associated values on both side. Then group_1 consists of the dimensions  where eta is larger than the top of
        the gap, and group_2 consists of the other ones. Then uses similar technique to initial_exact_sampling
        """

        scaled_eta = self.eta_ ** (self.r / (2 - self.r))
        eta_temp = np.c_[scaled_eta, np.arange(self.n_features_in_)]
        eta_sort = eta_temp[eta_temp[:, 0].argsort()]  # order smaller to bigger
        gap = eta_sort[1:self.n_features_in_, :] - np.c_[
            eta_sort[0:self.n_features_in_ - 1, 0], np.zeros(self.n_features_in_ - 1)]
        eta_border = scaled_eta[int(gap[np.argmax(gap[:, 0]), 1])]
        group_1 = eta_temp[eta_temp[:, 0] >= eta_border, 1].astype(int)
        group_2 = eta_temp[eta_temp[:, 0] < eta_border, 1].astype(int)
        assert (len(group_2) > 0)

        # compute eta_tilde, which is equal to the bottom of the largest gap on group_2 and the top of the largest gap
        # on group_1
        d1 = len(group_1)
        d2 = len(group_2)
        eta_tilde_1 = np.min(self.eta_[group_1])
        eta_tilde_2 = np.max(self.eta_[group_2])
        eta_group = eta_tilde_1 * np.ones(self.n_features_in_)
        eta_group[group_2] = eta_tilde_2

        # sample using the intermediate sampling on the sum of the sample on each group of dimensions
        alphas = np.zeros((self.m, self.n_features_in_), dtype=int)
        # choosing a maximum possible value for k1 and k2
        k_max = 40
        probabilities = np.zeros((k_max, k_max))
        values = np.zeros(k_max * k_max, dtype=int)
        # filing the array of probabilities
        for k1 in range(k_max):
            for k2 in range(k_max):
                probabilities[k1, k2] = comb(k1 + d1 - 1, d1 - 1) * comb(k2 + d2 - 1, d2 - 1) * (
                        self.rho ** (k1 + k2)) / (self.lamb_ + self.mu * (k1 / eta_tilde_1 + k2 / eta_tilde_2))
                # values[k1 + k_max * k2] = [k1, k2]
                values[k1 + k_max * k2] = k1 + k_max * k2
        probabilities[0, 0] = 0

        # sampling the pairs according to the previously defined distribution
        ks = random_state.choice(values, self.m, True, probabilities.flatten('F') / np.sum(probabilities))
        ks2 = ks // k_max
        ks1 = ks % k_max

        # computes the constance for importance sampling
        c_eta = np.sum(probabilities)

        # samples the alphas using the two groups
        alphas[:, group_1] = self.uniform_tuple_sampling(ks1, d1, random_state)
        alphas[:, group_2] = self.uniform_tuple_sampling(ks2, d2, random_state)
        self.uncollapsed_alphas_ = alphas
        self.alphas_, weight_sample = np.unique(alphas, axis=0, return_counts=True)  # collapses the non-unique alphas
        self.m_prime_ = len(self.alphas_)  # number of unique alphas
        self.weight_feature_ = np.diag(
            np.sqrt(weight_sample * c_eta * (self.lamb_ + self.mu * np.dot(self.alphas_, 1 / eta_group)) / (
                    self.m * (self.lamb_ + self.mu * np.dot(self.alphas_, 1 / self.eta_)))))
        return self

    def update_theta(self):
        """
        update_theta finds the regression function for a fixed kernel, either using the feature pov or the kernel pov,
        with the same result but faster computation depending on the dimensions of the problem.
        """
        if self.n_ > self.m_prime_:  # feature pov
            weighted_phi = np.c_[np.dot(self.phi_, self.weight_feature_), np.ones(self.n_)]
            regularization_matrix = np.identity(self.m_prime_ + 1)
            regularization_matrix[-1, -1] = 0
            self.theta_ = np.linalg.solve(np.dot(weighted_phi.T, weighted_phi) + self.n_ * regularization_matrix,
                                          np.dot(weighted_phi.T, self.y_))
        else:  # kernel pov
            weighted_phi = np.dot(self.phi_, self.weight_feature_)
            k = np.dot(weighted_phi, weighted_phi.T)
            k_tilde = np.c_[k, np.ones(self.n_)]
            big_k = np.zeros((self.n_ + 1, self.n_ + 1))
            big_k[:self.n_, :self.n_] = k
            delta = np.linalg.solve(np.dot(k_tilde.T, k_tilde) + self.n_ * big_k, np.dot(k_tilde.T, self.y_))
            self.theta_ = np.hstack([np.dot(weighted_phi.T, delta[:-1]), delta[-1]])
        return self

    def update_eta(self):
        theta = self.theta_[:-1]
        eta = np.sqrt(
            np.dot(np.multiply(np.dot(self.weight_feature_, theta), self.alphas_.T),
                   np.dot(self.weight_feature_, theta)))
        eta = np.sqrt(eta ** 2 + self.epsilon)
        self.eta_ = eta ** (2 - self.r) / np.sum(eta ** self.r) ** (2 / self.r - 1)
        return self

    @staticmethod
    @numba.njit
    def update_lambda(n_features_in_, r, theta_, weight_feature_, alphas_, m_prime_, U_, epsilon):
        theta = theta_[:-1]
        weighted_theta = np.dot(weight_feature_, theta)
        sqrt_alphas = np.sqrt(alphas_)

        # compute big lambda temp
        big_lambda_temp = np.zeros((n_features_in_, n_features_in_))
        identity = np.identity(n_features_in_)
        for j in range(m_prime_):
            for k in range(j):
                if np.abs(alphas_[j, :] - alphas_[k, :]).sum() == 2:
                    for s in range(n_features_in_):
                        for t in range(s):
                            if np.all(
                                    np.equal(alphas_[j, :] - identity[s, :],
                                             alphas_[k, :] - identity[t, :])):
                                big_lambda_temp[s, t] += weighted_theta[j] * weighted_theta[k] * sqrt_alphas[j, s] * \
                                                         sqrt_alphas[k, t]
                            elif np.all(
                                    np.equal(alphas_[k, :] - identity[s, :],
                                             alphas_[j, :] - identity[t, :])):
                                big_lambda_temp[s, t] += weighted_theta[j] * weighted_theta[k] * sqrt_alphas[k, s] * \
                                                         sqrt_alphas[j, t]

        diag = np.diag(np.dot(np.multiply(weighted_theta, alphas_.T), weighted_theta))
        big_lambda_temp += big_lambda_temp.T + diag
        M_f = np.dot(np.dot(U_, big_lambda_temp), U_.T)
        eta, U_ = np.linalg.eigh(M_f + epsilon * np.identity(n_features_in_))
        eta_ = eta ** ((2 - r) / 2) / np.sum(eta ** (r / 2)) ** (2 / r - 1)
        return eta_, U_

    @staticmethod
    def hermite_polynomials_features(X, alphas):
        """
        hermite_polynomials_features computes the Hermite polynomials described by the tuples in alphas on the data X
        """
        X = check_array(X)
        n, d = np.shape(X)
        m = len(alphas)
        max_alpha = np.max(alphas).astype(int)

        # construct 1D Hermite Polynomials up to the maximal possible value of alpha
        hermite_pol = np.zeros((n, d, max_alpha + 1))
        hermite_pol[:, :, 0] = 1
        if max_alpha > 0:
            hermite_pol[:, :, 1] = X
        for i_alpha in range(2, max_alpha + 1):
            hermite_pol[:, :, i_alpha] = X * hermite_pol[:, :, i_alpha - 1] / np.sqrt(i_alpha) - np.sqrt(
                (i_alpha - 1) / i_alpha) * hermite_pol[:, :, i_alpha - 2]

        # compute phi by iterating over all d dimensions and selecting the right 1D hermite polynomials
        phi = np.ones((n, m))
        for k in range(d):
            phi *= hermite_pol[:, k, alphas[:, k]]
        return phi

    def select_dim(self):
        self.dim_ = np.sum(self.eta_ ** (self.r / (2 - self.r)) > 1 / self.n_features_in_)
        if self.feature:
            self.p_hat_ = self.U_[:, self.n_features_in_ - self.dim_:self.n_features_in_]
        else:
            self.p_hat_ = np.zeros((self.n_features_in_, self.dim_))
            order_variables = np.argsort(self.eta_)
            for a in range(self.dim_):
                self.p_hat_[order_variables[self.n_features_in_ - a - 1], a] = 1

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        n = np.shape(X)[0]
        phi = self.hermite_polynomials_features(np.dot(X, self.U_), self.alphas_)
        weighted_phi = np.c_[np.dot(phi, self.weight_feature_), np.ones(n)]
        y_pred = np.dot(weighted_phi, self.theta_)
        return y_pred

    def _more_tags(self):
        return {
            'poor_score': True
        }

    def score(self, X, y, sample_weight=None):
        # R2 score, best score is 1, worst score can be -infinite, constant mean predictor score is 0
        y_pred = self.predict(X)
        return 1 - ((y - y_pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    def feature_learning_score(self, p):
        # Best score is 1 and worst score is 0
        s = np.minimum(np.shape(p)[0], np.shape(p)[1])
        if self.feature:
            p_hat = self.U_[:, self.n_features_in_ - s:self.n_features_in_]
        else:
            p_hat = np.zeros((self.n_features_in_, s))
            order_variables = np.argsort(self.eta_)
            for a in range(s):
                p_hat[order_variables[self.n_features_in_ - a - 1], a] = 1
        pi_p_hat = np.dot(np.dot(p_hat, np.linalg.inv(np.dot(p_hat.T, p_hat))), p_hat.T)
        pi_p = np.dot(np.dot(p, np.linalg.inv(np.dot(p.T, p))), p.T)
        if s <= self.n_features_in_ / 2:
            error = (np.linalg.norm(pi_p - pi_p_hat)) ** 2 / (2 * s)
        else:
            error = (np.linalg.norm(pi_p - pi_p_hat)) ** 2 / (2 * (self.n_features_in_ - s))
        return 1 - error

    def dimension_score(self, s):
        # Best score is 1 and worst score is 0
        if s <= self.n_features_in_ / 2:
            error = np.abs(self.dim_ - s) / (self.n_features_in_ - s)
        else:
            error = np.abs(self.dim_ - s) / s
        return 1 - error
