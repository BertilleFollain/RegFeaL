from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from rpy2.robjects.packages import importr
from rpy2.robjects.pandas2ri import activate
import numpy as np
import pyearth
import warnings

importr("RcppArmadillo")
Mave = importr("MAVE")
activate()


class PyMave(RegressorMixin, BaseEstimator):
    """
    PyMave implements the MAVE regressor from  Xia, Y., Tong, H., Li, W.K. and Zhu, L.-X. (2002), An adaptive estimation
    of dimension reduction space. Journal of the Royal Statistical Society: Series B (Statistical Methodology),
    64: 363-410. https://doi.org/10.1111/1467-9868.03411and and integrates it into the Scikit-learn framework.
    """

    def __init__(self):
        return

    def fit(self, X, y):
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

        self.dr_mave_ = Mave.mave_compute(self.X_, self.y_)
        self.dim_ = np.argmin(np.array(Mave.mave_dim(dr=self.dr_mave_)[-3])) + 1
        self.p_hat_ = np.array(Mave.coef_mave(self.dr_mave_, dim=self.dim_))

        # fit MARS model on top, training on projected data
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.mars_ = pyearth.Earth(max_degree=self.dim_)
            self.mars_.fit(np.dot(self.X_, self.p_hat_), self.y_)

        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        y_pred = self.mars_.predict(np.dot(X, self.p_hat_))
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
        p_hat = np.array(Mave.coef_mave(self.dr_mave_, dim=s))
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
