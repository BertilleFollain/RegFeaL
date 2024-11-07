# RegFeaL, Companion code to 
## *Follain, B. and Bach, F. (2024), Nonparametric Linear Feature Learning in Regression Through Regularisation, Electronic Journal of Statistics, 18(2), 4075-4118. (https://https://doi.org/10.1214/24-EJS2301 https://arxiv.org/abs/2307.12754)*

## What is this project for?
This is the companion code to Follain, B. and Bach, F. (2024), Nonparametric Linear Feature Learning in Regression Through Regularisation Electronic Journal of Statistics, 18(2), 4075-4118. (https://https://doi.org/10.1214/24-EJS2301 https://arxiv.org/abs/2307.12754).
It contains the estimator **RegFeaL** introduced in the previously cited article, the code to run the experiments from the article
and the results of said experiments. **RegFeaL** is a method for non-parametric regression with linear feature learning, 
which consists in empirical risk minimisation regularised by derivatives. See the article for more details. The method is available through the class RegFeaL in
'RegFeaL.py'. It is easy to use thanks to compatibility with Scikit-learn. 
The code is maintained by Bertille Follain (https://bertillefollain.netlify.app/, email address available on website). Do not 
hesitate to reach out if you need help using it.

## Organisation of the code
The regressors used in the experiments are available in the folder 'Regressors', while the functions corresponding to each 
experiment and the data generation are available in the folder 'Experiments'.
The results of the experiments (in .pkl format) are in the folder 'Experiments_results', 
while the figures can be found in the folder 'Figures'. The file to launch the experiments
is 'launch.py' for Experiments 1, 2 and 3. Note however that in practice, Experiment 1 was run on a cluster (cleps, https://paris-cluster-2019.gitlabpages.inria.fr/cleps/cleps-userguide/index.html). The requirements for use of the code are in 'requirements.txt'.
Note that some packages are necessary for the experiments but not to use **RegFeaL**. Note also that we used 
Python 3.7, mostly for compatibility with the package 'r2py' which we used to compare with another method, but
which is not necessary to run **RegFeaL**.

## Regressors
This folder calls for more detailed explanations. 'PyMave' corresponds to the Regressor described in 
'Yingcun Xia and others, An Adaptive Estimation of Dimension Reduction Space, Journal of the Royal Statistical Society Series B: Statistical Methodology, Volume 64, Issue 3, August 2002, Pages 363–410, https://doi.org/10.1111/1467-9868.03411'
but usable in Python (the original package is in the language R, which you will need if you want to use **PyMave** as **PyMave** simply calls the R package from Python).
'BasicRegFeaL' corresponds to the regressor described in
'Follain, B., Simsekli, U., and Bach, F. (2023), Nonparametric Linear Feature Learning in Regression Through Regularisation', 
while 'RegFeaL' (which we recommend using) is composed of the previously described regressor, which allows us to
select the space (with its dimension) on which to project the data, and an added retraining on the projected data using
MARS (Multiple Adaptive Regression Splines) from the package pyearth (https://contrib.scikit-learn.org/py-earth/). 'Scikit-learn_test.py' allows us to check that all 
three estimators are compatible with the Scikit-learn API (https://scikit-learn.org/stable/). 

## Example
The class RegFeaL has many parameters, which are detailed in the definition of the class. Here is a (simple) example of 
usage.
```
from Regressors.RegFeaL import RegFeaL
import numpy as np
import scipy.stats
n = 500 # number of samples
n_test = 500
d = 5 # original dimension
s= 2 # dimension of hidden linear subspace
X = np.sqrt(3) * (2 * np.random.uniform(size=(n, d)) - 1)
X_test = np.sqrt(3) * (2 * np.random.uniform(size=(n_test, d)) - 1)
p = scipy.stats.ortho_group.rvs(d)
p = p[:, 0:s]
y = np.sum(np.dot(X, p)**2, axis=1)
y_test = np.sum(np.dot(X_test, p)**2, axis=1)
regfeal = RegFeaL(feature = True, m =2000, rho =0.2)
regfeal.fit(X,y) # trains the estimator
y_pred = regfeal.predict(X_test) # predicts on new dataset
score = regfeal.score(X_test, y_test) # computes R2 score
feature_learning_score = regfeal.feature_learning_score(p) # computes feature learning score
print('score', score, 'feature learning score', feature_learning_score)
}
```
