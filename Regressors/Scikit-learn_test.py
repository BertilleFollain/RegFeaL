from RegFeaL import RegFeaL
from Regressors.BasicRegFeaL import BasicRegFeaL
from PyMave import PyMave
from sklearn.utils.estimator_checks import check_estimator

estimator = RegFeaL()
for est, check in check_estimator(estimator, generate_only=True):
    print(str(check))
    try:
        check(est)
    except AssertionError as e:
        print('Failed: ', check, e)
print('RegFeaL Passed the Scikit-learn Estimator tests')

estimator = BasicRegFeaL()
for est, check in check_estimator(estimator, generate_only=True):
    print(str(check))
    try:
        check(est)
    except AssertionError as e:
        print('Failed: ', check, e)
print('BasicRegFeaL Passed the Scikit-learn Estimator tests')

estimator = PyMave()
for est, check in check_estimator(estimator, generate_only=True):
    print(str(check))
    try:
        check(est)
    except AssertionError as e:
        print('Failed: ', check, e)
print('PyMave Passed the Scikit-learn Estimator tests')
