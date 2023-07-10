from Experiments.Experiment1 import Experiment1
from Experiments.Experiment2 import Experiment2
from Experiments.Experiment3 import Experiment3
from Run_plots.Experiment1_plot import Experiment1_plot
from Run_plots.Experiment2_plot import Experiment2_plot
from Run_plots.Experiment3_plot import Experiment3_plot
import datetime

start = datetime.datetime.now()
########################################################################################################################
save_res = False  # todo change this to True is you want to save the results
save_fig = False  # todo change this to True if you want to save the figure
folder_fig = 'Figures/'  # todo change this to your own folder or it will overwrite!
folder_res = 'Experiments_results/'  # todo change this to your own folder or it will overwrite!
########################################################################################################################
# Check performance dependency on m
range_n = [150, 300, 1000]
range_m = [100, 500, 1000, 2000, 3000, 4000, 5000]
d = 10
filename = Experiment2(range_n, range_m, d=d, std_noise=0.0, save=save_res, easy=False, number_experiments=5,
                       filename=folder_res + 'results_Experiment2_d' + str(d) + '.pkl')
Experiment2_plot(filename, folder_fig, save_fig)
########################################################################################################################
# Check training behavior
n = 5000
m = 2500
d = 10
filename = Experiment3(n=n, m=m, d=d, save=save_res, easy=False, std_noise=0.0, n_iter=10,
                       filename=folder_res + 'results_Experiment3_d' + str(d) + '.pkl')
Experiment3_plot(filename, folder_fig, save_fig)
########################################################################################################################
# Check performance dependency on n and d
# todo warning! this is not the code that was used to generate the results of Experiment1 from the article, see 'readme.md'
feature = True  # todo change this to False if you want to see performance in the variable selection setting
d = 5
range_n = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
m = 5000
filename = Experiment1(range_n, m=m, easy=True, save=save_res, feature=True, std_noise=2.5, number_experiments=5,
                       d=d, filename=folder_res + 'results_Experiment1_no_feature_d' + str(d) + 'bis.pkl')
Experiment1_plot(filename, folder_fig, save_fig)
########################################################################################################################
# Check performance dependency on n and d
# todo warning! this is the code to generate the results of Experiment1, but we don't advise running it as is, as it would take a very long time.
# todo We ran this experiment on the cluster Cleps https://paris-cluster-2019.gitlabpages.inria.fr/cleps/cleps-userguide/index.html
#range_n = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
#for feature in [False, True]:
#    if feature:
#        m = 10000
#    else:
#        m = 5000
#    for d in [5, 10, 15, 20, 30, 40]:
#        filename = Experiment1(range_n, m=m, easy=False, save=save_res, feature=feature, std_noise=0.5,
#                               number_experiments=5,
#                               d=d, filename=folder_res + 'results_Experiment1_no_feature_d' + str(d) + '.pkl')
#        Experiment1_plot(filename, folder_fig, save_fig)
#######################################################################################################################
# Check performance dependency on n and d
# todo warning! this is the code to generate the results of Experiment1, but we don't advise running it as is, as it would take a very long time.
# todo We ran this experiment on the cluster Cleps https://paris-cluster-2019.gitlabpages.inria.fr/cleps/cleps-userguide/index.html
#range_n = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
#for feature in [False, True]:
#    for d in [5, 10, 15, 20, 30, 40]:
#        filename = Experiment1(range_n, m=m, easy=True, save=save_res, feature=feature, std_noise=2.5,
#                               number_experiments=5,
#                               d=d, filename=folder_res + 'results_Experiment1_no_feature_d' + str(d) + 'polynomial.pkl')
#        Experiment1_plot(filename, folder_fig, save_fig)
#######################################################################################################################
end = datetime.datetime.now()
print(start, end)
print('Launch over')
