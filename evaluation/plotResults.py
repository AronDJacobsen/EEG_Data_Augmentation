
import scipy.stats
import numpy as np
from prepData.dataLoader import *
import pickle
import matplotlib.pyplot as plt

#pickle_path = r"C:\Users\Albert Kjøller\Documents\GitHub\EEG_epilepsia"
pickle_path = r"/Users/Jacobsen/Documents/GitHub/EEG_epilepsia" + "/"
windowsOS = False


# fold, artifact, model, scores
results = LoadNumpyPickles(pickle_path=pickle_path, file_name = r'\results.npy', windowsOS = windowsOS)
results = results[()]
# fold, artifact, model, hyperopt iterations
HO_trials = LoadNumpyPickles(pickle_path=pickle_path, file_name = r'\HO_trials.npy', windowsOS = windowsOS)
HO_trials = HO_trials[()]

# only choose one fold
#construct keys
folds = list(results.keys())
artifacts = list(results[folds[0]].keys())
models = list(results[folds[0]][artifacts[0]].keys())
scores = list(results[folds[0]][artifacts[0]][models[0]].keys())

single = HO_trials[folds[0]][artifacts[0]][models[0]]



# hyperopt
cols = list(single.columns)
n = len(cols)
for i in range(n-1): # for every parameter
    plt.scatter(single[cols[i]], single[cols[n-1]])
    plt.title('HyperOpt: model: {}, artifact: {}'.format(models[0], cols[i]))
    plt.xlabel(cols[i])
    plt.ylabel('accuracy')
    plt.show()


'''

allModelsStats = []
model_names = []

for model in CV_scores:
    model_names.append(model)
    stats = np.empty((3,2))

    for j, metric in enumerate(CV_scores[model]):
        stats[j,0] = CV_scores[model][metric].mean()
        stats[j, 1] = CV_scores[model][metric].std()

    allModelsStats.append(stats)

modelMeans_acc = [modelStats[0, 0] for modelStats in allModelsStats]
modelError_acc = [modelStats[0, 1] for modelStats in allModelsStats]

x_pos = np.arange(len(modelMeans_acc))
fig, ax = plt.subplots()
ax.bar(x_pos, modelMeans_acc, yerr=modelError_acc, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel('Metric: accuracy')
ax.set_xticks(x_pos)
ax.set_xticklabels(model_names)
ax.set_title('Mean accuracy through {:d}-fold CV'.format(K))
ax.yaxis.grid(True)

# Save the figure and show
plt.tight_layout()
#plt.savefig('bar_plot_with_error_bars.png')
plt.show()


print("hej")
"""
        trials = Trials()
        best = fmin(f, space, algo=tpe.suggest, max_evals=100, trials=trials)

        xs = [t['misc']['vals']['n'] for t in trials.trials]
        ys = [-t['result']['loss'] for t in trials.trials]
        stop = 0"""


#TODO: Error bars and plots


'''
