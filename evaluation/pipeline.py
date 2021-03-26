from prepData.dataLoader import LoadPickles
from models.modelFitting_vol2 import *
from sklearn.model_selection import cross_val_score, KFold
from models.models import models
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from prepData.dataLoader import *
from time import time
from models.dataAugmentation import *
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import matplotlib.pyplot as plt
import scipy.stats

#prep_dir = r"C:\Users\Albert Kjøller\Documents\GitHub\TUAR_full_data\tempData" + "\\"
#pickle_path = r"C:\Users\Albert Kjøller\Documents\GitHub\EEG_epilepsia"

pickle_path = r"/Users/Jacobsen/Documents/GitHub/EEG_epilepsia" + "/"
mac = True

# Create pickles from preprocessed data based on the paths above. Unmuted when pickles exist
# subject_dict = createSubjectDict(prep_directory=prep_dir, windowsOS=True)
# PicklePrepData(subjects_dict=subject_dict, prep_directory=prep_dir, save_path=pickle_path, windowsOS = True)
#windows
#loading data - define which pickles to load (with NaNs or without)
if mac:
    X_file = r"X_clean.npy"    #X_file = r"\X.npy"
    y_file = r"y_clean.npy"    #y_file = r"\y.npy"
    ID_file = r"ID_frame_clean.npy"   #ID_file = r"\ID_frame.npy"
else:
    X_file = r"/X_clean.npy"    #X_file = r"\X.npy"
    y_file = r"/y_clean.npy"    #y_file = r"\y.npy"
    ID_file = r"/ID_frame_clean.npy"   #ID_file = r"\ID_frame.npy"



X, y, ID_frame = LoadNumpyPickles(pickle_path=pickle_path, X_file=X_file, y_file=y_file, ID_file=ID_file, DelNan = False)


# extract a subset
X, y, ID_frame = subset(X, y, ID_frame, no_indiv=10)


X, y, ID_frame = binary(X, y, ID_frame)

individuals = np.unique(ID_frame)

#downsampel none to twice that of the second largest
size = []
for i in range(len(y[0,:])):
    size.append(np.sum(y[:,i]))

second = np.sort(size)[-2]
down_to = int(second)

X, y = rand_undersample(X, y, arg = {5:down_to})


#upsample all to the majority
X, y = smote(X, y, multi = True)


# TODO: Specify hyperparameters for optimization
# TODO: This should be done both in the model_dict and in the functions in the moodels.py file

# Shape of the dictionary : model_dict = {Name of method: (function name (from models.py), hyperparams for opt) }

# TODO: XGBoost does not work at the moment
# TODO: KNN TAKES A REALLY LONG TIME (NOT ENDING!)
# model_dict = {'XGBoost' : ('XGBoost', None)} # for testing a single classifier.
'''
model_dict = {'Baseline': ('baseline', None), 'LogisticReg' : ('lr', None), 'Naive-Bayes' : ('gnb', None),
               'KNN' : ('knn', spaceknn), 'RandomForest' : ('rf', None), 'LinearDiscriminantAnalysis' : ('LDA', None),
               'MultiLayerPerceptron' : ('MLP', None), 'AdaBoost' : ('AdaBoost', None),
               'StochGradientDescent_SVM' : ('SGD', None)} #, 'XGBoost' : ('XGBoost', None)}
'''

spaceknn = {'n_neighbors': hp.choice('n_neighbors', range(1,60,1))}

spacerf = {'n_estimators': hp.choice('n_estimators', range(1,100,1))}

#spacerf = {'n_estimators': hp.choice('n_estimators', range(50,150,1)), 'criterion': hp.choice('criterion', ['gini', 'entropy'])}


#model_dict = {'Baseline': ('baseline', None), 'LogisticReg' : ('lr', None), 'KNN' : ('knn', spaceknn)}


model_dict = {'Baseline': ('baseline', None), 'LogisticReg' : ('lr', None), 'KNN' : ('knn', spaceknn), 'RF': ('rf',spacerf)}


# Dictionary holding keys and values for all functions from the models.py file. Used to "look up" functions in the CV
# and hyperoptimization part
function_dict = models.__dict__

#setting fold details
K = 5 # 80% training and 20% testing
kf = KFold(n_splits=K, random_state=None, shuffle=True)

# Initializing
CV_scores = defaultdict(dict)
#CV fold index
i = 0
#looping CV folds
#based on individuals
'''
for train_index, test_index in kf.split(individuals):

    print("\n-----------------------------------------------")
    print("Running {:d}-fold CV - fold {:d}/{:d}".format(K, i+1, K))
    print("-----------------------------------------------")

    #their IDs
    trainindv = individuals[train_index]
    testindv = individuals[test_index]

    #their indexes in train and test
    train_indices = [i for i, ID in enumerate(ID_frame) if ID in trainindv]
    test_indices = [i for i, ID in enumerate(ID_frame) if ID in testindv]

    # TODO: Extend to multiple binary classifiers (they should classify on the same data as well!).
    X_train, y_train = X[train_indices,:], y[train_indices][:,0]
    X_test, y_test = X[test_indices,:], y[test_indices][:,0]
'''
# not based on individuals
classes = 5

artifact_names = ['Eyemovement', 'Chew', 'Shiver', 'Elpp', 'Musc']

for train_index, test_index in kf.split(X):

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]


    for artifact in range(classes):

        #define new
        ytrain = y_train[:,artifact]
        ytest = y_test[:,artifact]

        #make classes even
        Xtrain, ytrain = smote(X_train, ytrain, multi=False)

        env = models(Xtrain, ytrain, X_test, ytest)


    #loop through models
        for key in model_dict:
        #https://medium.com/district-data-labs/parameter-tuning-with-hyperopt-faa86acdfdce
        #https://towardsdatascience.com/hyperparameter-optimization-in-python-part-2-hyperopt-5f661db91324

            start_time = time()

            name, space = model_dict[key]

            #hyperopt
            #no = 0
            #if no == 1:
            if space is not None:

                trials = Trials()

                def objective(params):
                    accuracy, f1_s, sensitivity = function_dict[name](env, **params)
                    #it minimizes
                    return -f1_s

                best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=8, trials=trials)

                # visualize one
                if i==0 and artifact == 0 and name == 'knn':

                    def unpack(x):
                        if x:
                            return x[0]
                        return np.nan
                    trials_df = pd.DataFrame([pd.Series(t["misc"]["vals"]).apply(unpack) for t in trials])
                    trials_df["loss"] = [t["result"]["loss"] for t in trials]
                    plt.scatter(trials_df["n_neighbors"], -trials_df["loss"])
                    plt.title('HyperOpt: model: {}, artifact: {}'.format(name, artifact_names[artifact]))
                    plt.xlabel('accuracy')
                    plt.ylabel('n_neighbors')


                #define best found function
                f = function_dict[name](env,**best)

            #without hyperopt
            else:
                f = function_dict[name](env)
            end_time = time()
            took_time = (end_time - start_time)

            print(key + ": \t" + str(f) + ". Time: {:f} seconds".format(took_time))

            # acc, F1, sensitivity = f
            acc, F1, sensitivity = f


            if artifact in CV_scores[name].keys():
                CV_scores[name][artifact]['accuracy'][i] = acc
                CV_scores[name][artifact]['F1'][i] = F1
                CV_scores[name][artifact]['sensitivity'][i] = sensitivity

            else:
                CV_scores[name][artifact] = {'accuracy' : np.zeros(K), 'F1' : np.zeros(K), 'sensitivity': np.zeros(K)}

                CV_scores[name][artifact]['accuracy'][i] = acc
                CV_scores[name][artifact]['F1'][i] = F1
                CV_scores[name][artifact]['sensitivity'][i] = sensitivity

    i += 1


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m-h, m, m+h


confidence = {}


artifact_names = ['Eyemovement', 'Chew', 'Shiver', 'Elpp', 'Musc']

model_names = []

initial_data = {}

for model in CV_scores:
    model_names.append(model)

    #initialize for appending
    accuracies = []
    weighted_f1s = []

    for artifact in CV_scores[model]:
        #add sensitivity to the data
        sensitivities = CV_scores[model][artifact]['sensitivity']

        if artifact_names[artifact] in initial_data:
            initial_data[artifact_names[artifact]].append(sensitivities.mean())
        else:
            initial_data[artifact_names[artifact]] = [sensitivities.mean()]

        #confidence interval
        if model == 'knn':
            minus, mean, plus = mean_confidence_interval(sensitivities)

            confidence[artifact_names[artifact]] = [minus, mean, plus]

        accuracies.append(CV_scores[model][artifact]['accuracy'].mean())
        weighted_f1s.append(CV_scores[model][artifact]['F1'].mean())


    if 'accuracy' in initial_data:
        initial_data['accuracy'].append(np.mean(accuracies))
    else:
        initial_data['accuracy'] = [np.mean(accuracies)]

    if 'weighted_f1' in initial_data:
            initial_data['weighted_f1'].append(np.mean(accuracies))
    else:
        initial_data['weighted_f1'] = [np.mean(accuracies)]

df_eval = pd.DataFrame.from_dict(initial_data)
df_eval.index = model_names


df_conf = pd.DataFrame.from_dict(confidence)
df_conf.index = ['lower', 'mean', 'upper']

#display all dataframe
pd.set_option("display.max_rows", None, "display.max_columns", None)

print(df_eval)

print('\n')
print('Based on KNN')
print(df_conf)

stop = 0

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

