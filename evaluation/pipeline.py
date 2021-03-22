from prepData.dataLoader import LoadPickles
from models.modelFitting_vol2 import *
from sklearn.model_selection import cross_val_score, KFold
from models.models import models
import numpy as np
from collections import defaultdict
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


#loading data
pickle_dir = r"C:\Users\Albert Kjøller\Documents\GitHub\EEG_epilepsia"
X, y, ID_frame = LoadPickles(pickle_path=pickle_dir, DelNan = True)
#finding individuals
individuals = np.unique(ID_frame)




spaceknn = {'n_neighbors': range(1,100)}

# TODO: Specify hyperparameters for optimization
# TODO: This should be done both in the model_dict and in the functions in the moodels.py file

# Shape of the dictionary : model_dict = {Name of method: (function name (from models.py), hyperparams for opt) }

# TODO: XGBoost does not work at the moment
# model_dict = {'XGBoost' : ('XGBoost', None)} # for testing a single classifier.
model_dict = {'Baseline': ('baseline', None), 'LogisticReg' : ('lr', None), 'Naive-Bayes' : ('gnb', None),
              'KNN' : ('knnf', spaceknn), 'RandomForest' : ('rf', None), 'LinearDiscriminantAnalysis' : ('LDA', None), 
               'MultiLayerPerceptron' : ('MLP', None), 'AdaBoost' : ('AdaBoost', None), 
               'StochGradientDescent_SVM' : ('SGD', None)} #, 'XGBoost' : ('XGBoost', None)}

# Dictionary holding keys and values for all functions from the models.py file. Used to "look up" functions in the CV
# and hyperoptimization part
function_dict = models.__dict__

#setting fold details
K = 5
kf = KFold(n_splits=K, random_state=None, shuffle=True)

# Initializing
CV_scores = defaultdict(dict)
i = 0
#looping CV folds
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

    env = models(X_train, y_train, X_test, y_test)

    #loop through models
    for key in model_dict:
        #https://medium.com/district-data-labs/parameter-tuning-with-hyperopt-faa86acdfdce
        #https://towardsdatascience.com/hyperparameter-optimization-in-python-part-2-hyperopt-5f661db91324

        name, space = model_dict[key]
        f = function_dict[name](env)
        print(key + ": \t" + str(f))
        # acc, F1, sensitivity = f
        acc, F1, sensitivity = f



        if name in CV_scores.keys():
            CV_scores[name]['accuracy'][i] = acc
            CV_scores[name]['F1'][i] = F1
            CV_scores[name]['sensitivity'][i] = sensitivity

        else:
            CV_scores[name] = {'accuracy' : np.zeros(K), 'F1' : np.zeros(K), 'sensitivity': np.zeros(K)}

            CV_scores[name]['accuracy'][i] = acc
            CV_scores[name]['F1'][i] = F1
            CV_scores[name]['sensitivity'][i] = sensitivity

    i += 1



print("Hej")
"""
        trials = Trials()
        best = fmin(f, space, algo=tpe.suggest, max_evals=100, trials=trials)

        xs = [t['misc']['vals']['n'] for t in trials.trials]
        ys = [-t['result']['loss'] for t in trials.trials]
        stop = 0"""


#TODO: Error bars and plots




