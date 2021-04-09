

#### initializing seed ####
import random
random.seed(10)
###########################


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
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

#prep_dir = r"C:\Users\Albert Kjøller\Documents\GitHub\TUAR_full_data\tempData" + "\\"

pickle_path = r"C:\Users\Albert Kjøller\Documents\GitHub\EEG_epilepsia"
#pickle_path = r"/Users/Jacobsen/Documents/GitHub/EEG_epilepsia" + "/"
windowsOS = True

# Create pickles from preprocessed data based on the paths above. Unmuted when pickles exist
# subject_dict = createSubjectDict(prep_directory=prep_dir, windowsOS=True)
# PicklePrepData(subjects_dict=subject_dict, prep_directory=prep_dir, save_path=pickle_path, windowsOS = True)
#windows
#loading data - define which pickles to load (with NaNs or without)

X_file = r"\X_clean.npy"    #X_file = r"\X.npy"
y_file = r"\y_clean.npy"    #y_file = r"\y.npy"
ID_file = r"\ID_frame_clean.npy"   #ID_file = r"\ID_frame.npy"


X = LoadNumpyPickles(pickle_path=pickle_path, file_name = X_file, windowsOS = windowsOS)
y = LoadNumpyPickles(pickle_path=pickle_path, file_name = y_file, windowsOS = windowsOS)
ID_frame = LoadNumpyPickles(pickle_path=pickle_path, file_name = ID_file, windowsOS = windowsOS)

# extract a subset for faster running time
#X, y, ID_frame = subset(X, y, ID_frame, no_indiv=20)

# apply the inclusion principle
X, y, ID_frame = binary(X, y, ID_frame)

# the split data
individuals = np.unique(ID_frame)


#### defining model spaces ####

#baseline
spaceb = None

#logistic regression, inverse regularization strength
spacelr = {'C': hp.loguniform('C', np.log(0.00001), np.log(0.2))} # we can increase interval


#adaboost,
spaceab = {'learning_rate': hp.loguniform('learning_rate', np.log(0.0001), np.log(0.1)),
           'n_estimators': 1 + hp.randint('n_estimators', 199)
           } # we can increase interval

#don't think it is necessary
#spacegnb = {'var_smoothing': hp.choice('var_smoothing', range(0.01,1,0.01))}
spacegnb = None

#knn, nr neighbors
#vals = [ int for int in range(1, 500, 1) ] # remove since no number correlation
spaceknn = {'n_neighbors': 1 + hp.randint('n_neighbors', 249)}

#lda, solvers for shrinkage
spacelda = {'solver': hp.choice('solver', ['svd', 'lsqr', 'eigen'])}

#mlp,
spacemlp = {'hidden_layer_sizes': 1+hp.randint('hidden_layer_sizes', 149),
            'solver' : hp.choice('solver', ['lbfgs','sgd','adam']),
            'learning_rate' : hp.choice('learning_rate', ['constant','adaptive']),
            'alpha' : hp.loguniform('alpha', np.log(0.00001), np.log(0.01))
            }

#rf, trees in estimating
spacerf = {'n_estimators': 1+hp.randint('n_estimators', 149),
           'criterion': hp.choice('criterion', ["gini", "entropy"]),
           'max_depth': 1+hp.randint('max_depth', 100)
           }

#sgd,
spacesgd = {'alpha': hp.loguniform('alpha', np.log(0.00001), np.log(1))
            } # we can increase interval

# Shape of the dictionary : model_dict = {Name of method: (function name (from models.py), hyperparams for opt) }
# TODO: XGBoost does not work at the moment
# model_dict = {'XGBoost' : ('XGBoost', None)} # for testing a single classifier.


# all
"""
model_dict = {'baseline': ('baseline', spaceb),
              'LR' : ('LR', spacelr),
              #'AdaBoost' : ('AdaBoost', spaceab),
              'GNB' : ('GNB', spacegnb),
              #'KNN' : ('KNN', spaceknn),
              'RF' : ('RF', spacerf),
              'LDA' : ('LDA', spacelda),
              #'MLP' : ('MLP', spacemlp),
              'SGD' : ('SGD', spacesgd)} #, 'XGBoost' : ('XGBoost', None)}
"""
'''
individual:

model_dict = {'baseline': ('baseline', spaceb)}
model_dict = {'LR' : ('LR', spacelr)}
model_dict = {'AdaBoost' : ('AdaBoost', spaceab)}
model_dict = {'GNB' : ('GNB', spacegnb)}
model_dict = {'KNN' : ('KNN', spaceknn)}
model_dict = {'RF' : ('RF', spacerf)}
model_dict = {'LDA' : ('LDA', spacelda)}
model_dict = {'MLP' : ('MLP', spacemlp)}
model_dict = {'SGD' : ('SGD', spacesgd)}
'''

experiment_name = "_pilot_KNN_default"
model_dict = {'KNN_default' : ('KNN_default', None)}

#Pilot til Phillip:

#model_dict = {'AdaBoost' : ('AdaBoost', None)}


#Pilot til Aron:
#model_dict = {'KNN' : ('KNN', None)}


#Pilot til Albert:

#model_dict = {'MLP_default' : ('MLP_default', None)}


# Dictionary holding keys and values for all functions from the models.py file. Used to "look up" functions in the CV
# and hyperoptimization part
function_dict = models.__dict__

#### define no. hyperopt evaluations ####
HO_evals = 25 # for hyperopt
random_state_val = 0

#for hyperopt data to save
def unpack(x):
    if x:
        return x[0]
    return np.nan

#### define classes ####
artifact_names = ['eyem', 'chew', 'shiv', 'elpp', 'musc', 'null']
classes = len(artifact_names)

ho_trials = {} # fold, artifact, model, hyperopt iterations
results = {} # fold, artifact, model, scores

#### define no. of folds ####
K = 5 # 80% training and 20% testing
#setting fold details
kf = KFold(n_splits=K, random_state=random_state_val, shuffle = True) # random state + shuffle ensured same repeated experiments


i = 0 # CV fold index
cross_val_time_start = time()

for train_idx, test_idx in kf.split(individuals):
#single loop
#while i < 1:
#    trainindv, testindv = train_test_split(individuals, test_size=0.20, random_state=random_state_val, shuffle = True)
    #   REMEMBER to # the other below
    print("\n-----------------------------------------------")
    print("Running {:d}-fold CV - fold {:d}/{:d}".format(K, i+1, K))
    print("-----------------------------------------------")


    #### initializing data ####
    #their IDs
    trainindv = individuals[train_idx]
    testindv = individuals[test_idx]
    #their indexes in train and test
    train_indices = [ i for i, ID in enumerate(ID_frame) if ID in trainindv]
    test_indices = [i for i, ID in enumerate(ID_frame) if ID in testindv]
    X_train, y_train = X[train_indices,:], y[train_indices] # we keep the original and balance new later
    X_test, y_test = X[test_indices,:], y[test_indices]

    #### initializing hyperopt split #### #TODO: Det her er overflødigt, vi kan bare ændre HO_individuals til trainindv (tror jeg)
    train_ID_frame = ID_frame[train_indices]
    HO_individuals = np.unique(train_ID_frame) # for hyperopt split

    #### initializing dict for this fold ####
    ho_trials[i] = {} # for this fold
    results[i] = {}

    #### for each artifact ####
    for artifact in range(classes):

        print("\nTraining on the class: " + artifact_names[artifact] + "\n")

        #### initializing data ####
        # only include the artifact of interest
        #new name for the ones with current artifact
        Xtrain = X_train # only to keep things similar
        Xtest = X_test # only to keep things similar
        ytrain = y_train[:, artifact]
        ytest = y_test[:, artifact]


        ##################################
        # on small runs
        #ytrain[5:8] = 1
        #ytest[5:8] = 1
        ##################################


        #### balancing data ####
        # now resample majority down to minority to achive equal
        Xtrain_new, ytrain_new = rand_undersample(Xtrain, ytrain, arg = 'majority', state = 0, multi = False)
        # - called new in order to not interfere with hyperopt

        #### creating test environment ####
        Xtrain_new, ytrain_new = shuffle(Xtrain_new, ytrain_new, random_state=random_state_val)
        Xtest, ytest = shuffle(Xtest, ytest, random_state=random_state_val)
        env = models(Xtrain_new, ytrain_new, Xtest, ytest)

        #### initializing validation data for hyperopt ####
        #TODO: Jeg tror vi bør kalde variablene noget andet, så vi ikke overwriter det vi har kaldt dem tidligere.
        trainindv, testindv = train_test_split(HO_individuals, test_size=0.20, random_state=random_state_val, shuffle = True)
        # indices of these individuals from ID_frame
        HO_train_indices = [i for i, ID in enumerate(train_ID_frame) if ID in trainindv]
        HO_test_indices = [i for i, ID in enumerate(train_ID_frame) if ID in testindv]
        #constructing sets
        HO_Xtrain, HO_ytrain = Xtrain[HO_train_indices,:], ytrain[HO_train_indices] # we keep the original and balance new later
        HO_Xtest, HO_ytest = Xtrain[HO_test_indices,:], ytrain[HO_test_indices]
        # undersampling
        HO_Xtrain_new, HO_ytrain_new = rand_undersample(HO_Xtrain, HO_ytrain, arg = 'majority', state = 1, multi = False)
        # creating environment
        HO_Xtrain_new, HO_ytrain_new = shuffle(HO_Xtrain_new, HO_ytrain_new, random_state=random_state_val)
        HO_Xtest, HO_ytest = shuffle(Xtest, ytest, random_state=random_state_val)
        HO_env = models(HO_Xtrain_new, HO_ytrain_new, HO_Xtest, HO_ytest)

        #### initializing dict for this artifact ####
        ho_trials[i][artifact_names[artifact]] = {} # for this artifact
        results[i][artifact_names[artifact]] = {}

        #### for each model ####
        for key in model_dict:
            #https://medium.com/district-data-labs/parameter-tuning-with-hyperopt-faa86acdfdce
            #https://towardsdatascience.com/hyperparameter-optimization-in-python-part-2-hyperopt-5f661db91324
            #http://hyperopt.github.io/hyperopt/getting-started/search_spaces/
            start_time = time()

            name, space = model_dict[key]

            #### HyperOpt ####
            if space is not None: # if hyperopt is defined

                #### initializing dict for this model ####
                ho_trials[i][artifact_names[artifact]][key] = {} # for this model


                print('HyperOpt on: ', key) # print model name

                trials = Trials()

                def objective(params):
                    accuracy, f1_s, sensitivity = function_dict[name](HO_env, **params) # hyperopt environment
                    #it minimizes
                    return -sensitivity

                best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=HO_evals, trials=trials)

                #### saving evaluations ####
                ho_trials[i][artifact_names[artifact]][key] = pd.DataFrame([pd.Series(t["misc"]["vals"]).apply(unpack) for t in trials])
                ho_trials[i][artifact_names[artifact]][key]["sensitivity"] = [-t["result"]["loss"] for t in trials]
                print('best parameter/s:', best)
                #define best found function
                f = function_dict[name](env, **best) # now test environment

            #without hyperopt
            else: # sapce is none
                f = function_dict[name](env)
            end_time = time()
            took_time = (end_time - start_time)

            print(key + ": \t" + str(f) + ". Time: {:f} seconds".format(took_time))

            acc, F1, sensitivity = f

            #### initializing dict for this model ####
            results[i][artifact_names[artifact]][key] = {}
            #### saving results ####
            results[i][artifact_names[artifact]][key]['accuracy'] = acc
            results[i][artifact_names[artifact]][key]['weighted_F1'] = F1
            results[i][artifact_names[artifact]][key]['sensitivity'] = sensitivity

    # new fold
    i += 1


cross_val_time_end = time()
cross_val_time = cross_val_time_end - cross_val_time_start
print("The cross-validation took " + str(cross_val_time) + " seconds = " + str(cross_val_time/60) + " minutes")

print('\n\n')

#### saving data ####
# Remember to change name of pickle when doing a new experiment
SaveNumpyPickles(pickle_path + r"\results\performance", r"\results" + experiment_name, results, windowsOS)
SaveNumpyPickles(pickle_path + r"\results\hyperopt", r"\ho_trials" + experiment_name, ho_trials, windowsOS)




