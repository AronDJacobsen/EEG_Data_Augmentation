



# TODO: Figure out how to save the predictions from the models such that we can use them for the ensemble_pipeline.
# EDIT: This will be done at a later date, as ensemble should use models trained without cross-val.



#### initializing seed ####
import random
import numpy as np
random.seed(0)
np.random.seed(0)
###########################


from sklearn.model_selection import cross_val_score, KFold
from models.models import models
import matplotlib.pyplot as plt
from collections import defaultdict
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from prepData.dataLoader import *
from time import time
from models.balanceData import *
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import matplotlib.pyplot as plt
import scipy.stats
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

from models.GAN import GAN
from models.mixup import mixup


from collections import Counter

#prep_dir = r"C:\Users\Albert Kjøller\Documents\GitHub\TUAR_full_data\tempData" + "\\"
pickle_path = r"C:\Users\Albert Kjøller\Documents\GitHub\EEG_epilepsia"
#pickle_path = r"/Users/Jacobsen/Documents/GitHub/EEG_epilepsia" + "/"
#pickle_path = r"/Users/philliphoejbjerg/Documents/GitHub/EEG_epilepsia" + "/"

windowsOS = True

# Create pickles from preprocessed data based on the paths above. Unmuted when pickles exist
# subject_dict = createSubjectDict(prep_directory=prep_dir, windowsOS=True)
# PicklePrepData(subjects_dict=subject_dict, prep_directory=prep_dir, save_path=pickle_path, windowsOS = True)
#windows
#loading data - define which pickles to load (with NaNs or without)

X_file = r"\X_clean.npy"    #X_file = r"\X.npy"
y_file = r"\y_clean.npy"    #y_file = r"\y_clean.npy"
ID_file = r"\ID_frame_clean.npy"   #ID_file = r"\ID_frame.npy"


X = LoadNumpyPickles(pickle_path=pickle_path, file_name = X_file, windowsOS = windowsOS)
y = LoadNumpyPickles(pickle_path=pickle_path, file_name = y_file, windowsOS = windowsOS)
ID_frame = LoadNumpyPickles(pickle_path=pickle_path, file_name = ID_file, windowsOS = windowsOS)


# extract a subset for faster running time
#X, y, ID_frame = subset(X, y, ID_frame, no_indiv=30)
fast_run = False

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
           'n_estimators': hp.randint('n_estimators', 1, 101)
           } # we can increase interval

#don't think it is necessary
#spacegnb = {'var_smoothing': hp.choice('var_smoothing', range(0.01,1,0.01))}
spacegnb = None

#knn, nr neighbors
#vals = [ int for int in range(1, 500, 1) ] # remove since no number correlation
spaceknn = {'n_neighbors': hp.randint('n_neighbors', 1, 101)}

#lda, solvers for shrinkage
spacelda = {'solver': hp.choice('solver', ['svd', 'lsqr', 'eigen'])}

#mlp,
spacemlp = {'hidden_layer_sizes': hp.randint('hidden_layer_sizes', 1, 151),
            'learning_rate' : hp.choice('learning_rate', ['constant','adaptive']),
            'alpha' : hp.loguniform('alpha', np.log(0.00001), np.log(0.01))
            }

#rf, trees in estimating
spacerf = {'n_estimators': hp.randint('n_estimators', 1, 151),
           'criterion': hp.choice('criterion', ["gini", "entropy"]),
           'max_depth': hp.randint('max_depth', 1, 76)
           }

#sgd,
spacesgd = {'alpha': hp.loguniform('alpha', np.log(0.00001), np.log(1))
            } # we can increase interval

# Shape of the dictionary : model_dict = {Name of method: (function name (from models.py), hyperparams for opt) }
# TODO: XGBoost does not work at the moment
# model_dict = {'XGBoost' : ('XGBoost', None)} # for testing a single classifier.


# all
'''
model_dict = {'baseline_perm': ('baseline_perm', spaceb),
              'baseline_major': ('baseline_major', spaceb),
              'LR' : ('LR', spacelr),
              #'AdaBoost' : ('AdaBoost', spaceab),
              'GNB' : ('GNB', spacegnb),
              #'KNN' : ('KNN', spaceknn),
              'RF' : ('RF', spacerf),
              'LDA' : ('LDA', spacelda),
              #'MLP' : ('MLP', spacemlp),
              'SGD' : ('SGD', spacesgd)} #, 'XGBoost' : ('XGBoost', None)}

---

individual:

model_dict = {'baseline_perm': ('baseline_perm', spaceb)}
model_dict = {'baseline_major': ('baseline_major', spaceb)}
---
model_dict = {'LR' : ('LR', spacelr)}
model_dict = {'AdaBoost' : ('AdaBoost', spaceab)}
model_dict = {'GNB' : ('GNB', spacegnb)}
model_dict = {'KNN' : ('KNN', spaceknn)}
model_dict = {'RF' : ('RF', spacerf)}
model_dict = {'LDA' : ('LDA', spacelda)}
model_dict = {'MLP' : ('MLP', spacemlp)}
model_dict = {'SGD' : ('SGD', spacesgd)}
'''


#### define model to be evaluated and filename ####
experiment = 'smote_f2_LR_test' #'DataAug_color_noiseAdd_LR'
experiment_name = "_smote_f2_LR_test" #"_DataAug_color_Noise" added to saving files. For augmentation end with "_Noise" or so.
noise_experiment = None #r"\whitenoise_covarOne" # r"\colornoise30Hz_covarOne" #
# --> Should be named either GAN / Noise / MixUp, after _ in name.
# So that the following line will work:
# if experiment_name.split("_")[-1] == 'GAN':

model_dict = {'LR' : ('LR', spacelr)}

#### define augmentation ####
smote_ratio = np.array([0, 0.5, 1, 1.5, 2]) + 1 # np.array([0, 0.5, 1, 1.5, 2]) + 1 # Changed to be more in line with report
DataAug_ratio = np.array([0])#, 0.5, 1, 1.5, 2])
GAN_epochs = 20
clean_files = False

pickle_path_aug = pickle_path + r"\augmentation_pickles"

#TODO: Optimér noise augmentation, så den kan tage flere noise-addition måder (white_noise og colored) og køre samtidig.

if noise_experiment != None:
    # Load noise augmentation file
    X_noise = LoadNumpyPickles(pickle_path=pickle_path_aug + noise_experiment, file_name = X_file, windowsOS = windowsOS)
    y_noise = LoadNumpyPickles(pickle_path=pickle_path_aug + noise_experiment, file_name = y_file, windowsOS = windowsOS)
    ID_frame_noise = LoadNumpyPickles(pickle_path=pickle_path_aug + noise_experiment, file_name = ID_file, windowsOS = windowsOS)

    X_noise, y_noise, ID_frame_noise = binary(X_noise, y_noise, ID_frame_noise)

    if clean_files:
        X, y, ID_frame, X_noise, y_noise, ID_frame_noise = DeleteNanNoise(X, y, ID_frame, X_noise,y_noise, ID_frame_noise, save_path=pickle_path_aug + noise_experiment, windowsOS=windowsOS)

#### define no. hyperopt evaluations ####
HO_evals = 50 # for hyperopt


# Dictionary holding keys and values for all functions from the models.py file. Used to "look up" functions in the CV
# and hyperoptimization part
function_dict = models.__dict__

random_state_val = 0

#### define classes ####
artifact_names = ['eyem', 'chew', 'shiv', 'elpp', 'musc', 'null'] # ['null']
classes = len(artifact_names)

#for hyperopt data to save
def unpack(x):
    if x:
        return x[0]
    return np.nan

ho_trials = {} # fold, artifact, model, hyperopt iterations
results = {} # fold, artifact, model, scores


#### define no. of folds ####
K = 5 # 80% training and 20% testing
#setting fold details
kf = KFold(n_splits=K, random_state=random_state_val, shuffle = True) # random state + shuffle ensured same repeated experiments



for aug_ratio in DataAug_ratio:
    print("\n####---------------------------------------####")
    print("Running a", aug_ratio, "ratio of (augmented / real) using", experiment_name.split("_")[-1])
    print("####---------------------------------------####")
    #### Initializing dict for this Augmenation ratio:
    ho_trials[aug_ratio] = {}
    results[aug_ratio] = {}

    for ratio in smote_ratio:
        print("\n####---------------------------------------####")
        print("Running a", ratio-1, "ratio of (augmented / real)")
        print("####---------------------------------------####")

        i = 0 # CV fold index
        cross_val_time_start = time()

        #### initializing dict for this ratio ####
        ho_trials[aug_ratio][ratio] = {} # for this fold
        results[aug_ratio][ratio] = {}

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
            ho_trials[aug_ratio][ratio][i] = {} # for this fold
            results[aug_ratio][ratio][i] = {}

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

                if fast_run:
                    ytrain[:3] = 1
                    ytest[:3] = 1


                #### balancing data ####
                # now resample majority down to minority to achieve equal
                # - new name in order to not interfere with hyperopt
                if ratio == 1: # if no augmentation
                    Xtrain_new, ytrain_new = rand_undersample(Xtrain, ytrain, arg = 'majority', state = random_state_val, multi = False)
                else:
                    # for class 0 (artifact) we downsample to no. obs. for artifact * ratio(to smote up)
                    label_size = Counter(ytrain)
                    major = max(label_size, key = label_size.get)
                    decrease = label_size[1 - major] * ratio
                    label_size[major] = int( np.round( decrease, decimals = 0 ) )
                    Xtrain_new, ytrain_new = rand_undersample(Xtrain, ytrain, arg = label_size, state = random_state_val, multi = False)
                    Xtrain_new, ytrain_new = smote(Xtrain_new, ytrain_new, multi = False, state = random_state_val)


                #%% Data Augmentation step:
                if aug_ratio != 0:

                    if experiment_name.split("_")[-1] == 'GAN':

                        class_size = int(sum(ytrain_new)) # Sum of all the ones. Since data is balanced, the other class is same size

                        # Existing data for class 0 and 1 (Since not yet shuffled)
                        class0 = Xtrain_new[:class_size]
                        class1 = Xtrain_new[class_size:]

                        # GAN-augmented data, generated from existing data of each class.
                        GAN_class0 = GAN(class0, NtoGenerate = int(aug_ratio * class_size), epochs=GAN_epochs)
                        print("GAN class 0 complete")
                        GAN_class1 = GAN(class1, NtoGenerate = int(aug_ratio * class_size), epochs=GAN_epochs)
                        print("GAN class 1 complete")

                        Xtrain_new = np.concatenate( (Xtrain_new, GAN_class0, GAN_class1) )
                        ytrain_new = np.concatenate( (ytrain_new, np.zeros(int(aug_ratio * class_size)), np.ones(int(aug_ratio * class_size))) )


                    if experiment_name.split("_")[-1] == "MixUp":

                        # Onehot-encoding for mixup to work
                        y_onehot_encoded = OneHotEncoder(sparse=False).fit_transform(ytrain_new.reshape(len(ytrain_new), 1))

                        # Running mixup
                        mix_X, mix_y, _ = mixup(Xtrain_new, y_onehot_encoded, ratio)

                        # Undoing the onehot-encoding
                        mix_y = np.argmax(mix_y, axis=1)

                        Xtrain_new = np.concatenate( (Xtrain_new, mix_X) )
                        ytrain_new = np.concatenate( (ytrain_new, mix_y) )


                    if experiment_name.split("_")[-1] == "Noise":
                        X_noise_new = X_noise[train_indices, :]
                        y_noise_new = y_noise[train_indices, :]

                        # Balance noisy data
                        label_size = Counter(y_noise_new[:,artifact])
                        major = max(label_size, key=label_size.get)
                        decrease = label_size[1 - major]
                        label_size[major] = int(np.round(decrease, decimals=0))
                        X_noise_new, y_noise_new = rand_undersample(X_noise_new, y_noise_new[:,artifact], arg=label_size,
                                                                  state=random_state_val, multi=False)

                        # Find new points
                        N_noise = X_noise_new.shape[0]
                        N_clean = Xtrain_new.shape[0]
                        n_new_points = int(aug_ratio * N_clean)
                        noise_idxs = np.random.choice(N_noise, n_new_points)


                        # Select noisy data
                        noise_X = X_noise_new[noise_idxs,:]
                        noise_y = y_noise_new[noise_idxs]

                        # Concatenate
                        Xtrain_new = np.concatenate( (Xtrain_new, noise_X) )
                        ytrain_new = np.concatenate( (ytrain_new, noise_y) )

                        #TODO: Investigate noise plots - call function

                # TODO: Data augmentation for hyperopt

                #### creating test environment ####
                Xtrain_new, ytrain_new = shuffle(Xtrain_new, ytrain_new, random_state=random_state_val)
                Xtest, ytest = shuffle(Xtest, ytest, random_state=random_state_val)


                env = models(Xtrain_new, ytrain_new, Xtest, ytest, state = random_state_val)

                #### initializing validation data for hyperopt ####
                #TODO: Jeg tror vi bør kalde variablene noget andet, så vi ikke overwriter det vi har kaldt dem tidligere.
                trainindv, testindv = train_test_split(HO_individuals, test_size=0.20, random_state=random_state_val, shuffle = True)
                # indices of these individuals from ID_frame
                HO_train_indices = [i for i, ID in enumerate(train_ID_frame) if ID in trainindv]
                HO_test_indices = [i for i, ID in enumerate(train_ID_frame) if ID in testindv]
                #constructing sets
                HO_Xtrain, HO_ytrain = Xtrain[HO_train_indices,:], ytrain[HO_train_indices] # we keep the original and balance new later
                HO_Xtest, HO_ytest = Xtrain[HO_test_indices,:], ytrain[HO_test_indices]


                if fast_run:
                    HO_ytrain[:2] = 1
                    HO_ytest[:2] = 1

                #### initializing validation data for hyperopt ####

                if ratio == 1: # if no augmentation
                    HO_Xtrain_new, HO_ytrain_new = rand_undersample(HO_Xtrain, HO_ytrain, arg = 'majority', state = random_state_val, multi = False)
                else:
                    label_size = Counter(HO_ytrain)
                    major = max(label_size, key = label_size.get)
                    decrease = label_size[1 - major] * ratio
                    label_size[major] = int( np.round( decrease, decimals = 0 ) )

                    HO_Xtrain_new, HO_ytrain_new = rand_undersample(HO_Xtrain, HO_ytrain, arg = label_size, state = random_state_val, multi = False)
                    HO_Xtrain_new, HO_ytrain_new = smote(HO_Xtrain_new, HO_ytrain_new, multi = False, state = random_state_val)

                if aug_ratio != 0:

                    if experiment_name.split("_")[-1] == 'GAN':

                        class_size = int(sum(HO_ytrain_new)) # Sum of all the ones. Since data is balanced, the other class is same size

                        # Existing data for class 0 and 1 (Since not yet shuffled)
                        class0 = HO_Xtrain_new[:class_size]
                        class1 = HO_Xtrain_new[class_size:]

                        # GAN-augmented data, generated from existing data of each class.
                        GAN_class0 = GAN(class0, NtoGenerate = int(aug_ratio * class_size), epochs=GAN_epochs)
                        print("GAN class 0 complete")
                        GAN_class1 = GAN(class1, NtoGenerate = int(aug_ratio * class_size), epochs=GAN_epochs)
                        print("GAN class 1 complete")

                        HO_Xtrain_new = np.concatenate( (HO_Xtrain_new, GAN_class0, GAN_class1) )
                        HO_ytrain_new = np.concatenate( (HO_ytrain_new, np.zeros(int(aug_ratio * class_size)), np.ones(int(aug_ratio * class_size))) )


                    if experiment_name.split("_")[-1] == "MixUp":

                        # Onehot-encoding for mixup to work
                        y_onehot_encoded = OneHotEncoder(sparse=False).fit_transform(HO_ytrain_new.reshape(len(HO_ytrain_new), 1))

                        # Running mixup
                        mix_X, mix_y, _ = mixup(HO_Xtrain_new, y_onehot_encoded, ratio)

                        # Undoing the onehot-encoding
                        mix_y = np.argmax(mix_y, axis=1)

                        HO_Xtrain_new = np.concatenate( (HO_Xtrain_new, mix_X) )
                        HO_ytrain_new = np.concatenate( (HO_ytrain_new, mix_y) )


                    if experiment_name.split("_")[-1] == "Noise":
                        X_noise_new = X_noise[HO_train_indices, :]
                        y_noise_new = y_noise[HO_train_indices, :]

                        # Balance noisy data
                        label_size = Counter(y_noise_new[:,artifact])
                        major = max(label_size, key=label_size.get)
                        decrease = label_size[1 - major]
                        label_size[major] = int(np.round(decrease, decimals=0))
                        X_noise_new, y_noise_new = rand_undersample(X_noise_new, y_noise_new[:,artifact], arg=label_size,
                                                                  state=random_state_val, multi=False)

                        # Find new points
                        N_noise = X_noise_new.shape[0]
                        N_clean = HO_Xtrain_new.shape[0]
                        n_new_points = int(aug_ratio * N_clean)
                        noise_idxs = np.random.choice(N_noise, n_new_points)


                        # Select noisy data
                        noise_X = X_noise_new[noise_idxs,:]
                        noise_y = y_noise_new[noise_idxs]

                        # Concatenate
                        HO_Xtrain_new = np.concatenate( (HO_Xtrain_new, noise_X) )
                        HO_ytrain_new = np.concatenate( (HO_ytrain_new, noise_y) )


                HO_Xtrain_new, HO_ytrain_new = shuffle(HO_Xtrain_new, HO_ytrain_new, random_state=random_state_val)
                HO_Xtest, HO_ytest = shuffle(Xtest, ytest, random_state=random_state_val)



                HO_env = models(HO_Xtrain_new, HO_ytrain_new, HO_Xtest, HO_ytest, state = random_state_val)

                #### initializing dict for this artifact ####
                ho_trials[aug_ratio][ratio][i][artifact_names[artifact]] = {} # for this artifact
                results[aug_ratio][ratio][i][artifact_names[artifact]] = {}

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
                        ho_trials[aug_ratio][ratio][i][artifact_names[artifact]][key] = {} # for this model


                        print('HyperOpt on: ', key) # print model name

                        trials = Trials()

                        def objective(params):
                            accuracy, f2_s, sensitivity, y_pred = function_dict[name](HO_env, **params) # hyperopt environment
                            #it minimizes
                            return -sensitivity

                        best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=HO_evals, trials=trials)

                        #### saving evaluations ####
                        ho_trials[aug_ratio][ratio][i][artifact_names[artifact]][key] = pd.DataFrame([pd.Series(t["misc"]["vals"]).apply(unpack) for t in trials])
                        ho_trials[aug_ratio][ratio][i][artifact_names[artifact]][key]["sensitivity"] = [-t["result"]["loss"] for t in trials]
                        print('best parameter/s:', best)
                        #define best found function
                        f = function_dict[name](env, **best) # now test environment

                    #without hyperopt
                    else: # space is none
                        f = function_dict[name](env)
                    end_time = time()
                    took_time = (end_time - start_time)

                    print(key + ": \t" + str(f) + ". Time: {:f} seconds".format(took_time))

                    acc, F2, sensitivity, y_pred = f

                    #### initializing dict for this model ####
                    results[aug_ratio][ratio][i][artifact_names[artifact]][key] = {}
                    #### saving results[aug_ratio] ####
                    results[aug_ratio][ratio][i][artifact_names[artifact]][key]['y_pred'] = y_pred
                    results[aug_ratio][ratio][i][artifact_names[artifact]][key]['accuracy'] = acc
                    results[aug_ratio][ratio][i][artifact_names[artifact]][key]['weighted_F2'] = F2
                    results[aug_ratio][ratio][i][artifact_names[artifact]][key]['sensitivity'] = sensitivity

            # new fold
            i += 1


        cross_val_time_end = time()
        cross_val_time = cross_val_time_end - cross_val_time_start
        print("The cross-validation for ratio" + str(ratio) + " took " + str(np.round(cross_val_time, 3)) + " seconds = " + str(np.round(cross_val_time/60, 3)) + " minutes")
        print('\n\n')
        results[aug_ratio][ratio]['time'] = cross_val_time


#### saving data ####
# Remember to change name of pickle when doing a new first_pilot



if windowsOS:
    os.makedirs(pickle_path + r"\results\performance" + "\\" + experiment, exist_ok=True)
    os.makedirs(pickle_path + r"\results\hyperopt" + "\\" + experiment, exist_ok=True)
    os.makedirs(pickle_path + r"\results\y_true" + "\\" + experiment, exist_ok=True)

    SaveNumpyPickles(pickle_path + r"\results\performance" + "\\" + experiment, r"\results" + experiment_name, results, windowsOS)
    SaveNumpyPickles(pickle_path + r"\results\hyperopt" + "\\" + experiment, r"\ho_trials" + experiment_name, ho_trials, windowsOS)
    SaveNumpyPickles(pickle_path + r"\results\y_true" + "\\" + experiment, r"\y_true" + experiment_name, y_true_dict, windowsOS)


else:
    os.makedirs(pickle_path + r"results/performance" + "/" + experiment, exist_ok=True)
    os.makedirs(pickle_path + r"results/hyperopt" + "/" + experiment, exist_ok=True)
    os.makedirs(pickle_path + r"\results\y_true" + "/" + experiment, exist_ok=True)

    SaveNumpyPickles(pickle_path + r"results/performance" + "/" + experiment, r"/results" + experiment_name, results, windowsOS=False)
    SaveNumpyPickles(pickle_path + r"results/hyperopt" + "/" + experiment, r"/ho_trials" + experiment_name, ho_trials, windowsOS=False)
    SaveNumpyPickles(pickle_path + r"\results\y_true" + "/" + experiment, r"\y_true" + experiment_name, y_true_dict, windowsOS=False)


