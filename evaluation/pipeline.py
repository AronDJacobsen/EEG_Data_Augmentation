

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

from models.GAN import useGAN
from models.mixup import useMixUp
from models.noiseAddition import useNoiseAddition, prepareNoiseAddition

from collections import Counter


class pipeline:
    """A class for easily running experiments on EEG Artifact classification.
     """

    def __init__(self, pickle_path, X_file, y_file, ID_file, windowsOS=False):
        """The constructor for the pipeline class. Initializing all necessary information.

        Attributes:
            pickle_path (str): location of the preprocessed EEG data pickles.
            X_file (str): name of the feature matrix pickle, i.e. r"\X.npy".
            y_file (str): name of the target array pickle, i.e. r"\y.npy".
            ID_file (str): name of the the ID array holding info of the ID of each observation, i.e."r\ID_frame.npy".
            windowsOS (bool): True if running on a Windows operating system.
        """
        if windowsOS:
            self.slash = "\\"
        else:
            self.slash = "/"

        self.pickle_path = pickle_path
        self.X_file = X_file
        self.y_file = y_file
        self.ID_file = ID_file
        self.windowsOS = windowsOS

        #### defining model spaces ####
        # Shape of the dictionary : model_dict = {Name of method: (function name (from models.py), hyperparams for opt) }

        spaceb = None  # Baseline

        spacelr = {'C': hp.loguniform('C', np.log(0.00001), np.log(0.2))}  # Logistic Regression

        spaceab = {'learning_rate': hp.loguniform('learning_rate', np.log(0.0001), np.log(0.1)),  # AdaBoost,
                   'n_estimators': hp.randint('n_estimators', 1, 101)}

        spacegnb = None  # Gaussian Naive Bayes

        spaceknn = {'n_neighbors': hp.randint('n_neighbors', 1, 101)}  # KNN

        spacelda = {'solver': hp.choice('solver', ['svd', 'lsqr', 'eigen'])}  # LDA

        spacemlp = {'hidden_layer_sizes': hp.randint('hidden_layer_sizes', 1, 151),  # MLP
                    'learning_rate': hp.choice('learning_rate', ['constant', 'adaptive']),
                    'alpha': hp.loguniform('alpha', np.log(0.00001), np.log(0.01))
                    }

        spacerf = {'n_estimators': hp.randint('n_estimators', 1, 151),  # Random Forest
                   'criterion': hp.choice('criterion', ["gini", "entropy"]),
                   'max_depth': hp.randint('max_depth', 1, 76)
                   }

        spacesgd = {'alpha': hp.loguniform('alpha', np.log(0.00001), np.log(1))}  # Stochastic gradient descent

        self.full_model_dict = {'baseline_perm': {'baseline_perm': ('baseline_perm', spaceb)},
                                'baseline_major': {'baseline_major': ('baseline_major', spaceb)},
                                'LR': {'LR': ('LR', spacelr)},
                                'AdaBoost': {'AdaBoost': ('AdaBoost', spaceab)},
                                'GNB': {'GNB': ('GNB', spacegnb)},
                                'KNN': {'KNN': ('KNN', spaceknn)},
                                'RF': {'RF': ('RF', spacerf)},
                                'LDA': {'LDA': ('LDA', spacelda)},
                                'MLP': {'MLP': ('MLP', spacemlp)},
                                'SGD': {'SGD': ('SGD', spacesgd)}}
        super(pipeline, self)

    def runPipeline(self, model, HO_evals, smote_ratios, aug_ratios, experiment, experiment_name,
                    artifact_names=None, GAN_epochs=100, noise_experiment=None,
                    DelNan_noiseFiles=False, fast_run=False, K=5, random_state=0, save_y_true=False):
        """
        Parameters:
            :param model (str): Model name. Should follow the naming of the defined models in the
                                models.models.py script, i.e. 'LR'.
            :param HO_evals (int): Number of queries to call in the optimization using Bayesian Optimization with Hyperopt.
            :param smote_ratios (np.array(float)): array of floats. For creating 100% additional samples of the minority
                                                    class, the float-value should be set to 1. Similarly 50% upsampling
                                                    means a float-value of 0.5. Compatible with multiple values.
            :param aug_ratios (np.array(float)): array of floats. For creating 100% augmented data compared to original
                                                    data, the float-value should be set to 1. Similarly 50% extra
                                                    augmented data means a float-value of 0.5. Compatible with multiple values.
            :param experiment (str): name of the wider experiment, i.e. MixUp_experiment_No5
            :param experiment_name (str): name of the pickles created.
                                            When doing Augmentation it should end with either _GAN / _Noise / _MixUp,
                                            such that the following command can work properly:

                                            if experiment_name.split("_")[-1] == 'GAN':

                                            When running the pipeline on a single artifact, one should manually write this in the
                                            'experiment_name', i.e. experiment + model + 'Null' + aug_method
            :param artifact_names (list(str)): a list of the artifacts that are to be investigated. Default artifact are
                                                ['eyem', 'chew', 'shiv', 'elpp', 'musc', 'null'] as the study is carried
                                                out using the TUH EEG Artifact corpus.
            :param GAN_epochs (int): number of epochs to use if the GAN augmentation technique is used.
            :param noise_experiment (str): directory containing the noise pickles to be used. Should be None when
                                            not experimenting with Noise Addition augmentation technique.
            :param DelNan_noiseFiles (bool): False per default. Should be True if using data that contains NaN-values.
            :param fast_run (bool): True if a quick run-through of the pipeline is needed in order to detect bugs or new
                                    functions.
            :param K (int): number of folds for the cross-validation (CV) applied. Default value is 5.
            :param random_state (int): random seed for the used functions such as the train_test_split in KFold CV.
            :param save_y_true (bool): True if a pickle of the true y-values in each CV-fold should be saved.

        Returns:
            :return: Saves pickles of dictionaries of results from an experiment.
                        The "results_x_.npy" file is a nested dictionary holding information about accuracy, F2-score,
                        sensitivity score and the predicted labels on the test data for the model throughout the specified
                        augmentation ratios, SMOTE-ratios and CV-folds for the specified artifact(s).

                        The "ho_trials_x_.npy" file is a nested dictionary holding information about Hyperopt calls for the
                        model throughout the specified augmentation ratios, SMOTE-ratios and CV-folds for the specified artifact(s).

                        The "y_true_x_.npy" file is a nested dictionary holding information about the true y-labels
                        throughout CV-folds for the specified artifact(s).
        """

        X = LoadNumpyPickles(pickle_path=self.pickle_path + self.slash , file_name=self.X_file, windowsOS=self.windowsOS)
        y = LoadNumpyPickles(pickle_path=self.pickle_path + self.slash  , file_name=self.y_file, windowsOS=self.windowsOS)
        ID_frame = LoadNumpyPickles(pickle_path=self.pickle_path + self.slash , file_name=self.ID_file, windowsOS=self.windowsOS)
        X = LoadNumpyPickles(pickle_path=self.pickle_path + self.slash, file_name=r"\X.npy", windowsOS=self.windowsOS)
        # extract a subset for faster running time
        # X, y, ID_frame = subset(X, y, ID_frame, no_indiv=30)

        # apply the inclusion principle
        X, y, ID_frame = binary(X, y, ID_frame)

        # The KFold will be splitted by
        individuals = np.unique(ID_frame)

        # Choose model
        model_dict = self.full_model_dict[model]
        HO_evals = HO_evals

        pickle_path_aug = self.pickle_path + r"\augmentation_pickles"

        if noise_experiment != None:
            X_noise, y_noise, ID_frame_noise = prepareNoiseAddition(pickle_path_aug, noise_experiment, self.X_file, self.y_file,
                                                                    self.ID_file, windowsOS=self.windowsOS)

            if DelNan_noiseFiles:
                X, y, ID_frame, X_noise, y_noise, ID_frame_noise = DeleteNanNoise(X, y, ID_frame, X_noise, y_noise,
                                                                                  ID_frame_noise,
                                                                                  save_path=pickle_path_aug + noise_experiment,
                                                                                  windowsOS=self.windowsOS)

        # Dictionary holding keys and values for all functions from the models.py file. Used to "look up" functions in the CV
        # and hyperoptimization part
        function_dict = models.__dict__

        random_state_val = random_state

        #### define classes ####
        if artifact_names is None:
            artifact_names = ['eyem', 'chew', 'shiv', 'elpp', 'musc', 'null']

        classes = len(artifact_names)

        # for hyperopt data to save
        def unpack(x):
            if x:
                return x[0]
            return np.nan

        ho_trials = {}  # fold, artifact, model, hyperopt iterations
        results = {}  # fold, artifact, model, scores
        y_true_dict = {}

        # setting fold details
        kf = KFold(n_splits=K, random_state=random_state_val,
                   shuffle=True)  # random state + shuffle ensured same repeated experiments

        for aug_ratio in aug_ratios:
            if aug_ratio != 0:
                print("\n####---------------------------------------####")
                print("Running a", aug_ratio, "ratio of (augmented / real) using", experiment_name.split("_")[-1])
                print("####---------------------------------------####")
            #### Initializing dict for this Augmenation ratio:
            ho_trials[aug_ratio] = {}
            results[aug_ratio] = {}

            for ratio in smote_ratios:
                ratio += 1  # For consistency with augmentation ratio

                if ratio != 1:
                    print("\n####---------------------------------------####")
                    print("Running a", ratio - 1, "ratio of (SMOTE upsampling)")
                    print("####---------------------------------------####")

                i = 0  # CV fold index
                cross_val_time_start = time()

                #### initializing dict for this ratio ####
                ho_trials[aug_ratio][ratio] = {}  # for this fold
                results[aug_ratio][ratio] = {}

                for train_idx, test_idx in kf.split(individuals):
                    # single loop
                    # while i < 1:
                    #    trainindv, testindv = train_test_split(individuals, test_size=0.20, random_state=random_state_val, shuffle = True)
                    #   REMEMBER to # the other below
                    print("\n-----------------------------------------------")
                    print("Running {:d}-fold CV - fold {:d}/{:d}".format(K, i + 1, K))
                    print("-----------------------------------------------")

                    #### initializing data ####
                    # their IDs
                    trainindv = individuals[train_idx]
                    testindv = individuals[test_idx]
                    # their indexes in train and test
                    train_indices = [i for i, ID in enumerate(ID_frame) if ID in trainindv]
                    test_indices = [i for i, ID in enumerate(ID_frame) if ID in testindv]
                    testID_list = [ID for i, ID in enumerate(ID_frame) if ID in testindv]
                    X_train, y_train = X[train_indices, :], y[
                        train_indices]  # we keep the original and balance new later
                    X_test, y_test = X[test_indices, :], y[test_indices]

                    #### initializing hyperopt split #### #TODO: Det her er overflødigt, vi kan bare ændre HO_individuals til trainindv (tror jeg)
                    train_ID_frame = ID_frame[train_indices]
                    HO_individuals = np.unique(train_ID_frame)  # for hyperopt split

                    #### initializing dict for this fold ####
                    ho_trials[aug_ratio][ratio][i] = {}  # for this fold
                    results[aug_ratio][ratio][i] = {}

                    y_true_dict[i] = {}

                    #### for each artifact ####
                    for artifact in range(classes):

                        print("\nTraining on the class: " + artifact_names[artifact] + "\n")

                        #### initializing data ####
                        # only include the artifact of interest
                        # new name for the ones with current artifact
                        Xtrain = X_train  # only to keep things similar
                        Xtest = X_test  # only to keep things similar
                        ytrain = y_train[:, artifact]
                        ytest = y_test[:, artifact]

                        ##################################
                        # on small runs
                        # ytrain[5:8] = 1
                        # ytest[5:8] = 1
                        ##################################

                        if fast_run:
                            ytrain[:3] = 1
                            ytest[:3] = 1

                        #### balancing data ####
                        # now resample majority down to minority to achieve equal
                        # - new name in order to not interfere with hyperopt
                        if ratio == 1:  # if no augmentation
                            Xtrain_new, ytrain_new = rand_undersample(Xtrain, ytrain, arg='majority',
                                                                      state=random_state_val, multi=False)
                        else:
                            # Using mix of undersampling and smote
                            Xtrain_new, ytrain_new = balanceData(Xtrain, ytrain, ratio, random_state_val=random_state_val)

                        # %% Data Augmentation step:
                        if aug_ratio != 0:

                            if experiment_name.split("_")[-1] == 'GAN':
                                Xtrain_new, ytrain_new = useGAN(Xtrain_new, ytrain_new, aug_ratio, GAN_epochs, experiment_name)

                            if experiment_name.split("_")[-1] == "MixUp":
                                Xtrain_new, ytrain_new = useMixUp(Xtrain_new, ytrain_new, aug_ratio)

                            if experiment_name.split("_")[-1] == "Noise":
                                X_noise_new = X_noise[train_indices, :]
                                y_noise_new = y_noise[train_indices, :]
                                y_noise_new = y_noise_new[:, artifact]

                                Xtrain_new, ytrain_new = useNoiseAddition(X_noise_new, y_noise_new, Xtrain_new,
                                                                          ytrain_new, aug_ratio, random_state_val)

                        #### creating test environment ####
                        Xtrain_new, ytrain_new = shuffle(Xtrain_new, ytrain_new, random_state=random_state_val)
                        Xtest, ytest, testID_list = shuffle(Xtest, ytest, testID_list, random_state=random_state_val)

                        env = models(Xtrain_new, ytrain_new, Xtest, ytest, state=random_state_val)

                        #### initializing validation data for hyperopt ####
                        trainindv, testindv = train_test_split(HO_individuals, test_size=0.20,
                                                               random_state=random_state_val, shuffle=True)

                        # indices of these individuals from ID_frame
                        HO_train_indices = [i for i, ID in enumerate(train_ID_frame) if ID in trainindv]
                        HO_test_indices = [i for i, ID in enumerate(train_ID_frame) if ID in testindv]

                        # constructing sets
                        HO_Xtrain, HO_ytrain = Xtrain[HO_train_indices, :], ytrain[
                            HO_train_indices]  # we keep the original and balance new later
                        HO_Xtest, HO_ytest = Xtrain[HO_test_indices, :], ytrain[HO_test_indices]

                        if fast_run:
                            HO_ytrain[:2] = 1
                            HO_ytest[:2] = 1

                        #### initializing validation data for hyperopt ####

                        if ratio == 1:  # if no augmentation
                            HO_Xtrain_new, HO_ytrain_new = rand_undersample(HO_Xtrain, HO_ytrain, arg='majority',
                                                                            state=random_state_val, multi=False)
                        else:
                            # Using mix of undersampling and smote
                            HO_Xtrain_new, HO_ytrain_new = balanceData(HO_Xtrain, HO_ytrain, ratio, random_state_val=random_state_val)

                        if aug_ratio != 0:

                            if experiment_name.split("_")[-1] == 'GAN':
                                HO_Xtrain_new, HO_ytrain_new = useGAN(HO_Xtrain_new, HO_ytrain_new, aug_ratio,
                                                                      GAN_epochs, experiment_name)

                            if experiment_name.split("_")[-1] == "MixUp":
                                HO_Xtrain_new, HO_ytrain_new = useMixUp(HO_Xtrain_new, HO_ytrain_new, aug_ratio)

                            if experiment_name.split("_")[-1] == "Noise":
                                X_noise_new = X_noise[HO_train_indices, :]
                                y_noise_new = y_noise[HO_train_indices, :]
                                y_noise_new = y_noise_new[:, artifact]

                                HO_Xtrain_new, HO_ytrain_new = useNoiseAddition(X_noise_new, y_noise_new, HO_Xtrain_new,
                                                                                HO_ytrain_new, aug_ratio,
                                                                                random_state_val)

                        # Hyperopt environment
                        HO_Xtrain_new, HO_ytrain_new = shuffle(HO_Xtrain_new, HO_ytrain_new,
                                                               random_state=random_state_val)
                        HO_Xtest, HO_ytest = shuffle(Xtest, ytest, random_state=random_state_val)

                        HO_env = models(HO_Xtrain_new, HO_ytrain_new, HO_Xtest, HO_ytest, state=random_state_val)

                        #### initializing dict for this artifact ####
                        ho_trials[aug_ratio][ratio][i][artifact_names[artifact]] = {}  # for this artifact
                        results[aug_ratio][ratio][i][artifact_names[artifact]] = {}
                        y_true_dict[i][artifact_names[artifact]] = {}

                        #### for each model ####
                        for key in model_dict:
                            # https://medium.com/district-data-labs/parameter-tuning-with-hyperopt-faa86acdfdce
                            # https://towardsdatascience.com/hyperparameter-optimization-in-python-part-2-hyperopt-5f661db91324
                            # http://hyperopt.github.io/hyperopt/getting-started/search_spaces/
                            start_time = time()

                            name, space = model_dict[key]

                            #### HyperOpt ####
                            if space is not None:  # if hyperopt is defined

                                #### initializing dict for this model ####
                                ho_trials[aug_ratio][ratio][i][artifact_names[artifact]][key] = {}  # for this model

                                print('HyperOpt on: ', key)  # print model name

                                trials = Trials()

                                def objective(params):
                                    accuracy, f2_s, sensitivity, y_pred = function_dict[name](HO_env,
                                                                                              **params)  # hyperopt environment
                                    # it minimizes
                                    return -sensitivity

                                best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=HO_evals,
                                            trials=trials)

                                #### saving evaluations ####
                                ho_trials[aug_ratio][ratio][i][artifact_names[artifact]][key] = pd.DataFrame(
                                    [pd.Series(t["misc"]["vals"]).apply(unpack) for t in trials])
                                ho_trials[aug_ratio][ratio][i][artifact_names[artifact]][key]["sensitivity"] = [
                                    -t["result"]["loss"] for t in trials]
                                print('best parameter/s:', best)

                                # define best found function
                                f = function_dict[name](env, **best)  # now test environment

                            # without hyperopt
                            else:  # space is none
                                f = function_dict[name](env)
                            end_time = time()
                            took_time = (end_time - start_time)

                            print(key + ": \t" + str(f[:3]) + ". Time: {:f} seconds".format(took_time))

                            acc, F2, sensitivity, y_pred = f

                            #### initializing dictionary of results for this model ####
                            results[aug_ratio][ratio][i][artifact_names[artifact]][key] = {}

                            #### saving results[aug_ratio] ####
                            results[aug_ratio][ratio][i][artifact_names[artifact]][key]['y_pred'] = y_pred
                            results[aug_ratio][ratio][i][artifact_names[artifact]][key]['accuracy'] = acc
                            results[aug_ratio][ratio][i][artifact_names[artifact]][key]['weighted_F2'] = F2
                            results[aug_ratio][ratio][i][artifact_names[artifact]][key]['sensitivity'] = sensitivity

                        if aug_ratio == 0:
                            y_true_dict[i][artifact_names[artifact]]["y_true"] = env.y_test
                            y_true_dict[i][artifact_names[artifact]]["ID_list"] = testID_list

                    # new fold
                    i += 1

                cross_val_time_end = time()
                cross_val_time = cross_val_time_end - cross_val_time_start
                print("The cross-validation for ratio" + str(ratio - 1) + " took " + str(
                    np.round(cross_val_time, 3)) + " seconds = " + str(np.round(cross_val_time / 60, 3)) + " minutes")
                print('\n\n')
                results[aug_ratio][ratio]['time'] = cross_val_time

        #### saving data ####
        # Remember to change name of pickle when doing a new first_pilot

        if self.windowsOS:
            os.makedirs(self.pickle_path + r"\results\performance" + "\\" + experiment, exist_ok=True)
            os.makedirs(self.pickle_path + r"\results\hyperopt" + "\\" + experiment, exist_ok=True)
            os.makedirs(self.pickle_path + r"\results\y_true", exist_ok=True)

            SaveNumpyPickles(self.pickle_path + r"\results\performance" + "\\" + experiment,
                             r"\results" + experiment_name, results, self.windowsOS)
            SaveNumpyPickles(self.pickle_path + r"\results\hyperopt" + "\\" + experiment,
                             r"\ho_trials" + experiment_name, ho_trials, self.windowsOS)

            if save_y_true:
                SaveNumpyPickles(self.pickle_path + r"\results\y_true", r"\y_true_" + str(K) + "fold_randomstate_"
                                 + str(random_state_val), y_true_dict, self.windowsOS)


        else:
            os.makedirs(self.pickle_path + r"results/performance" + "/" + experiment, exist_ok=True)
            os.makedirs(self.pickle_path + r"results/hyperopt" + "/" + experiment, exist_ok=True)
            os.makedirs(self.pickle_path + r"results/y_true", exist_ok=True)

            SaveNumpyPickles(self.pickle_path + r"results/performance" + "/" + experiment,
                             r"/results" + experiment_name, results, windowsOS=self.windowsOS)
            SaveNumpyPickles(self.pickle_path + r"results/hyperopt" + "/" + experiment, r"/ho_trials" + experiment_name,
                             ho_trials, windowsOS=self.windowsOS)

            if save_y_true:
                SaveNumpyPickles(self.pickle_path + r"results/y_true", r"/y_true_" + str(K) + "fold_randomstate_"
                                 + str(random_state_val) + y_true_dict, windowsOS=self.windowsOS)


if __name__ == '__main__':
    """ Select path to the data-pickles ! """
    pickle_path = r"C:\Users\Albert Kjøller\Documents\GitHub\EEG_epilepsia"
    # pickle_path = r"/Users/Jacobsen/Documents/GitHub/EEG_epilepsia" + "/"
    # pickle_path = r"/Users/philliphoejbjerg/Documents/GitHub/EEG_epilepsia" + "/"

    windowsOS = True

    """ Loading data - define which pickles to load (with NaNs or without) """
    X_file = r"\X_clean.npy"  # X_file = r"\X.npy"
    y_file = r"\y_clean.npy"  # y_file = r"\y.npy"
    ID_file = r"\ID_frame_clean.npy"  # ID_file = r"\ID_frame

    """ Define pipeline-class to use."""
    this_pipeline = pipeline(pickle_path, X_file, y_file, ID_file, windowsOS=windowsOS)

    """ Define model to be evaluated and filename:    
    >>> 'model'             should be the name of the function that should be called form the models.models.py script.
                            I.e. 'LR' for calling the Logistic Regression.
    
    >>> 'experiment'        is the folder in which the files will be created. 
    
    >>> 'experiment_name'   is the name ending of the name of the pickles created.
                            When doing Augmentation it should end with either _GAN / _Noise / _MixUp,
                            such that the following command can work properly:  
                            
                            if experiment_name.split("_")[-1] == 'GAN':
                            
                            When running the pipeline on a single artifact, one should manually write this in the
                            'experiment_name', i.e. experiment + model + "Null" + aug_method
                            
    >>> 'noise_experiment'  is the directory of the folder containing the noise files to be used. Should be None when
                            not experimenting with Noise Addition augmentation technique. """

    model = 'baseline_perm'
    aug_method = ""  # or '_Noise' or so.

    experiment = 'smote_f2'  # 'DataAug_color_noiseAdd_LR'
    experiment_name = experiment + "_" + model + aug_method  # "_DataAug_color_Noise" added to saving files. For augmentation end with "_Noise" or so.
    noise_experiment = None  # r"\whitenoise_covarOne" # r"\colornoise30Hz_covarOne" #

    """ Define ratios to use for SMOTE and data augmentation techniques !"""
    smote_ratios = np.array([0, 0.5, 1, 1.5, 2])
    aug_ratios = np.array([0, 0.5, 1, 1.5, 2])

    """ Specify other parameters"""
    HO_evals = 25
    K = 5
    random_state_val = 0

    # Obtaining y_true_file
    this_pipeline.runPipeline(model=model,
                              HO_evals=HO_evals,
                              smote_ratios=np.array([0]), aug_ratios=np.array([0]),
                              experiment=experiment,
                              experiment_name=experiment_name,
                              random_state=random_state_val,
                              K=K, save_y_true=True)


    # Example of normal run - with no smote and no augmentation. For illustration, 1-Fold CV.
    this_pipeline.runPipeline(model=model,
                              HO_evals=HO_evals,
                              smote_ratios=smote_ratios, aug_ratios=np.array([0]),
                              experiment=experiment,
                              experiment_name=experiment_name,
                              random_state=random_state_val,
                              K=K)

    # Example of running with MixUp and no SMOTE.
    aug_method = "_MixUp"
    experiment_name = experiment + "_" + model + aug_method

    this_pipeline.runPipeline(model=model,
                              HO_evals=HO_evals,
                              smote_ratios=np.array([0]), aug_ratios=np.array([0.5]),
                              experiment=experiment,
                              experiment_name=experiment_name,
                              random_state=random_state_val,
                              K=K)

    # Example of running on a single artifact.
    artifact = 'null'
    experiment_name = experiment + model + artifact + aug_method

    this_pipeline.runPipeline(model=model,
                              HO_evals=HO_evals,
                              smote_ratios=np.array([0]), aug_ratios=np.array([0]),
                              experiment=experiment,
                              experiment_name=experiment_name,
                              artifact_names=[artifact],
                              random_state=random_state_val,
                              K=K)
