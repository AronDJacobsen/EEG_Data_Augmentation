import scipy.stats
import numpy as np
from prepData.dataLoader import *
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import glob
import pandas as pd
from tqdm import tqdm
from models.models import *
import seaborn as sn
from sklearn.metrics import auc, multilabel_confusion_matrix, balanced_accuracy_score


class getResults:
    def __init__(self, dir, experiment, experiment_name, windowsOS=False, merged_file=False):

        print("-" * 80)
        self.dir = dir
        self.windowsOS = windowsOS
        self.slash = "\\" if self.windowsOS == True else "/"

        self.experiment = experiment
        self.pickle_path = (self.slash).join([dir, "results", "performance", experiment])
        self.merged_file = merged_file

        self.experiment_name = experiment_name
        self.basepath = self.slash.join(self.pickle_path.split(self.slash)[:-2])

        self.aug_ratios = [0, 0.5, 1, 1.5, 2]
        self.smote_ratios = [1, 1.5, 2, 2.5, 3]
        self.folds = [0, 1, 2, 3, 4]
        self.artifacts = ['eyem', 'chew', 'shiv', 'elpp', 'musc', 'null']
        self.models = ['baseline_perm', 'LR', 'GNB', 'RF', 'LDA', 'MLP', 'SGD']  # , 'KNN', 'AdaBoost'
        self.scores = ['y_pred', 'accuracy', 'weighted_F2', 'sensitivity']
        self.improvementExperiment = False

        print(f"Created object for experiment: {experiment_name}\n")

        super(getResults, self).__init__()

    def addSingleArtifact(self, dir_name, new_file_name="merged", experiment="", main_file="", sec_file=""):

        file_path = (self.slash).join(self.pickle_path.split(self.slash)[:-1]) + self.slash + dir_name

        # Find files in the directory. Check if the main file is the first one to be filled into the new dictionary.
        file_names = [results_file.split(self.slash)[-1] for results_file in glob.glob(file_path + self.slash + "**")]
        if file_names[0] != main_file:
            file_names = file_names[::-1]

        i = 0
        all_results_dict = {}

        # Extract information from the original pickles from the experiments and save them in a new nested dictionary.
        for model_file in file_names:
            results = LoadNumpyPickles(file_path + self.slash, model_file, windowsOS=self.windowsOS)[()]

            if i == 0:
                for aug_ratio in self.aug_ratios:
                    all_results_dict[aug_ratio] = defaultdict(dict)
                    for ratio in self.smote_ratios:
                        all_results_dict[aug_ratio][ratio] = defaultdict(dict)
                        for fold in self.folds:
                            all_results_dict[aug_ratio][ratio][fold] = defaultdict(dict)
                            for artifact in self.artifacts:
                                all_results_dict[aug_ratio][ratio][fold][artifact] = defaultdict(dict)
                                for model in self.models:
                                    all_results_dict[aug_ratio][ratio][fold][artifact][model] = defaultdict(dict)
                                    try:
                                        all_results_dict[aug_ratio][ratio][fold][artifact][model] = \
                                            results[aug_ratio][ratio][fold][artifact][model]
                                    except KeyError:
                                        pass
            else:
                for aug_ratio in self.aug_ratios:
                    for ratio in self.smote_ratios:
                        for fold in self.folds:
                            for artifact in self.artifacts:
                                for model in self.models:
                                    try:
                                        all_results_dict[aug_ratio][ratio][fold][artifact][model] = \
                                            results[aug_ratio][ratio][fold][artifact][model]
                                    except KeyError:
                                        pass

            # TODO: Right now it only works if you use it in the correct way - meaning only to change artifact of files

            # TODO: Maybe it is possible to use the mergeResultFiles-function for better overview but it requires
            # TODO: some more parameters for that function.

            i += 1

        # Save file in merged_files dir
        results_basepath = self.slash.join(file_path.split(self.slash)[:-2])

        exp = experiment
        save_path = (self.slash).join([results_basepath, "performance", exp])

        SaveNumpyPickles(save_path, self.slash + new_file_name, all_results_dict, self.windowsOS)
        print(f"\nNew file created ({new_file_name})! Saved to:\n {save_path}")

    def changePicklePath(self):

        if self.pickle_path == (self.slash).join([self.dir, "results", "performance", self.experiment]):
            self.pickle_path = (self.slash).join([self.dir, "results", "merged_files", self.experiment])

        else:
            self.pickle_path = (self.slash).join([self.dir, "results", "performance", self.experiment])

        print(f"\n ---> Pickle path changed to: {self.pickle_path}")

    def mergeResultFiles(self, file_name="merged", merge_smote_files=False, merge_aug_files=False, exclude_models=None,
                         merge_smote=False, model_name='AdaBoost', ensemble_files=[]):

        if self.merged_file == False:
            raise Exception("Merged file set to 'False'")
        else:
            if exclude_models is None:
                exclude_models = ['baseline_major']

            if ensemble_files != []:
                file_names = []
                pickle_paths = []
                for experiment in ensemble_files:
                    file_path = (self.slash).join([self.basepath, "performance", experiment, ""])
                    file_names.append([results_file.split(self.slash)[-1] for results_file in
                                       glob.glob(file_path + "**")])

                    path1 = (self.slash).join(self.pickle_path.split(self.slash)[:-1])
                    pickle_paths.append((self.slash).join([path1, experiment]))
                file_names = list(np.concatenate(file_names))
            else:
                # Find files to be merged
                file_path = (self.slash).join([self.basepath, "performance", self.experiment, ""])
                file_names = [results_file.split(self.slash)[-1] for results_file in
                              glob.glob(file_path + "**")]

            if len(file_names) == 0:
                raise FileNotFoundError("There are no files in the specified directory.")

            i = 0
            all_results_dict = {}

            # Extract information from the original pickles from the experiments and save them in a new nested dictionary.
            file_names.append(file_names[0])
            for model_idx, model_file in enumerate(file_names):

                if ensemble_files != []:
                    results = \
                    LoadNumpyPickles(pickle_paths[model_idx] + self.slash, model_file, windowsOS=self.windowsOS)[()]
                else:
                    results = LoadNumpyPickles(self.pickle_path + self.slash, model_file, windowsOS=self.windowsOS)[()]

                if merge_aug_files and merge_smote_files:
                    raise AttributeError("Only possible to merge one file-type at a time!")

                # Rereference the dictionary. The if-statements are created to ensure that the accumulated dictionary is not
                # overwritten when looping through either aug_ratios, smote_ratios, etc.

                """
                if merge_smote_files:
                    for aug_ratio in self.aug_ratios:
                        if i == 0:  # This line makes the difference
                            all_results_dict[aug_ratio] = defaultdict(dict)
                        for ratio in self.smote_ratios:
                            all_results_dict[aug_ratio][ratio] = defaultdict(dict)
                            for fold in self.folds:
                                all_results_dict[aug_ratio][ratio][fold] = defaultdict(dict)
                                for artifact in self.artifacts:
                                    all_results_dict[aug_ratio][ratio][fold][artifact] = defaultdict(dict)
                                    for model in self.models:
                                        all_results_dict[aug_ratio][ratio][fold][artifact][model] = defaultdict(dict)
                                        if model in exclude_models:
                                            pass
                                        else:
                                            try:
                                                all_results_dict[aug_ratio][ratio][fold][artifact][model] = \
                                                    results[aug_ratio][ratio][fold][artifact][model]
                                            except KeyError:
                                                pass

                elif merge_aug_files:
                    for aug_ratio in self.aug_ratios:  # Default dictionary is set from the start so no problem here
                        all_results_dict[aug_ratio] = defaultdict(dict)
                        for ratio in self.smote_ratios:
                            all_results_dict[aug_ratio][ratio] = defaultdict(dict)
                            for fold in self.folds:
                                all_results_dict[aug_ratio][ratio][fold] = defaultdict(dict)
                                for artifact in self.artifacts:
                                    all_results_dict[aug_ratio][ratio][fold][artifact] = defaultdict(dict)
                                    for model in self.models:
                                        all_results_dict[aug_ratio][ratio][fold][artifact][model] = defaultdict(dict)
                                        if model in exclude_models:
                                            pass
                                        else:
                                            all_results_dict[aug_ratio][ratio][fold][artifact][model] = \
                                                results[aug_ratio][ratio][fold][artifact][model]
                
                else:
                    if i == 0:  # If merging full files we will first create a dictionary and continue to
                        # the else-statement ...
                        for aug_ratio in self.aug_ratios:
                            all_results_dict[aug_ratio] = defaultdict(dict)
                            for ratio in self.smote_ratios:
                                all_results_dict[aug_ratio][ratio] = defaultdict(dict)
                                for fold in self.folds:
                                    all_results_dict[aug_ratio][ratio][fold] = defaultdict(dict)
                                    for artifact in self.artifacts:
                                        all_results_dict[aug_ratio][ratio][fold][artifact] = defaultdict(dict)
                                        for model in self.models:
                                            if model in exclude_models:
                                                pass
                                            else:
                                                try:
                                                    all_results_dict[aug_ratio][ratio][fold][artifact][model] = \
                                                        results[aug_ratio][ratio][fold][artifact][model]
                                                except KeyError:
                                                    pass

                    else:  # ... where we ensure that no keys get overwritten. Only functions when merging full model files, not smote
                        for aug_ratio in self.aug_ratios:
                            for ratio in self.smote_ratios:
                                for fold in self.folds:
                                    for artifact in self.artifacts:
                                        for model in self.models:
                                            try:
                                                all_results_dict[aug_ratio][ratio][fold][artifact][model] = \
                                                    results[aug_ratio][ratio][fold][artifact][model]
                                            except KeyError:
                                                pass
                    """
                if i == 0:  # If merging full files we will first create a dictionary and continue to
                    # the else-statement ...
                    for aug_ratio in self.aug_ratios:
                        all_results_dict[aug_ratio] = defaultdict(dict)
                        for ratio in self.smote_ratios:
                            all_results_dict[aug_ratio][ratio] = defaultdict(dict)
                            for fold in self.folds:
                                all_results_dict[aug_ratio][ratio][fold] = defaultdict(dict)
                                for artifact in self.artifacts:
                                    all_results_dict[aug_ratio][ratio][fold][artifact] = defaultdict(dict)
                                    for model in self.models:
                                        all_results_dict[aug_ratio][ratio][fold][artifact][model] = defaultdict(dict)
                                        try:
                                            scores = results[aug_ratio][ratio][fold][artifact][model]
                                            all_results_dict[aug_ratio][ratio][fold][artifact][model] = scores

                                        except KeyError:
                                            pass

                else:  # ... where we ensure that no keys get overwritten. Only functions when merging full model files, not smote
                    for aug_ratio in self.aug_ratios:
                        for ratio in self.smote_ratios:
                            for fold in self.folds:
                                for artifact in self.artifacts:
                                    for model in self.models:
                                        try:
                                            scores = results[aug_ratio][ratio][fold][artifact][model]
                                            all_results_dict[aug_ratio][ratio][fold][artifact][model] = scores

                                        except KeyError:
                                            pass
                i += 1

            # Save new file in merged_files dir
            results_basepath = self.slash.join(self.pickle_path.split(self.slash)[:-2])

            exp = self.pickle_path.split(self.slash)[-1]

            save_path = (self.slash).join([results_basepath, "merged_files", exp])
            os.makedirs(save_path, exist_ok=True)

            SaveNumpyPickles(save_path, self.slash + "results" + file_name, all_results_dict, self.windowsOS)
            print(f"\nNew file created ({'results' + file_name})! Saved to:\n {save_path}")

    def tableResults_Augmentation(self, experiment_name, y_true_path=None, smote_ratios=None, measure="sensitivity"):

        if smote_ratios == None:
            smote_ratios = [1]

        if measure == "balanced_acc":
            if y_true_path == None:
                raise AttributeError("Path to the pickle containing the true labels is not specified.")
            y_true_path_list = y_true_path.split("\\")

            results_y_true = LoadNumpyPickles(pickle_path=("\\").join(y_true_path_list[:-1]),
                                              file_name=self.slash + y_true_path_list[-1],
                                              windowsOS=self.windowsOS)
            results_y_true = results_y_true[()]
            y_true_dict = {}

            for artifact in self.artifacts[:6]:
                y_true_art = []
                for i in self.folds:
                    y_true_art.append(results_y_true[i][artifact]['y_true'])

                y_true_dict[artifact] = y_true_art

        # Specifies basepath
        results_basepath = self.slash.join(self.pickle_path.split(self.slash)[:-2])
        exp = self.pickle_path.split(self.slash)[-1]

        # Loads merged file or not.
        if self.merged_file:
            results_all = LoadNumpyPickles(
                pickle_path=(self.slash).join([results_basepath, "merged_files", exp]),
                file_name=self.slash + "results" + experiment_name + '.npy', windowsOS=self.windowsOS)
            results_all = results_all[()]
        else:
            results_all = LoadNumpyPickles(pickle_path=(self.slash).join([results_basepath, "performance", exp]),
                                           file_name=self.slash + "results" + experiment_name + '.npy',
                                           windowsOS=self.windowsOS)
            results_all = results_all[()]
            # fold, artifact, model, hyperopt iterations
            HO_trials = LoadNumpyPickles(pickle_path=(self.slash).join([results_basepath, "hyperopt", exp]),
                                         file_name=self.slash + 'ho_trials' + experiment_name + '.npy',
                                         windowsOS=self.windowsOS)
            HO_trials = HO_trials[()]

        table_aug_dict = defaultdict(dict)
        table_aug_std_dict = defaultdict(dict)

        for aug_ratio in results_all:
            results = results_all[aug_ratio]

            pd.set_option("display.max_rows", None, "display.max_columns", None)

            # want to average each fold

            # row-wise models, column-wise artifacts
            acc = np.zeros((len(self.models), len(self.artifacts)))
            acc_std = np.zeros((len(self.models), len(self.artifacts)))
            f2s = np.zeros((len(self.models), len(self.artifacts)))
            f2s_std = np.zeros((len(self.models), len(self.artifacts)))

            table_smote_dict = defaultdict(dict)
            table_smote_std_dict = defaultdict(dict)

            for smote_ratio in smote_ratios:
                table = defaultdict(dict)  # {}
                table_std = defaultdict(dict)  # {}

                for idx_art, artifact in enumerate(self.artifacts):
                    # table[artifact] = []
                    store_model = [0] * len(self.models)
                    measure_std = [0] * len(self.models)

                    for idx_mod, model in enumerate(self.models):

                        store_scores = []

                        temp_acc = []
                        temp_f2 = []
                        for fold in self.folds:
                            if results[smote_ratio][fold][artifact][model] == {}:
                                store_scores.append(np.nan)
                                temp_acc.append(np.nan)
                                temp_f2.append(np.nan)
                                empty = True

                            else:
                                if measure == "balanced_acc":
                                    # TODO: HER!
                                    actual = y_true_dict[artifact][fold]
                                    predictions = results[smote_ratio][fold][artifact][model]['y_pred']
                                    ba = balanced_accuracy_score(y_true=actual, y_pred=predictions)

                                    store_scores.append(ba)
                                    temp_acc.append(results[smote_ratio][fold][artifact][model]['accuracy'])
                                    temp_f2.append(ba)  # Not f2 - naming is incorrect but calculation is good!
                                    empty = False

                                else:
                                    store_scores.append(results[smote_ratio][fold][artifact][model][measure])
                                    temp_acc.append(results[smote_ratio][fold][artifact][model]['accuracy'])
                                    temp_f2.append(
                                        results[smote_ratio][fold][artifact][model][measure])  # 'weighted_F2'])
                                    empty = False

                        acc[idx_mod, idx_art] = np.mean(temp_acc)
                        f2s[idx_mod, idx_art] = np.mean(temp_f2)

                        # Standard deviation for acc. and F2 on each artifact
                        acc_std[idx_mod, idx_art] = np.std(temp_acc)
                        f2s_std[idx_mod, idx_art] = np.std(temp_f2)

                        # Store standard deviation for sensitivies for each classifier/ model
                        store_model[idx_mod] = np.mean(store_scores)
                        measure_std[idx_mod] = np.std(store_scores)

                    table[artifact] = store_model
                    table_std[artifact] = measure_std

                table['avg. accuracy'] = np.mean(acc, axis=1)
                table[f'avg. {measure}'] = np.mean(f2s, axis=1)

                # Mean standard deviation
                table_std['avg. accuracy'] = np.mean(acc_std, axis=1)
                table_std[f'avg. {measure}'] = np.mean(f2s_std, axis=1)

                table_smote_dict[smote_ratio] = table
                table_smote_std_dict[smote_ratio] = table_std

                table_aug_dict[aug_ratio][smote_ratio] = table
                table_aug_std_dict[aug_ratio][smote_ratio] = table_std

        return table_aug_dict, table_aug_std_dict

    def plotResultsAugmentation(self, performances_dict, errors_dict, experiment, aug_technique, measure="sensitiviy",
                                save_img=False):

        save_path = (self.slash).join([self.dir, "Plots", experiment])

        colorlist = ["lightslategrey", "lightsteelblue", "darkcyan", "firebrick", "lightcoral"]
        # Plotting results
        art = len(self.artifacts)

        for indv_art, name in enumerate(self.artifacts):
            for j, smote_ratio in enumerate(self.smote_ratios):
                for i, aug_ratio in enumerate(self.aug_ratios):
                    performance_dict = performances_dict[aug_ratio][smote_ratio]
                    error_dict = errors_dict[aug_ratio][smote_ratio]

                    performance_vals = np.array(list(performance_dict.values())[:art])
                    error_vals = np.array(list(error_dict.values())[:art])

                    X_axis = np.arange(len(self.models)) - 0.3
                    plt.bar(x=X_axis + 0.15 * i, height=performance_vals[indv_art, :], width=0.15, color=colorlist[i],
                            label="Aug. ratio = " + str(self.aug_ratios[i]))

                    plt.errorbar(x=X_axis + 0.15 * i, y=performance_vals[indv_art, :], yerr=error_vals[indv_art, :],
                                 fmt='.', color='k')

                models = self.models
                """
                for i, model in enumerate(models):
                    if model == 'baseline_perm':
                        models[i] = "base-\nline"
                    elif model == 'AdaBoost':
                        models[i] = "Ada-\nBoost"
                """

                plt.xticks(np.arange(len(models)), models, rotation=60)
                plt.ylim(0, 1)

                if aug_technique == None:
                    plt.title(
                        f"{measure} with {aug_technique} augmentation on the '{name}'-class - SMOTE = {self.smote_ratios[j] - 1}")
                else:
                    plt.title(
                        f"{measure} with {aug_technique} augmentation on the '{name}'-class - SMOTE = {self.smote_ratios[j] - 1}")

                plt.xlabel("Model")
                plt.ylabel(measure)
                plt.legend(loc='center right', bbox_to_anchor=(1.36, 0.5))
                plt.subplots_adjust(bottom=0.2, right=0.775)

                if save_img:
                    img_path = f"{(self.slash).join([save_path, measure])}{self.slash}{name}_aug{aug_technique}_smote{self.smote_ratios[j] - 1}.png"
                    os.makedirs((self.slash).join(img_path.split(self.slash)[:-1]), exist_ok=True)
                    plt.savefig(img_path)

                plt.show()

    def plotResultsHelper(self, performances_dict, errors_dict, experiment, smote_ratios=None, aug_ratios=None,
                          aug_technique=None, measure="sensitiviy",
                          across_SMOTE=True, save_img=False):

        if smote_ratios == None:
            smote_ratios = self.smote_ratios
        if aug_ratios == None:
            aug_ratios = self.aug_ratios

        save_path = (self.slash).join([self.dir, "Plots", experiment])

        number = 5
        cmap = plt.get_cmap('coolwarm')
        colorlist = [cmap(i) for i in np.linspace(0, 1, number)]

        # colorlist = ["lightslategrey", "lightsteelblue", "darkcyan", "firebrick", "lightcoral"]
        # Plotting results
        art = len(self.artifacts)

        if across_SMOTE:
            for indv_art, name in enumerate(self.artifacts):
                for j, aug_ratio in enumerate(aug_ratios):
                    for i, smote_ratio in enumerate(smote_ratios):
                        performance_dict = performances_dict[aug_ratio][smote_ratio]
                        error_dict = errors_dict[aug_ratio][smote_ratio]

                        performance_vals = np.array(list(performance_dict.values())[:art])
                        error_vals = np.array(list(error_dict.values())[:art])

                        w = 0.15
                        w = 0.75 / len(smote_ratios)
                        if len(smote_ratios) == 1:
                            X_axis = np.arange(len(self.models))
                        else:
                            X_axis = np.arange(len(self.models)) - 0.3
                        plt.bar(x=X_axis + w * i,
                                height=performance_vals[indv_art, :],
                                width=w,
                                color=colorlist[i],
                                label="SMOTE-ratio = " + str(smote_ratios[i] - 1))

                        plt.errorbar(x=X_axis + w * i, y=performance_vals[indv_art, :], yerr=error_vals[indv_art, :],
                                     fmt='.', color='k')

                    models = self.models
                    """
                    for i, model in enumerate(models):
                        if model == 'baseline_perm':
                            models[i] = "base-\nline"
                        elif model == 'AdaBoost':
                            models[i] = "Ada-\nBoost"
                    """

                    plt.xticks(np.arange(len(models)), models, rotation=-15)
                    plt.ylim(0, 1)

                    if aug_technique == None:
                        plt.title(
                            f"No augmentation, the '{name}'-class")
                    else:
                        plt.title(
                            f"{aug_technique} augmentation, the '{name}'-class - Aug. ratio = {aug_ratios[j]}")

                    plt.xlabel("Model")
                    plt.ylabel(measure)
                    if len(smote_ratios) > 1:
                        plt.legend(loc='center right', bbox_to_anchor=(1.36, 0.5))
                        plt.subplots_adjust(bottom=0.2, right=0.775)

                    if save_img:
                        img_path = f"{(self.slash).join([save_path, measure])}{self.slash}{name}_aug{aug_technique}_augRatio{aug_ratios[j]}.png"
                        os.makedirs((self.slash).join(img_path.split(self.slash)[:-1]), exist_ok=True)
                        plt.savefig(img_path)
                    plt.show()

        if self.improvementExperiment:
            best = np.zeros((len(self.artifacts), len(self.models)))
            best_errors = np.zeros((len(self.artifacts), len(self.models)))
            best_augRatios = pd.DataFrame(100 * np.ones((len(self.artifacts), len(self.models))), index=self.artifacts,
                                          columns=self.models)

            number = len(self.models)
            cmap = plt.get_cmap('coolwarm')
            colorlist = [cmap(i) for i in np.linspace(0, 1, number)]

            if aug_ratios[0] == 0:
                aug_ratios = aug_ratios[1:]

            for j, smote_ratio in enumerate(smote_ratios):
                for indv_model, name in enumerate(self.models):
                    scoresOverAugModel = np.zeros((len(self.artifacts), len(aug_ratios)))
                    errorsOverAugModel = np.zeros((len(self.artifacts), len(aug_ratios)))

                    for indv_art, artif in enumerate(self.artifacts):
                        for i, aug_ratio in enumerate(aug_ratios):
                            performance_dict = performances_dict[aug_ratio][smote_ratio]
                            error_dict = errors_dict[aug_ratio][smote_ratio]

                            performance_vals = np.array(list(performance_dict.values())[:art])
                            error_vals = np.array(list(error_dict.values())[:art])

                            scoresOverAugModel[indv_art, i] = performance_vals[indv_art, indv_model]
                            errorsOverAugModel[indv_art, i] = error_vals[indv_art, indv_model]

                    maxPos = np.argmax(scoresOverAugModel, axis=1)
                    maxVal = np.max(scoresOverAugModel, axis=1)
                    maxValError = [errorsOverAugModel[pos, maxPos[pos]] for pos in range(len(self.artifacts))]

                    best[:, indv_model] = maxVal
                    best_errors[:, indv_model] = maxValError
                    best_augRatios.iloc[:, indv_model] = [aug_ratios[maxPos[pos]] for pos in range(len(self.artifacts))]

                    w = 0.15
                    w = 0.75 / len(self.models)

                    X_axis = np.arange(len(self.artifacts)) - 0.33
                    plt.bar(x=X_axis + w * indv_model,
                            height=best[:, indv_model],
                            width=w,
                            color=colorlist[indv_model],
                            label=f"{name}")

                    plt.errorbar(x=X_axis + w * indv_model, y=best[:, indv_model],
                                 yerr=best_errors[:, indv_model],
                                 fmt='.', color='k')

                    models = self.models

                # ax1.set_xticks(np.arange(len(self.artifacts)), self.artifacts)#, rotation=-15)
                plt.ylim(0, 1)
                plt.xticks(np.arange(len(self.artifacts)), labels=self.artifacts)
                if measure == "balanced_acc":
                    measure = "Balanced acc."
                plt.title(f"{aug_technique}, {measure}")

                plt.xlabel("Artifact")
                plt.ylabel(measure)
                if save_img:
                    img_path = f"{(self.slash).join([save_path, measure])}{self.slash}ImprovementExperiment.png"
                    os.makedirs((self.slash).join(img_path.split(self.slash)[:-1]), exist_ok=True)
                    plt.savefig(img_path)
                plt.legend(loc='center right', bbox_to_anchor=(1.36, 0.5))
                plt.subplots_adjust(bottom=0.2, right=0.775)
                plt.show()
                # ax1.subplots_adjust(bottom=0.2, right=0.75)
            print("Best augmentation ratios (excluding 0)")
            print(best_augRatios.to_latex())

            LaTeX = True

            # Print dataframes
            df_eval = pd.DataFrame(best).T

            if measure == 'weighted_F2':
                df_eval = np.round(df_eval, 2)  # * 100
            else:
                df_eval = np.round(df_eval, 4) * 100

            df_latex = df_eval.to_latex()
            print(df_latex)

            df_eval_std = pd.DataFrame(best_errors).T

            if measure == 'weighted_F2':
                df_eval_std = np.round(df_eval_std, 2)  # * 100
            else:
                df_eval_std = np.round(df_eval_std, 4) * 100

            if LaTeX:
                column_things = self.artifacts.copy()
                column_things.append(f'Weighted {measure}')
                df = np.empty((len(self.models), len(column_things)))
                df = pd.DataFrame(df, index=self.models, columns=column_things)
                for i in range(len(self.models)):
                    for j, artifact in enumerate(column_things):
                        if artifact == f"Weighted {measure}":

                            df.iloc[i, j] = str(
                                f"{np.round(np.mean(df_eval.iloc[i, :6]), 2)} $\pm$ {np.round(np.mean(df_eval_std.iloc[i, :6]), 2)}")
                        else:
                            df.iloc[i, j] = str(
                                f"{np.round(df_eval.iloc[i, j], 2)} $\pm$ {np.round(df_eval_std.iloc[i, j], 2)}")

                df_latex = df.to_latex()
                print(df_latex)

            print("")
            print(100 * "#")

        if across_SMOTE == False and self.improvementExperiment == False:
            for indv_art, name in enumerate(self.artifacts):
                for j, smote_ratio in enumerate(smote_ratios):
                    for i, aug_ratio in enumerate(aug_ratios):
                        performance_dict = performances_dict[aug_ratio][smote_ratio]
                        error_dict = errors_dict[aug_ratio][smote_ratio]

                        performance_vals = np.array(list(performance_dict.values())[:art])
                        error_vals = np.array(list(error_dict.values())[:art])

                        X_axis = np.arange(len(self.models)) - 0.3
                        plt.bar(x=X_axis + 0.15 * i, height=performance_vals[indv_art, :], width=0.15,
                                color=colorlist[i],
                                label="Aug. ratio = " + str(aug_ratios[i]))

                        plt.errorbar(x=X_axis + 0.15 * i, y=performance_vals[indv_art, :], yerr=error_vals[indv_art, :],
                                     fmt='.', color='k')

                    models = self.models
                    """
                    for i, model in enumerate(models):
                        if model == 'baseline_perm':
                            models[i] = "base-\nline"
                        elif model == 'AdaBoost':
                            models[i] = "Ada-\nBoost"
                    """

                    plt.xticks(np.arange(len(models)), models, rotation=60)
                    plt.ylim(0, 1)

                    if aug_technique == None:
                        plt.title(
                            f"No augmentation, the '{name}'-class - SMOTE = {smote_ratios[j] - 1}")
                    else:
                        plt.title(
                            f"{aug_technique} augmentation, the '{name}'-class - SMOTE = {smote_ratios[j] - 1}")

                    plt.xlabel("Model")
                    plt.ylabel(measure)
                    plt.legend(loc='center right', bbox_to_anchor=(1.36, 0.5))
                    plt.subplots_adjust(bottom=0.2, right=0.775)

                    if save_img:
                        img_path = f"{(self.slash).join([save_path, measure])}{self.slash}{name}_aug{aug_technique}_smote{smote_ratios[j] - 1}.png"
                        os.makedirs((self.slash).join(img_path.split(self.slash)[:-1]), exist_ok=True)
                        plt.savefig(img_path)
                    plt.show()

    def plotResults(self, experiment_name, aug_technique, y_true_path=None, smote_ratios=None, aug_ratios=None,
                    measure='sensitivity',
                    across_SMOTE=True, save_img=False):

        if smote_ratios == None:
            smote_ratios = self.smote_ratios
        if aug_ratios == None:
            aug_ratios = self.aug_ratios

        # Loading statistically calculated results as dictionaries
        # For single files and their HO_trials
        # List of dictionaries of results. Each entry in the list is a results dictionary for one SMOTE ratio
        performances, errors = self.tableResults_Augmentation(measure=measure, y_true_path=y_true_path,
                                                              smote_ratios=smote_ratios,
                                                              experiment_name=experiment_name)
        self.smote_ratios.sort()

        # This function will plot results created in the augmentation experiment (with aug. ratio key in the dict)
        self.plotResultsHelper(performances_dict=performances, errors_dict=errors,
                               experiment=self.experiment, smote_ratios=smote_ratios,
                               aug_ratios=aug_ratios, across_SMOTE=across_SMOTE,
                               save_img=save_img, aug_technique=aug_technique,
                               measure=measure)

    def plotResultsHelperPlain(self, performances_F2, errors_F2, performances_sens, errors_sens, experiment,
                               smote_ratio=None, aug_ratio=None, measure="sensitiviy",
                               across_SMOTE=True, save_img=False):

        if smote_ratio == None:
            smote_ratio = 1
        if aug_ratio == None:
            aug_ratio = 0

        save_path = (self.slash).join([self.dir, "Plots", "plain_experiment"])

        number = 7
        cmap = plt.get_cmap('coolwarm')
        colorlist = [cmap(i) for i in np.linspace(0, 1, number)]
        # colorlist = ["royalblue", "lightslategrey", "lightsteelblue", "darkcyan", "lightcoral", "firebrick", "darkorange", "darkkhaki", "olive"]

        # Plotting results
        art = len(self.artifacts)

        if across_SMOTE:

            fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(10, 5))

            measure = "sensitivity"
            for indv_model, name in enumerate(self.models):
                # for i, artifact in enumerate(self.artifacts):
                performance_dict = performances_F2[aug_ratio][smote_ratio]
                error_dict = errors_F2[aug_ratio][smote_ratio]

                performance_vals = np.array(list(performance_dict.values())[:art])
                error_vals = np.array(list(error_dict.values())[:art])

                w = 0.15
                w = 0.75 / len(self.models)

                X_axis = np.arange(len(self.artifacts)) - 0.33
                ax1.bar(x=X_axis + w * indv_model,
                        height=performance_vals[:, indv_model],
                        width=w,
                        color=colorlist[indv_model],
                        label=name)

                ax1.errorbar(x=X_axis + w * indv_model, y=performance_vals[:, indv_model],
                             yerr=error_vals[:, indv_model],
                             fmt='.', color='k')

                models = self.models
                """
                for i, model in enumerate(models):
                    if model == 'baseline_perm':
                        models[i] = "base-\nline"
                    elif model == 'AdaBoost':
                        models[i] = "Ada-\nBoost"
                """

            # ax1.set_xticks(np.arange(len(self.artifacts)), self.artifacts)#, rotation=-15)
            ax1.set_ylim(0, 1)
            ax1.set_xticks(np.arange(len(self.artifacts)))
            ax1.set_xticklabels(self.artifacts)
            if measure == "balanced_acc":
                measure = "Balanced acc."
            ax1.set_title(measure)

            ax1.set_xlabel("Artifact")
            ax1.set_ylabel(measure)
            # ax1.subplots_adjust(bottom=0.2, right=0.75)

            measure = 'balanced_acc'
            for indv_model, name in enumerate(self.models):
                # for i, artifact in enumerate(self.artifacts):
                performance_dict = performances_sens[aug_ratio][smote_ratio]
                error_dict = errors_sens[aug_ratio][smote_ratio]

                performance_vals = np.array(list(performance_dict.values())[:art])
                error_vals = np.array(list(error_dict.values())[:art])

                w = 0.15
                w = 0.75 / len(self.models)

                X_axis = np.arange(len(self.artifacts)) - 0.33
                ax2.bar(x=X_axis + w * indv_model,
                        height=performance_vals[:, indv_model],
                        width=w,
                        color=colorlist[indv_model],
                        label=name)

                ax2.errorbar(x=X_axis + w * indv_model, y=performance_vals[:, indv_model],
                             yerr=error_vals[:, indv_model],
                             fmt='.', color='k')

                models = self.models
                """
                for i, model in enumerate(models):
                    if model == 'baseline_perm':
                        models[i] = "base-\nline"
                    elif model == 'AdaBoost':
                        models[i] = "Ada-\nBoost"
                """

            # ax2.set_xticks(np.arange(len(self.artifacts)), self.artifacts)#, rotation=-15)
            ax2.set_ylim(0, 1)
            ax2.set_xticks(np.arange(len(self.artifacts)))
            ax2.set_xticklabels(self.artifacts)
            # ax2.set_yticks()

            if measure == "weighted_F2":
                measure = "F2"
            if measure == "balanced_acc":
                measure = "Balanced acc."

            ax2.set_title(measure)

            ax2.set_xlabel("Artifact")
            ax2.set_ylabel(measure)
            plt.subplots_adjust(right=0.84)

            plt.legend(loc='center right', bbox_to_anchor=(1.5, 0.5))

            if save_img:
                img_path = f"{save_path}{self.slash}plain_experiment.png"
                os.makedirs((self.slash).join(img_path.split(self.slash)[:-1]), exist_ok=True)
                fig.savefig(img_path)

            # plt.setp(((ax1, ax2)), xticks=np.arange(len(self.artifacts)), xticklabels=self.artifacts)
            plt.show()

    def plotResultsPlainExp(self, experiment_name, y_true_path=None, smote_ratio=None, aug_ratio=None,
                            across_SMOTE=True, save_img=False):

        if smote_ratio == None:
            smote_ratio = 1
        if aug_ratio == None:
            aug_ratio = 0

        # Loading statistically calculated results as dictionaries
        # For single files and their HO_trials
        # List of dictionaries of results. Each entry in the list is a results dictionary for one SMOTE ratio
        performances_F2, errors_F2 = self.tableResults_Augmentation(measure="sensitivity", y_true_path=y_true_path,
                                                                    smote_ratios=[smote_ratio],
                                                                    experiment_name=experiment_name)
        performances_sens, errors_sens = self.tableResults_Augmentation(measure="balanced_acc", y_true_path=y_true_path,
                                                                        smote_ratios=[smote_ratio],
                                                                        experiment_name=experiment_name)
        self.smote_ratios.sort()

        # This function will plot results created in the augmentation experiment (with aug. ratio key in the dict)
        self.plotResultsHelperPlain(performances_F2=performances_F2, errors_F2=errors_F2,
                                    performances_sens=performances_sens, errors_sens=errors_sens,
                                    experiment=self.experiment, smote_ratio=smote_ratio,
                                    aug_ratio=aug_ratio, across_SMOTE=across_SMOTE,
                                    save_img=save_img)

    def plotResultsHelperImprovement(self, performances_bacc, errors_bacc, performances_F2, errors_F2, bacc_control,
                                     sens_control, sens_std_control, bacc_std_control,
                                     experiment, smote_ratio=None, aug_technique=None, aug_ratios=None,
                                     measure="sensitiviy",
                                     across_SMOTE=True, save_img=False):

        if smote_ratio == None:
            smote_ratio = 1
        if aug_ratios == None:
            aug_ratios = self.aug_ratios

        # save_path = (self.slash).join([self.dir, "Plots", "plain_experiment"])

        best = np.zeros((len(self.artifacts), len(self.models)))
        best_errors = np.zeros((len(self.artifacts), len(self.models)))
        best_augRatios = pd.DataFrame(100 * np.ones((len(self.artifacts), len(self.models))))

        number = len(self.models)
        cmap = plt.get_cmap('coolwarm')
        colorlist = [cmap(i) for i in np.linspace(0, 1, number)]
        art = 6

        fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(10, 5))

        measure = "sensitivity"

        for j, smote_ratio in enumerate([smote_ratio]):
            for indv_model, name in enumerate(self.models):
                scoresOverAugModel = np.zeros((len(self.artifacts), len(self.aug_ratios)))
                errorsOverAugModel = np.zeros((len(self.artifacts), len(self.aug_ratios)))

                for indv_art, artif in enumerate(self.artifacts):
                    for i, aug_ratio in enumerate(aug_ratios):
                        performance_dict = performances_bacc[aug_ratio][smote_ratio]
                        error_dict = errors_bacc[aug_ratio][smote_ratio]

                        performance_vals = np.array(list(performance_dict.values())[:art])
                        error_vals = np.array(list(error_dict.values())[:art])

                        scoresOverAugModel[indv_art, i] = performance_vals[indv_art, indv_model]
                        errorsOverAugModel[indv_art, i] = error_vals[indv_art, indv_model]

                maxPos = np.argmax(scoresOverAugModel, axis=1)
                maxVal = np.max(scoresOverAugModel, axis=1)
                maxValError = [errorsOverAugModel[pos, maxPos[pos]] for pos in range(len(self.artifacts))]

                best[:, indv_model] = maxVal
                best_errors[:, indv_model] = maxValError
                best_augRatios.iloc[:, indv_model] = [aug_ratios[maxPos[pos]] for pos in range(len(self.artifacts))]

                w = 0.15
                w = 0.75 / len(self.models)

                X_axis = np.arange(len(self.artifacts)) - 0.33
                ax1.bar(x=X_axis + w * indv_model,
                        height=best[:, indv_model] - sens_control[indv_art],
                        width=w,
                        color=colorlist[indv_model],
                        label=f"{name}")

                pooledError = np.sqrt((best_errors[:, indv_model]**2 + sens_std_control[indv_art]**2) / 2)

                ax1.errorbar(x=X_axis + w * indv_model, y=best[:, indv_model] - sens_control[indv_art],
                             yerr=pooledError, #best_errors[:, indv_model],
                             fmt='.', color='k')

                models = self.models

            # ax1.set_xticks(np.arange(len(self.artifacts)), self.artifacts)#, rotation=-15)
            if np.all(sens_control == 0):
                ax1.set_ylim(0, 1)
            else:
                ax1.set_ylim(-1, 1)
            ax1.set_xticks(np.arange(len(self.artifacts)))
            ax1.set_xticklabels(self.artifacts)

            if measure == "balanced_acc":
                measure = "Balanced acc."
            if measure == "weighted_F2":
                measure = "F2"
            ax1.set_title(f"{aug_technique}, {measure}")

            ax1.set_xlabel("Artifact")
            ax1.set_ylabel(measure)

            measure = "balanced_acc"

            for j, smote_ratio in enumerate([smote_ratio]):
                for indv_model, name in enumerate(self.models):
                    scoresOverAugModel = np.zeros((len(self.artifacts), len(self.aug_ratios)))
                    errorsOverAugModel = np.zeros((len(self.artifacts), len(self.aug_ratios)))

                    for indv_art, artif in enumerate(self.artifacts):
                        for i, aug_ratio in enumerate(aug_ratios):
                            performance_dict = performances_F2[aug_ratio][smote_ratio]
                            error_dict = errors_F2[aug_ratio][smote_ratio]

                            performance_vals = np.array(list(performance_dict.values())[:art])
                            error_vals = np.array(list(error_dict.values())[:art])

                            scoresOverAugModel[indv_art, i] = performance_vals[indv_art, indv_model]
                            errorsOverAugModel[indv_art, i] = error_vals[indv_art, indv_model]

                    maxPos = np.argmax(scoresOverAugModel, axis=1)
                    maxVal = np.max(scoresOverAugModel, axis=1)
                    maxValError = [errorsOverAugModel[pos, maxPos[pos]] for pos in range(len(self.artifacts))]

                    best[:, indv_model] = maxVal
                    best_errors[:, indv_model] = maxValError
                    best_augRatios.iloc[:, indv_model] = [aug_ratios[maxPos[pos]] for pos in range(len(self.artifacts))]

                    w = 0.15
                    w = 0.75 / len(self.models)

                    X_axis = np.arange(len(self.artifacts)) - 0.33
                    ax2.bar(x=X_axis + w * indv_model,
                            height=best[:, indv_model] - bacc_control[indv_art],
                            width=w,
                            color=colorlist[indv_model],
                            label=f"{name}")

                    pooledError = np.sqrt((best_errors[:, indv_model] ** 2 + bacc_std_control[indv_art] ** 2) / 2)

                    ax2.errorbar(x=X_axis + w * indv_model, y=best[:, indv_model] - bacc_control[indv_art],
                                 yerr=pooledError, #best_errors[:, indv_model],
                                 fmt='.', color='k')

                    models = self.models

                # ax1.set_xticks(np.arange(len(self.artifacts)), self.artifacts)#, rotation=-15)
                if np.all(bacc_control == 0):
                    ax2.set_ylim(0, 1)
                else:
                    ax2.set_ylim(-1, 1)
                ax2.set_xticks(np.arange(len(self.artifacts)))
                ax2.set_xticklabels(self.artifacts)

                if measure == "balanced_acc":
                    measure = "Balanced acc."
                if measure == "weighted_F2":
                    measure = "F2"
                ax2.set_title(f"{aug_technique}, {measure}")

                ax2.set_xlabel("Artifact")
                ax2.set_ylabel(measure)

            plt.subplots_adjust(right=0.84)

            plt.legend(loc='center right', bbox_to_anchor=(1.5, 0.5))

            save_path = (self.slash).join([self.dir, "Plots", self.experiment])
            if save_img:
                img_path = f"{save_path}{self.slash}ImprovementResults_{self.experiment}.png"
                os.makedirs((self.slash).join(img_path.split(self.slash)[:-1]), exist_ok=True)
                fig.savefig(img_path)
            plt.show()
            # ax1.subplots_adjust(bottom=0.2, right=0.75)

        print("")
        print(pd.DataFrame(best_augRatios))

    def plotResultsImprovementExp(self, experiment_name, y_true_path=None, smote_ratio=None, aug_ratios=None,
                                  across_SMOTE=True, save_img=False, aug_technique=None, bacc_control=None,
                                  sens_std_control=None, bacc_std_control=None,
                                  sens_control=None):

        if bacc_control == None:
            bacc_control = np.zeros(len(self.artifacts))
            bacc_std_control = np.zeros(len(self.artifacts))
        if sens_control == None:
            sens_control = np.zeros(len(self.artifacts))
            sens_std_control = np.zeros(len(self.artifacts))

        if smote_ratio == None:
            smote_ratio = 1
        if aug_ratios == None:
            aug_ratio = self.aug_ratios

        # Loading statistically calculated results as dictionaries
        # For single files and their HO_trials
        # List of dictionaries of results. Each entry in the list is a results dictionary for one SMOTE ratio
        performances_bacc, errors_bacc = self.tableResults_Augmentation(measure="sensitivity", y_true_path=y_true_path,
                                                                        smote_ratios=[smote_ratio],
                                                                        experiment_name=experiment_name)
        #TODO: Ændre til balanced_acc
        performances_F2, errors_F2 = self.tableResults_Augmentation(measure="sensitivity", y_true_path=y_true_path,
                                                                    smote_ratios=[smote_ratio],
                                                                    experiment_name=experiment_name)
        self.smote_ratios.sort()

        # This function will plot results created in the augmentation experiment (with aug. ratio key in the dict)
        self.plotResultsHelperImprovement(performances_bacc=performances_bacc, errors_bacc=errors_bacc,
                                          performances_F2=performances_F2, errors_F2=errors_F2,
                                          experiment=self.experiment, smote_ratio=smote_ratio,
                                          aug_ratios=aug_ratios, across_SMOTE=across_SMOTE,
                                          aug_technique=aug_technique,
                                          bacc_control=bacc_control, bacc_std_control=bacc_std_control,
                                          sens_control=sens_control, sens_std_control=sens_std_control,
                                          save_img=save_img)

    def printResults(self, experiment_name, y_true_path=None, smote_ratios=None, aug_ratios=None, measure='sensitivity',
                     printSTDTable=False, LaTeX=False, across_SMOTE=True):

        if smote_ratios == None:
            smote_ratios = self.smote_ratios
        if aug_ratios == None:
            aug_ratios = self.aug_ratios

        performances, errors = self.tableResults_Augmentation(measure=measure, y_true_path=y_true_path,
                                                              smote_ratios=smote_ratios,
                                                              experiment_name=experiment_name)
        self.smote_ratios.sort()

        if across_SMOTE:
            for j, aug_ratio in enumerate(aug_ratios):
                print("\n\n")
                print(80 * "#")
                print("{} scores with augmentation rate set to {:2f}".format(measure, self.aug_ratios[j]))
                print(100 * "#")

                performance_dict = performances[aug_ratio]
                errors_dict = errors[aug_ratio]

                for i, ratio in enumerate(smote_ratios):

                    performance = performance_dict[ratio]
                    error = errors_dict[ratio]

                    # Print dataframes
                    df_eval = pd.DataFrame.from_dict(performance)
                    df_eval.index = self.models
                    if measure == 'weighted_F2':
                        df_eval = np.round(df_eval, 2)  # * 100
                    else:
                        df_eval = np.round(df_eval, 4) * 100

                    if LaTeX:
                        df_latex = df_eval.to_latex()
                        print(df_latex)

                    else:
                        print('\nOVERALL PERFORMANCE')
                        print("SMOTE RATIO:" + str(ratio - 1) + "\n")
                        print(df_eval.to_string())
                        # print(df_eval)

                    if printSTDTable:
                        df_eval_std = pd.DataFrame.from_dict(error)
                        df_eval_std.index = self.models

                        if measure == 'weighted_F2':
                            df_eval_std = np.round(df_eval_std, 2)  # * 100
                        else:
                            df_eval_std = np.round(df_eval_std, 4) * 100

                        if LaTeX:
                            column_things = self.artifacts.copy()
                            column_things.append(f'Weighted {measure}')
                            df = np.empty((len(self.models), len(column_things)))
                            df = pd.DataFrame(df, index=self.models, columns=column_things)
                            for i in range(len(self.models)):
                                for j, artifact in enumerate(column_things):
                                    if artifact == f"Weighted {measure}":

                                        df.iloc[i, j] = str(
                                            f"{np.round(np.mean(df_eval.iloc[i, :6]), 2)} $\pm$ {np.round(np.mean(df_eval_std.iloc[i, :6]), 2)}")
                                    else:
                                        df.iloc[i, j] = str(
                                            f"{np.round(df_eval.iloc[i, j], 2)} $\pm$ {np.round(df_eval_std.iloc[i, j], 2)}")

                            df_latex = df.to_latex()
                            print(df_latex)

                        else:
                            print('\nSTANDARD DEVIATIONS')
                            print("SMOTE ratio:" + str(smote_ratio) + "\n")
                            print(df_eval_std.to_string())

                    print("")
                    print(100 * "#")
        else:
            for j, aug_ratio in enumerate(aug_ratios):
                print("\n\n")
                print(80 * "#")
                print("{} scores with smote rate set to {:2f}".format(measure, self.smote_ratios[j] - 1))
                print(100 * "#")

                performance_dict = performances[aug_ratio]
                errors_dict = errors[aug_ratio]

                for i, ratio in enumerate(smote_ratios):

                    performance = performance_dict[ratio]
                    error = errors_dict[ratio]

                    # Print dataframes
                    df_eval = pd.DataFrame.from_dict(performance)
                    df_eval.index = self.models

                    if measure == 'weighted_F2':
                        df_eval = np.round(df_eval, 2)  # * 100
                    else:
                        df_eval = np.round(df_eval, 4) * 100

                    if LaTeX:
                        df_latex = df_eval.to_latex()
                        print(df_latex)

                    else:
                        print('\nOVERALL PERFORMANCE')
                        print("Aug. ratio:" + str(aug_ratio) + "\n")
                        print(df_eval.to_string())
                        # print(df_eval)

                    if printSTDTable:
                        df_eval_std = pd.DataFrame.from_dict(error)
                        df_eval_std.index = self.models

                        if measure == 'weighted_F2':
                            df_eval_std = np.round(df_eval_std, 2)  # * 100
                        else:
                            df_eval_std = np.round(df_eval_std, 4) * 100

                        if LaTeX:
                            column_things = self.artifacts.copy()
                            column_things.append(f'Weighted {measure}')
                            df = np.empty((len(self.models), len(column_things)))
                            df = pd.DataFrame(df, index=self.models, columns=column_things)
                            for i in range(len(self.models)):
                                for j, artifact in enumerate(column_things):
                                    if artifact == f"Weighted {measure}":

                                        df.iloc[i, j] = str(
                                            f"{np.round(np.mean(df_eval.iloc[i, :6]), 2)} $\pm$ {np.round(np.mean(df_eval_std.iloc[i, :6]), 2)}")
                                    else:
                                        df.iloc[i, j] = str(
                                            f"{np.round(df_eval.iloc[i, j], 2)} $\pm$ {np.round(df_eval_std.iloc[i, j], 2)}")

                            df_latex = df.to_latex()
                            print(df_latex)

                        else:
                            print('\nSTANDARD DEVIATIONS')
                            print("Aug. ratio:" + str(aug_ratio) + "\n")
                            print(df_eval_std.to_string())

                    print("")
                    print(100 * "#")

    def getPredictions(self, aug_ratios=None, smote_ratios=None, artifacts=None, models=None, withFolds=False):

        correct_models = ['baseline_perm', 'LR', 'GNB', 'RF', 'LDA', 'MLP', 'SGD']
        if self.models != correct_models:
            self.models = correct_models

        if aug_ratios is None:
            aug_ratios = self.aug_ratios
        if smote_ratios is None:
            smote_ratios = self.smote_ratios
        if artifacts is None:
            artifacts = self.artifacts
        if models is None:
            models = self.models

        # Specifies basepath
        results_basepath = self.slash.join(self.pickle_path.split(self.slash)[:-2])
        exp = self.pickle_path.split(self.slash)[-1]

        # Loads merged file or not.
        if self.merged_file:
            results_all = LoadNumpyPickles(
                pickle_path=(self.slash).join([results_basepath, "merged_files", exp]),
                file_name=self.slash + "results" + self.experiment_name + '.npy', windowsOS=self.windowsOS)
            results_all = results_all[()]
        else:
            results_all = LoadNumpyPickles(pickle_path=(self.slash).join([results_basepath, "performance", exp]),
                                           file_name=self.slash + "results" + self.experiment_name + '.npy',
                                           windowsOS=self.windowsOS)
            results_all = results_all[()]

        # For all folds to get predictions of all data points.
        y_pred_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
        for aug_ratio in aug_ratios:
            for smote_ratio in smote_ratios:
                for artifact in artifacts:
                    for model in models:
                        y_pred = []
                        for fold in self.folds:
                            # if results_all[aug_ratio][smote_ratio][fold][artifact][model] != defaultdict(dict): # if model is not in merged files
                            y_pred_fold = results_all[aug_ratio][smote_ratio][fold][artifact][model]['y_pred']

                            y_pred.append(y_pred_fold)

                        try:
                            trial = np.concatenate(y_pred)
                            if withFolds == False:
                                y_pred = trial

                        except ValueError:
                            y_pred = np.nan

                        y_pred_dict[aug_ratio][smote_ratio][artifact][model] = y_pred

        return y_pred_dict

    def getCorrelation(self, aug_ratio=0, smote_ratio=1, artifact=None, models=None, print_matrix=True):

        if models == None:
            models = self.models

        if artifact == None:
            print("Please specify artifact!")

        else:
            y_pred_dict = self.getPredictions(aug_ratios=[aug_ratio], smote_ratios=[smote_ratio], artifacts=[artifact],
                                              models=models)
            matrix = []
            model_names = []

            for model in models:
                pred_models = y_pred_dict[aug_ratio][smote_ratio][artifact][model]

                if np.any(np.isnan(pred_models)):
                    pass
                else:
                    matrix.append(pred_models)
                    model_names.append(model)

        corr_matrix = pd.DataFrame(np.corrcoef(matrix), columns=model_names, index=model_names)

        if print_matrix:
            print("#" * 80)
            print(f"Correlation with augmentation rate: {aug_ratio}, SMOTE-ratio: {smote_ratio}")
            print(corr_matrix)

        return corr_matrix

    def getMutualInformation(self, aug_ratio=0, smote_ratio=1, artifact=None, models=None):
        if models == None:
            models = self.models

        if artifact == None:
            print("Please specify artifact!")

        else:
            corr_matrix = self.getCorrelation(artifact='eyem', print_matrix=False)

            # Formula from here: https://lips.cs.princeton.edu/correlation-and-mutual-information/
            I = -1 / 2 * np.log(1 - np.round(corr_matrix, 4) ** 2)

        return I

    def getCorrelationAcrossRatios(self, aug_ratio=0, smote_ratio=1, artifact=None, models=None, print_matrix=True):

        # TODO: Insert lines here!
        raise NotImplementedError("CREATE THIS FUNCTION!")

    def EnsemblePredictions(self, select_models, select_aug_ratios, select_smote_ratios, artifacts=None,
                            withFolds=True):

        if artifacts is None:
            artifacts = self.artifacts

        n_classifiers = len(select_models)
        y_pred_dict = self.getPredictions(models=self.models,
                                          aug_ratios=self.aug_ratios,
                                          smote_ratios=self.smote_ratios,
                                          artifacts=self.artifacts,
                                          withFolds=withFolds)

        ensemble_preds_art = defaultdict(dict)

        for j in range(len(artifacts)):
            ensemble_preds = []
            for i in range(n_classifiers):
                model_pred = y_pred_dict[select_aug_ratios[i]][select_smote_ratios[i]][artifacts[j]][select_models[i]]
                ensemble_preds.append(model_pred)

            # Hard voting classifier, since it is on class labels and not probability.
            ensemble_preds = np.array(ensemble_preds)

            print(f"\nCreating ensemble predictions for {artifacts[j]}")
            ensemble_preds_gathered = []

            if withFolds == False:
                for lst in tqdm(ensemble_preds.T):
                    ensemble_preds_gathered.append(Counter(lst).most_common(1)[0][0])
                    ensemble_preds_art[artifacts[j]] = np.array(ensemble_preds_gathered)
            else:
                for i, fold in enumerate(self.folds):
                    for point in tqdm(range(len(ensemble_preds[0, i]))):
                        preds = []
                        for m, model in enumerate(ensemble_preds[:, i]):
                            preds.append(model[point])
                        ensemble_preds_gathered.append(Counter(preds).most_common(1)[0][0])
                        ensemble_preds_art[artifacts[j]][fold] = np.array(ensemble_preds_gathered)

        return ensemble_preds_art

    def metrics_scores(self, y_true, y_pred):
        # zero_division sets it to 0 as default
        f2_s = fbeta_score(y_true, y_pred, average='weighted', beta=2.0, zero_division=0)

        conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

        TP = conf_matrix[1][1]
        TN = conf_matrix[0][0]
        FP = conf_matrix[0][1]
        FN = conf_matrix[1][0]

        accuracy = (TP + TN) / (TP + TN + FP + FN)

        if TP == 0 and FN == 0:
            print("No TP or FN found.")
            FN = 1  # Random number to account for division by zero

        sensitivity = (TP / float(TP + FN))

        # rounding digits
        accuracy, f2_s, sensitivity = np.round([accuracy, f2_s, sensitivity], 5)
        return accuracy, f2_s, sensitivity

    def printScores(self, pred_dict, y_true_filename, artifacts, smote_ratio=1, aug_ratio=0, model=None, ensemble=False,
                    print_confusion=True):

        if ensemble:
            model = "Ensemble method"
        else:
            pred_dict = pred_dict[model]

        results_basepath = self.slash.join(self.pickle_path.split(self.slash)[:-2])

        y_true = LoadNumpyPickles(
            pickle_path=(self.slash).join([results_basepath, "y_true"]),
            file_name=self.slash + y_true_filename + '.npy', windowsOS=self.windowsOS)
        y_true = y_true[()]

        y_true_dict = defaultdict(dict)

        for artifact in self.artifacts:
            y_true_art = []
            for i in self.folds:
                y_true_art.append(y_true[i][artifact]['y_true'])

            y_true_art = np.concatenate(y_true_art)
            y_true_dict[artifact] = y_true_art

        for art in artifacts:
            scores = self.metrics_scores(y_true=y_true_dict[art], y_pred=pred_dict[art])
            print(f"\nResults on {art}. Ensemble classifier = {ensemble}")
            print("Accuracy, F2, sensitivity")
            print(scores)

            if print_confusion:
                conf_matrix = confusion_matrix(y_true=y_true_dict[art], y_pred=pred_dict[art], labels=[0, 1])
                conf_matrix = conf_matrix / np.linalg.norm(conf_matrix)
                conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

                sn.heatmap(conf_matrix, annot=True, cmap=plt.cm.Blues)
                plt.xlabel("Predicted label")
                plt.ylabel("Actual label")
                plt.title(f"Confusion matrix for {art} artifact with {model}, SMOTE: {smote_ratio}, Aug.: {aug_ratio}")
                plt.show()

    def compressDict(self, pred_dict, smote_ratio=1, aug_ratio=0):

        pred_dict_new = defaultdict(dict)
        models = list(pred_dict[aug_ratio][smote_ratio]['eyem'].keys())

        for artifact in self.artifacts:
            for model in models:
                pred_dict_new[model][artifact] = pred_dict[aug_ratio][smote_ratio][artifact][model]

        return pred_dict_new

    def plot_multi_label_confusion(self, pred_dict, y_true_filename, artifacts, smote_ratio=1, aug_ratio=0,
                                   aug_technique=None, models=None, ensemble=False):
        pred_dict_orig = pred_dict
        for model in models:

            if ensemble:
                model = "Ensemble method"
            else:
                pred_dict = pred_dict_orig[model]

            results_basepath = self.slash.join(self.pickle_path.split(self.slash)[:-2])

            y_true = LoadNumpyPickles(
                pickle_path=(self.slash).join([results_basepath, "y_true"]),
                file_name=self.slash + y_true_filename + '.npy', windowsOS=self.windowsOS)
            y_true = y_true[()]

            y_true_dict = {}

            for artifact in self.artifacts:
                y_true_art = []
                for i in self.folds:
                    y_true_art.append(y_true[i][artifact]['y_true'])

                y_true_art = np.concatenate(y_true_art)
                y_true_dict[artifact] = y_true_art

            predictions = np.array([item for item in pred_dict.values()]).T
            actual = np.array([item for item in y_true_dict.values()]).T

            multi_cm = multilabel_confusion_matrix(y_true=actual, y_pred=predictions)
            multi_cm = [cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] for cm in multi_cm]

            fig, axs = plt.subplots(2, 3)  # , sharex=True, sharey=True)

            cbar_ax = fig.add_axes([0.89, .3, .03, .4])
            plt.subplots_adjust(left=0.1, right=0.85)

            for i, artifact in enumerate(self.artifacts):
                j = 0

                if i > 2:
                    j = 1

                conf_matrix = sn.heatmap(multi_cm[i], annot=True, cmap=plt.cm.Blues,
                                         ax=axs[j, i % 3], cbar_ax=cbar_ax,
                                         vmin=0, vmax=1)
                conf_matrix.set_aspect('equal', 'box')

                axs[j, 0].tick_params()
                axs[j, 1].tick_params()

                if i % 3 == 0:
                    conf_matrix.set_ylabel("Actual label")
                if j == 1:
                    conf_matrix.set_xlabel("Predicted label")

                conf_matrix.set_title(artifact)

            if aug_technique == None:
                fig.suptitle(f"Model: {model}, SMOTE: {smote_ratio}")
            else:
                fig.suptitle(f"Model: {model}, SMOTE: {smote_ratio}, {aug_technique}: {aug_ratio}")

            save_path = (self.slash).join([self.dir, "Plots" + self.slash + "confusion_matrix",
                                           f"SMOTE{smote_ratio}_aug{aug_technique}{aug_ratio}"])

            img_path = f"{save_path}{self.slash}{model}.png"
            os.makedirs((self.slash).join(img_path.split(self.slash)[:-1]), exist_ok=True)
            fig.savefig(img_path)

            fig.show()

    def plotAllModelsBestAug(self, experiment_name: object, aug_technique: object, y_true_path: object = None,
                             smote_ratios: object = None,
                             aug_ratios: object = None,
                             measure: object = 'balanced_acc', save_img: object = False) -> object:
        across_SMOTE = False
        self.improvementExperiment = True

        if smote_ratios == None:
            smote_ratios = [1]
        if aug_ratios == None:
            aug_ratios = self.aug_ratios

        # Loading statistically calculated results as dictionaries
        # For single files and their HO_trials
        # List of dictionaries of results. Each entry in the list is a results dictionary for one SMOTE ratio
        performances, errors = self.tableResults_Augmentation(measure=measure, y_true_path=y_true_path,
                                                              smote_ratios=smote_ratios,
                                                              experiment_name=experiment_name)
        smote_ratios.sort()

        # This function will plot results created in the augmentation experiment (with aug. ratio key in the dict)
        self.plotResultsHelper(performances_dict=performances, errors_dict=errors,
                               experiment=self.experiment, smote_ratios=smote_ratios,
                               aug_ratios=aug_ratios, across_SMOTE=across_SMOTE,
                               save_img=save_img, aug_technique=aug_technique,
                               measure=measure)


if __name__ == '__main__':
    dir = r"C:\Users\Albert Kjøller\Documents\GitHub\EEG_epilepsia"  # dir = "/Users/philliphoejbjerg/Documents/GitHub/EEG_epilepsia"  # dir = r"/Users/Jacobsen/Documents/GitHub/EEG_epilepsia" + "/"

    # Example of merging fully created files from different models.
    experiment = "SMOTE"  # directory containing the files we will look at
    experiment_name = '_control_experiment'
    fullSMOTE = getResults(dir, experiment, experiment_name, merged_file=True, windowsOS=True)
    fullSMOTE.mergeResultFiles(file_name=experiment_name)

    # To work with the merged file we have to change the pickle path to the "merged" folder.
    fullSMOTE.changePicklePath()
    y_true_path = r"C:\Users\Albert Kjøller\Documents\GitHub\EEG_epilepsia\results\y_true\y_true_5fold_randomstate_0.npy"
    # performances, errors = fullSMOTE.tableResults_Augmentation(experiment_name=experiment_name, y_true_path=y_true_path, smote_ratios=[1], measure="balanced_acc")

    fullSMOTE.plotResultsPlainExp(experiment_name=experiment_name, y_true_path=y_true_path, across_SMOTE=True,
                                  save_img=True)

    fullSMOTE.printResults(measure="sensitivity",
                           experiment_name=experiment_name,
                           y_true_path=y_true_path,
                           smote_ratios=[1],
                           aug_ratios=[0],
                           printSTDTable=True,
                           LaTeX=True)
    fullSMOTE.printResults(measure="balanced_acc",
                           experiment_name=experiment_name,
                           y_true_path=y_true_path,
                           smote_ratios=[1],
                           aug_ratios=[0],
                           printSTDTable=True,
                           LaTeX=True)

    fullSMOTE.plotResults(experiment_name,
                          aug_technique=None,
                          y_true_path=y_true_path,
                          smote_ratios=[1],
                          aug_ratios=[0],
                          measure='balanced_acc',
                          across_SMOTE=True, save_img=True)

    artifacts = fullSMOTE.artifacts
    smote_ratio = 1
    models = fullSMOTE.models  # 'GNB'

    y_pred_dict = fullSMOTE.getPredictions(models=models,
                                           aug_ratios=[0],
                                           smote_ratios=[smote_ratio],
                                           artifacts=artifacts)
    y_pred_dict = fullSMOTE.compressDict(y_pred_dict, smote_ratio=1, aug_ratio=0)

    fullSMOTE.printScores(pred_dict=y_pred_dict, y_true_filename="y_true_5fold_randomstate_0", model='LDA',
                          artifacts=artifacts, ensemble=False, print_confusion=True)

    fullSMOTE.plot_multi_label_confusion(pred_dict=y_pred_dict, y_true_filename="y_true_5fold_randomstate_0",
                                         models=models,
                                         artifacts=fullSMOTE.artifacts, smote_ratio=smote_ratio - 1, ensemble=False)

    print("")
