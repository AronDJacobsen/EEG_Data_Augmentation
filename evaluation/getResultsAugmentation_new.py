import scipy.stats
import numpy as np
from prepData.dataLoader import *
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict
import glob
import pandas as pd


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
        self.models = ['baseline_perm', 'LR', 'GNB', 'KNN', 'RF', 'LDA', 'MLP', 'AdaBoost', 'SGD']
        self.scores = ['y_pred', 'accuracy', 'weighted_F2', 'sensitivity']

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

    def mergeResultFiles(self, file_name="merged", merge_smote_files=False, merge_aug_files=False, exclude_models=None):

        if self.merged_file == False:
            raise Exception("Merged file set to 'False'")
        else:
            if exclude_models is None:
                exclude_models = ['baseline_major']

            # Find files to be merged
            file_path = (self.slash).join([self.basepath, "performance", self.experiment, ""])
            file_names = [results_file.split(self.slash)[-1] for results_file in
                          glob.glob(file_path + "**")]

            if len(file_names) == 0:
                raise FileNotFoundError("There are no files in the specified directory.")

            i = 0
            all_results_dict = {}

            # Extract information from the original pickles from the experiments and save them in a new nested dictionary.
            for model_file in file_names:
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

    def tableResults_Augmentation(self, experiment_name, measure="sensitivity"):

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

            for smote_ratio in self.smote_ratios:
                table = defaultdict(dict)  # {}
                table_std = defaultdict(dict)  # {}

                for idx_art, artifact in enumerate(self.artifacts):
                    # table[artifact] = []
                    store_model = [0] * len(self.models)
                    sensitivity_std = [0] * len(self.models)

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
                                store_scores.append(results[smote_ratio][fold][artifact][model][measure])
                                temp_acc.append(results[smote_ratio][fold][artifact][model]['accuracy'])
                                temp_f2.append(results[smote_ratio][fold][artifact][model]['weighted_F2'])
                                empty = False

                        acc[idx_mod, idx_art] = np.mean(temp_acc)
                        f2s[idx_mod, idx_art] = np.mean(temp_f2)

                        # Standard deviation for acc. and F2 on each artifact
                        acc_std[idx_mod, idx_art] = np.std(temp_acc)
                        f2s_std[idx_mod, idx_art] = np.std(temp_f2)

                        # Store standard deviation for sensitivies for each classifier/ model
                        store_model[idx_mod] = np.mean(store_scores)
                        sensitivity_std[idx_mod] = np.mean(store_scores)

                    table[artifact] = store_model
                    table_std[artifact] = sensitivity_std

                table['avg. accuracy'] = np.mean(acc, axis=1)
                table['avg. weighted f2'] = np.mean(f2s, axis=1)

                # Mean standard deviation
                table_std['avg. accuracy'] = np.mean(acc_std, axis=1)
                table_std['avg. weighted f2'] = np.mean(f2s_std, axis=1)

                table_smote_dict[smote_ratio] = table
                table_smote_std_dict[smote_ratio] = table_std

                table_aug_dict[aug_ratio][smote_ratio] = table
                table_aug_std_dict[aug_ratio][smote_ratio] = table_std

        return table_aug_dict, table_aug_std_dict

    def plotResultsAugmentation(self, performances_dict, errors_dict, experiment, aug_technique, measure="sensitiviy",
                                save_img=False):

        save_path = (self.slash).join([dir, "Plots", experiment])

        colorlist = ["lightslategrey", "lightsteelblue", "darkcyan", "firebrick", "lightcoral"]
        # Plotting results
        art = len(self.artifacts)

        for indv_art, name in enumerate(self.artifacts):
            for j, smote_ratio in enumerate(self.smote_ratios):
                for i, aug_ratio in enumerate(self.aug_ratios):
                    performance_dict = performance_dict[aug_ratio][smote_ratio]
                    error_dict = errors_dict[aug_ratio][smote_ratio]

                    performance_vals = np.array(list(performance_dict.values())[:art])
                    error_vals = np.array(list(error_dict.values())[:art])

                    X_axis = np.arange(len(self.models)) - 0.3
                    plt.bar(x=X_axis + 0.15 * i, height=performance_vals[indv_art, :], width=0.15, color=colorlist[i],
                            label="Aug. ratio = " + str(self.aug_ratios[i]))

                    plt.errorbar(x=X_axis + 0.15 * i, y=performance_vals[indv_art, :], yerr=error_vals[indv_art, :],
                                 fmt='.', color='k')

                for i, model in enumerate(self.models):
                    if model == 'baseline_perm':
                        self.models[i] = "base-\nline"
                    elif model == 'AdaBoost':
                        self.models[i] = "Ada-\nBoost"

                plt.xticks(np.arange(len(self.models)), self.models, rotation=0)
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

        save_path = (self.slash).join([dir, "Plots", experiment])

        colorlist = ["lightslategrey", "lightsteelblue", "darkcyan", "firebrick", "lightcoral"]
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

                        X_axis = np.arange(len(self.models)) - 0.3
                        plt.bar(x=X_axis + 0.15 * i, height=performance_vals[indv_art, :], width=0.15,
                                color=colorlist[i],
                                label="SMOTE-ratio = " + str(smote_ratios[i]))

                        plt.errorbar(x=X_axis + 0.15 * i, y=performance_vals[indv_art, :], yerr=error_vals[indv_art, :],
                                     fmt='.', color='k')

                    for i, model in enumerate(self.models):
                        if model == 'baseline_perm':
                            self.models[i] = "base-\nline"
                        elif model == 'AdaBoost':
                            self.models[i] = "Ada-\nBoost"

                    plt.xticks(np.arange(len(self.models)), self.models, rotation=0)
                    plt.ylim(0, 1)

                    if aug_technique == None:
                        plt.title(
                            f"{measure} with {aug_technique} augmentation on the '{name}'-class - Aug. ratio = {aug_ratios[j]}")
                    else:
                        plt.title(
                            f"{measure} with {aug_technique} augmentation on the '{name}'-class - Aug. ratio = {aug_ratios[j]}")

                    plt.xlabel("Model")
                    plt.ylabel(measure)
                    plt.legend(loc='center right', bbox_to_anchor=(1.36, 0.5))
                    plt.subplots_adjust(bottom=0.2, right=0.775)

                    if save_img:
                        img_path = f"{(self.slash).join([save_path, measure])}{self.slash}{name}_aug{aug_technique}_augRatio{aug_ratios[j]}.png"
                        os.makedirs((self.slash).join(img_path.split(self.slash)[:-1]), exist_ok=True)
                        plt.savefig(img_path)
                    plt.show()

        else:
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

                    for i, model in enumerate(self.models):
                        if model == 'baseline_perm':
                            self.models[i] = "base-\nline"
                        elif model == 'AdaBoost':
                            self.models[i] = "Ada-\nBoost"

                    plt.xticks(np.arange(len(self.models)), self.models, rotation=0)
                    plt.ylim(0, 1)

                    if aug_technique == None:
                        plt.title(
                            f"{measure} with {aug_technique} augmentation on the '{name}'-class - SMOTE = {smote_ratios[j] - 1}")
                    else:
                        plt.title(
                            f"{measure} with {aug_technique} augmentation on the '{name}'-class - SMOTE = {smote_ratios[j] - 1}")

                    plt.xlabel("Model")
                    plt.ylabel(measure)
                    plt.legend(loc='center right', bbox_to_anchor=(1.36, 0.5))
                    plt.subplots_adjust(bottom=0.2, right=0.775)

                    if save_img:
                        img_path = f"{(self.slash).join([save_path, measure])}{self.slash}{name}_aug{aug_technique}_smote{smote_ratios[j] - 1}.png"
                        os.makedirs((self.slash).join(img_path.split(self.slash)[:-1]), exist_ok=True)
                        plt.savefig(img_path)
                    plt.show()

    def plotResults(self, experiment_name, aug_technique, smote_ratios=None, aug_ratios=None, measure='sensitivity',
                    across_SMOTE=True, save_img=False):

        if smote_ratios == None:
            smote_ratios = self.smote_ratios
        if aug_ratios == None:
            aug_ratios = self.aug_ratios

        # Loading statistically calculated results as dictionaries
        # For single files and their HO_trials
        # List of dictionaries of results. Each entry in the list is a results dictionary for one SMOTE ratio
        performances, errors = self.tableResults_Augmentation(measure=measure, experiment_name=experiment_name)
        self.smote_ratios.sort()

        # This function will plot results created in the augmentation experiment (with aug. ratio key in the dict)
        self.plotResultsHelper(performances_dict=performances, errors_dict=errors,
                               experiment=self.experiment, smote_ratios=smote_ratios,
                               aug_ratios=aug_ratios, across_SMOTE=across_SMOTE,
                               save_img=save_img, aug_technique=aug_technique,
                               measure=measure)

    def printResults(self, experiment_name, smote_ratios=None, aug_ratios=None, measure='sensitivity',
                     printSTDTable=False, LaTeX=False):

        if smote_ratios == None:
            smote_ratios = self.smote_ratios
        if aug_ratios == None:
            aug_ratios = self.aug_ratios

        performances, errors = self.tableResults_Augmentation(measure=measure, experiment_name=experiment_name)
        self.smote_ratios.sort()

        for j in range(len(aug_ratios)):
            print("\n\n")
            print(80 * "#")
            print("Results with augmentation rate set to {:2f}".format(self.aug_ratios[j]))
            print(100 * "#")

            performance_dict = performances[j]
            errors_dict = errors[j]

            for i, ratio in enumerate(smote_ratios):

                performance = performance_dict[ratio]
                error = errors_dict[ratio]

                # Print dataframes
                df_eval = pd.DataFrame.from_dict(performance)
                df_eval.index = self.models
                df_eval = np.round(df_eval * 100, 2)

                if LaTeX:
                    df_latex = df_eval.to_latex()
                    print(df_latex)

                else:
                    print('\nOVERALL PERFORMANCE')
                    print("SMOTE RATIO:" + str(ratio - 1) + "\n")
                    print(df_eval.to_string())
                    # print(df_eval)

                if printSTDTable:
                    df_eval = pd.DataFrame.from_dict(error)
                    df_eval.index = self.models
                    df_eval = np.round(df_eval * 100, 2)
                    if LaTeX:
                        df_latex = pd.df_eval.to_latex()
                        print(df_latex)

                    else:
                        print('\nSTANDARD DEVIATIONS')
                        print("SMOTE RATIO:" + str(ratio - 1) + "\n")
                        print(df_eval.to_string)

                print("")
                print(100 * "#")

    def getPredictions(self, model):

        pass


if __name__ == '__main__':
    dir = r"C:\Users\Albert Kjøller\Documents\GitHub\EEG_epilepsia"  # dir = "/Users/philliphoejbjerg/Documents/GitHub/EEG_epilepsia"  # dir = r"/Users/Jacobsen/Documents/GitHub/EEG_epilepsia" + "/"

    experiment = "smote_f2"  # "DataAug_white_noiseAdd_LR"
    experiment_name = '_smote_f2_GNB'  # end by either _Noise (or _color_Noise), _GAN or _MixUp if Augmentation
    experiment_object = getResults(dir, experiment, experiment_name, merged_file=False, windowsOS=True)

    # Example of manipulating a single model by adding results from a single artifact, i.e. GNB Null added to GNB.
    GNB = "GNB_for_merge_smote_f2"
    experiment_object.addSingleArtifact(dir_name=GNB, new_file_name="results_smote_f2_GNB",
                                        main_file="results_smote_f2_GNB.npy",
                                        sec_file="results_smote_f2_GNBNull.npy",
                                        experiment=experiment)

    # Example of merging result-files created with same models but different SMOTE-ratios
    experiment = 'KNN_for_merge_smote_f2'
    experiment_name = '_smote_f2_KNN'  # end of the name of the new file to be created or the file(s) to be loaded
    KNN_smote = getResults(dir, experiment, experiment_name, merged_file=True, windowsOS=True)
    KNN_smote.mergeResultFiles(file_name=experiment_name)

    # Example of merging fully created files from different models.
    experiment = "smote_f2"  # directory containing the files we will look at
    experiment_name = '_smote_f2_mergeExample'
    fullSMOTE = getResults(dir, experiment, experiment_name, merged_file=True, windowsOS=True)
    fullSMOTE.mergeResultFiles(file_name=experiment_name)

    # To work with the merged file we have to change the pickle path to the "merged" folder.
    fullSMOTE.changePicklePath()

    # First we wish to examine sensitivities
    measure = "sensitivity"
    aug_technique = None  # "GAN" # "MixUp" #Noise Addition --> mainly for naming of the plots
    only_smote = True

    LaTeX = False
    save_img = True

    # performances is a nested list of performance dictionaries, the same is errors (but with errors instead)
    # The first list indices describes augmentation ratio, the inner list is smote ratio.
    performances, errors = fullSMOTE.tableResults_Augmentation(experiment_name=experiment_name, measure=measure)

    # Examples of getting tables of the results
    fullSMOTE.printResults(measure=measure,
                           experiment_name=experiment_name,
                           smote_ratios=fullSMOTE.smote_ratios,
                           aug_ratios=[0],
                           printSTDTable=False,
                           LaTeX=False)

    fullSMOTE.printResults(measure=measure,
                           experiment_name=experiment_name,
                           smote_ratios=fullSMOTE.smote_ratios,
                           aug_ratios=[0],
                           printSTDTable=False,
                           LaTeX=True)

    # Examples of getting plots of the results
    fullSMOTE.plotResults(measure=measure,
                          experiment_name=experiment_name,
                          aug_technique=aug_technique,
                          smote_ratios=fullSMOTE.smote_ratios,
                          aug_ratios=[0],
                          across_SMOTE=True,
                          save_img=save_img)

    fullSMOTE.plotResults(measure=measure,
                          experiment_name=experiment_name,
                          aug_technique=aug_technique,
                          smote_ratios=fullSMOTE.smote_ratios,
                          aug_ratios=[0],
                          across_SMOTE=False,
                          save_img=save_img)

    fullSMOTE.plotResults(measure='accuracy',
                          experiment_name=experiment_name,
                          aug_technique=aug_technique,
                          smote_ratios=fullSMOTE.smote_ratios,
                          aug_ratios=[0],
                          across_SMOTE=True,
                          save_img=save_img)

    fullSMOTE.getPredictions(model='LR')

    # Next we wish to examine accuracy
    measure = "accuracy"

    performances, errors = fullSMOTE.tableResults_Augmentation(experiment_name=experiment_name, measure=measure)
    fullSMOTE.plotResultsChooseMeasure(measure=measure,
                                       experiment_name=experiment_name,
                                       aug_technique=aug_technique,
                                       save_img=save_img)
    fullSMOTE.printResultsChooseMeasure(measure=measure, experiment_name=experiment_name, LaTeX=LaTeX)

    # Then we wish to examine F2-score
    measure = "weighted_F2"

    performances, errors = fullSMOTE.tableResults_Augmentation(experiment_name=experiment_name, measure=measure)
    fullSMOTE.plotResultsChooseMeasure(measure=measure,
                                       experiment_name=experiment_name,
                                       aug_technique=aug_technique,
                                       save_img=save_img)
    fullSMOTE.printResultsChooseMeasure(measure=measure, experiment_name=experiment_name, LaTeX=LaTeX)

    # TODO: Don't know how we will work with the augmented files

    # Example of merging result-files created with same models but different Augmentation-ratios
    # TODO: Not tried yet as we have not conducted the experiment
    # experiment = "LR_for_merge_MixUp"
    # experiment_name = '_MixUp_LR'
    # LR_MixUp = getResults(dir, experiment, experiment_name, merged_file=True, windowsOS=True)
    # LR_MixUp.mergeResultFiles(file_name=experiment_name)

    fullSMOTE.printResultsChooseMeasure(aug_ratios=fullSMOTE.aug_ratios)

    print("")
