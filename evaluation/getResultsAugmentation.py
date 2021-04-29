import scipy.stats
import numpy as np
from prepData.dataLoader import *
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict
import glob
import pandas as pd




def mergeResultFiles(file_path, file_name="merged", windowsOS=False, merge_smote_files=True,
                     merge_aug_files=False,
                     exclude_models=None):

    if exclude_models is None:
        exclude_models = ['baseline_major']

    if windowsOS:
        slash = "\\"
    else:
        slash = "/"

    # Nested dictionary
    all_results_dict = {}

    file_names = [results_file.split(slash)[-1] for results_file in glob.glob(file_path + slash + "**")]
    i = 0
    for model_file in file_names:
        results = LoadNumpyPickles(file_path + slash, model_file, windowsOS=windowsOS)[()]
        aug_ratios = list(results.keys())
        smote_ratios = list(results[aug_ratios[0]].keys())
        folds = [key for key in results[aug_ratios[0]][smote_ratios[0]].keys() if type(key) == int]
        artifacts = list(results[aug_ratios[0]][smote_ratios[0]][folds[0]].keys())
        models = list(results[aug_ratios[0]][smote_ratios[0]][folds[0]][artifacts[0]].keys())

        if merge_aug_files and merge_smote_files:
            raise AttributeError("Only possible to merge one file-type at a time!")

        if merge_smote_files:
            for aug_ratio in aug_ratios:
                if i == 0:
                    all_results_dict[aug_ratio] = defaultdict(dict)
                for ratio in smote_ratios:
                    all_results_dict[aug_ratio][ratio] = defaultdict(dict)
                    for fold in folds:
                        all_results_dict[aug_ratio][ratio][fold] = defaultdict(dict)
                        for artifact in artifacts:
                            all_results_dict[aug_ratio][ratio][fold][artifact] = defaultdict(dict)
                            for model in models:
                                all_results_dict[aug_ratio][ratio][fold][artifact][model] = defaultdict(dict)
                                if model in exclude_models:
                                    pass
                                else:
                                    all_results_dict[aug_ratio][ratio][fold][artifact][model] = \
                                        results[aug_ratio][ratio][fold][artifact][model]

        elif merge_aug_files:
            for aug_ratio in aug_ratios:
                all_results_dict[aug_ratio] = defaultdict(dict)
                for ratio in smote_ratios:
                    all_results_dict[aug_ratio][ratio] = defaultdict(dict)
                    for fold in folds:
                        all_results_dict[aug_ratio][ratio][fold] = defaultdict(dict)
                        for artifact in artifacts:
                            all_results_dict[aug_ratio][ratio][fold][artifact] = defaultdict(dict)
                            for model in models:
                                all_results_dict[aug_ratio][ratio][fold][artifact][model] = defaultdict(dict)
                                if model in exclude_models:
                                    pass
                                else:
                                    all_results_dict[aug_ratio][ratio][fold][artifact][model] = \
                                        results[aug_ratio][ratio][fold][artifact][model]

        else:
            if i == 0:
                for aug_ratio in aug_ratios:
                    all_results_dict[aug_ratio] = defaultdict(dict)
                    for ratio in smote_ratios:
                        all_results_dict[aug_ratio][ratio] = defaultdict(dict)
                        for fold in folds:
                            all_results_dict[aug_ratio][ratio][fold] = defaultdict(dict)
                            for artifact in artifacts:
                                all_results_dict[aug_ratio][ratio][fold][artifact] = defaultdict(dict)
                                for model in models:
                                    if model == 'baseline_major':
                                        pass
                                    else:
                                        all_results_dict[aug_ratio][ratio][fold][artifact][model] = \
                                            results[aug_ratio][ratio][fold][artifact][model]

            else:
                for aug_ratio in aug_ratios:
                    for ratio in smote_ratios:
                        for fold in folds:
                            for artifact in artifacts:
                                for model in models:
                                    all_results_dict[aug_ratio][ratio][fold][artifact][model] = \
                                        results[aug_ratio][ratio][fold][artifact][model]

        i += 1

    # Save file in merged_files dir
    results_basepath = slash.join(file_path.split(slash)[:-2])

    exp = file_path.split(slash)[-1]
    print("\nNew file created!")

    save_path = results_basepath + slash + "merged_files" + slash + exp
    try:
        os.makedirs(save_path)
        print("New directory created!")
    except FileExistsError:
        pass

    SaveNumpyPickles(save_path, slash + file_name, all_results_dict, windowsOS)

def manipulateFile(file_path, file_name="merged", experiment="", main_file="", sec_file="", windowsOS=False):

    if windowsOS:
        slash = "\\"
    else:
        slash = "/"

    # Nested dictionary
    all_results_dict = {}  # defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    file_names = [results_file.split(slash)[-1] for results_file in glob.glob(file_path + slash + "**")]

    if file_names[0] != main_file:
        file_names = file_names[::-1]

    i = 0
    for model_file in file_names:
        results = LoadNumpyPickles(file_path + slash, model_file, windowsOS=windowsOS)[()]
        aug_ratios = list(results.keys())
        smote_ratios = list(results[aug_ratios[0]].keys())
        folds = [key for key in results[aug_ratios[0]][smote_ratios[0]].keys() if type(key) == int]
        artifacts = list(results[aug_ratios[0]][smote_ratios[0]][folds[0]].keys())
        models = list(results[aug_ratios[0]][smote_ratios[0]][folds[0]][artifacts[0]].keys())

        if i == 0:
            for aug_ratio in aug_ratios:
                all_results_dict[aug_ratio] = {}

                for ratio in smote_ratios:
                    all_results_dict[aug_ratio][ratio] = {}

                    for fold in folds:
                        all_results_dict[aug_ratio][ratio][fold] = {}

                        for artifact in artifacts:
                            all_results_dict[aug_ratio][ratio][fold][artifact] = {}

                            for model in models:
                                all_results_dict[aug_ratio][ratio][fold][artifact][model] = \
                                    results[aug_ratio][ratio][fold][artifact][model]
        else:
            for aug_ratio in aug_ratios:
                for ratio in smote_ratios:
                    for fold in folds:
                        for artifact in artifacts:
                            for model in models:
                                all_results_dict[aug_ratio][ratio][fold][artifact][model] = \
                                    results[aug_ratio][ratio][fold][artifact][model]

        i += 1
    # Save file in merged_files dir
    results_basepath = slash.join(file_path.split(slash)[:-2])

    exp = experiment
    print("\nNew file created!")
    SaveNumpyPickles(results_basepath + slash + "performance" + slash + exp, slash + file_name, all_results_dict,
                     windowsOS)

def tableResults_Augmentation(pickle_path, experiment_name, merged_file=False, windowsOS=False,
                              measure="sensitivity"):
    # fold, artifact, model, scores
    if windowsOS:
        slash = "\\"
    else:
        slash = "/"

    results_basepath = slash.join(pickle_path.split(slash)[:-2])
    exp = pickle_path.split(slash)[-1]

    if merged_file:
        results_all = LoadNumpyPickles(pickle_path=results_basepath + slash + "merged_files" + slash + exp,
                                       file_name=slash + experiment_name + '.npy', windowsOS=windowsOS)
        results_all = results_all[()]
    else:
        results_all = LoadNumpyPickles(pickle_path=results_basepath + slash + "performance" + slash + exp,
                                       file_name=slash + "results" + experiment_name + '.npy', windowsOS=windowsOS)
        results_all = results_all[()]
        # fold, artifact, model, hyperopt iterations
        HO_trials = LoadNumpyPickles(pickle_path=results_basepath + slash + "hyperopt" + slash + exp,
                                     file_name=slash + 'ho_trials' + experiment_name + '.npy', windowsOS=windowsOS)
        HO_trials = HO_trials[()]

    def mean_confidence_interval(data, confidence=0.95):
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
        return m - h, m, m + h

    table_list_augmentation = []
    table_std_augmentation = []

    for aug_ratio in results_all:
        results = results_all[aug_ratio]

        pd.set_option("display.max_rows", None, "display.max_columns", None)

        table = defaultdict(dict)  # {}
        table_std = defaultdict(dict)  # {}

        # want to average each fold
        # construct keys
        SMOTE_ratios = list(results.keys())
        folds = [key for key in results[SMOTE_ratios[0]].keys() if type(key) == int]
        artifacts = list(results[SMOTE_ratios[0]][folds[0]].keys())
        models = list(results[SMOTE_ratios[0]][folds[0]][artifacts[0]].keys())
        scores = list(results[SMOTE_ratios[0]][folds[0]][artifacts[0]][models[0]].keys())

        # row-wise models, column-wise artifacts
        acc = np.zeros((len(models), len(artifacts)))
        acc_std = np.zeros((len(models), len(artifacts)))
        f2s = np.zeros((len(models), len(artifacts)))
        f2s_std = np.zeros((len(models), len(artifacts)))

        table_list_smote = []
        table_std_list_smote = []

        for smote_ratio in SMOTE_ratios:

            for idx_art, artifact in enumerate(artifacts):
                # table[artifact] = []
                store_model = [0] * len(models)
                sensitivity_std = [0] * len(models)

                for idx_mod, model in enumerate(models):

                    store_scores = []

                    temp_acc = []
                    temp_f2 = []
                    for fold in folds:
                        store_scores.append(results[smote_ratio][fold][artifact][model][measure])
                        temp_acc.append(results[smote_ratio][fold][artifact][model]['accuracy'])
                        temp_f2.append(results[smote_ratio][fold][artifact][model]['weighted_F2'])

                    acc[idx_mod, idx_art] = np.mean(temp_acc)
                    f2s[idx_mod, idx_art] = np.mean(temp_f2)

                    # Standard deviation for acc. and F2 on each artifact
                    acc_std[idx_mod, idx_art] = np.std(temp_acc)
                    f2s_std[idx_mod, idx_art] = np.std(temp_f2)

                    # Store standard deviation for sensitivies for each classifier/ model
                    store_model[idx_mod] = np.mean(store_scores)
                    sensitivity_std[idx_mod] = np.std(store_scores)

                table[artifact] = store_model
                table_std[artifact] = sensitivity_std

            table['avg. accuracy'] = np.mean(acc, axis=1)
            table['avg. weighted f2'] = np.mean(f2s, axis=1)

            # Mean standard deviation
            table_std['avg. accuracy'] = np.mean(acc_std, axis=1)
            table_std['avg. weighted f2'] = np.mean(f2s_std, axis=1)

            table_list_smote.append(table)
            table_std_list_smote.append(table_std)

        table_list_augmentation.append(table_list_smote)
        table_std_augmentation.append(table_std_list_smote)

    aug_ratios = list(results_all.keys())

    return table_list_augmentation, table_std_augmentation, models, artifacts, SMOTE_ratios, aug_ratios

def plotPerformanceModels(performance_dict, error_dict, experiment, model_names, artifact_names, ratio,
                          save_img=False,
                          windowsOS=False):
    if windowsOS:
        slash = "\\"
    else:
        slash = "/"

    save_path = dir + slash + 'Plots' + slash + experiment

    # Plotting results
    art = len(artifact_names)
    performance_vals = np.array(list(performance_dict.values())[:art]).T
    error_vals = np.array(list(error_dict.values())[:art]).T

    for indv_model, name in enumerate(model_names):
        plt.bar(x=artifact_names, height=performance_vals[indv_model, :], width=0.5, color="lightsteelblue")
        plt.errorbar(x=artifact_names, y=performance_vals[indv_model, :], yerr=error_vals[indv_model, :], fmt='.',
                     color='k')
        plt.title(name + " - SMOTE RATIO:" + str(ratio - 1))
        plt.ylim(0, 1)
        if save_img:
            plt.savefig(("{}{:s}{}_SMOTE_{}.png").format(save_path, slash, name, ratio - 1))
        plt.show()

def plotPerformanceClasses(performance_dict, error_dict, experiment, model_names, artifact_names, ratio,
                           save_img=False,
                           windowsOS=False):
    if windowsOS:
        slash = "\\"
    else:
        slash = "/"

    save_path = dir + slash + 'Plots' + slash + experiment

    # Plotting results
    art = len(artifact_names)
    performance_vals = np.array(list(performance_dict.values())[:art])
    error_vals = np.array(list(error_dict.values())[:art])

    for indv_art, name in enumerate(artifact_names):
        plt.bar(x=model_names, height=performance_vals[indv_art, :], width=0.5, color="lightsteelblue")
        plt.errorbar(x=model_names, y=performance_vals[indv_art, :], yerr=error_vals[indv_art, :], fmt='.',
                     color='k')
        plt.title(name + " - SMOTE RATIO:" + str(ratio - 1))
        plt.xticks(rotation=25)
        plt.ylim(0, 1)
        if save_img:
            plt.savefig(("{}{:s}{}_SMOTE_{}.png").format(save_path, slash, name, ratio - 1))
        plt.show()

def plotHyperopt(pickle_path, file_name, windowsOS=False):
    if windowsOS:
        slash = "\\"
    else:
        slash = "/"

    try:
        results_basepath = slash.join(pickle_path.split(slash)[:-1])

        # fold, artifact, model, scores
        results = LoadNumpyPickles(pickle_path=results_basepath + slash + "performance",
                                   file_name=slash + "results" + experiment_name + '.npy', windowsOS=windowsOS)
        results = results[()]

        # fold, artifact, model, hyperopt iterations
        HO_trials = LoadNumpyPickles(pickle_path=results_basepath + slash + "hyperopt",
                                     file_name=slash + "ho_trials" + experiment_name + '.npy', windowsOS=windowsOS)
        HO_trials = HO_trials[()]

        # only choose one fold
        # construct keys
        folds = list(results.keys())
        artifacts = list(results[folds[0]].keys())
        models = list(results[folds[0]][artifacts[0]].keys())
        scores = list(results[folds[0]][artifacts[0]][models[0]].keys())

        single = HO_trials[folds[0]][artifacts[0]][models[0]]

        # TODO: Not completely functioning! It does not show the plots

        # hyperopt
        cols = list(single.columns)
        n = len(cols)
        for i in range(n - 1):  # for every parameter
            plt.scatter(single[cols[i]], single[cols[n - 1]])
            plt.title('HyperOpt: model: {}, artifact: {}'.format(models[0], cols[i]))
            plt.xlabel(cols[i])
            plt.ylabel('accuracy')
            plt.show()

    except KeyError:
        print("\n\nERROR: No Hyperopt queries used for this model!")

def plotResultsAugmentation(performance_list, errors_list, experiment, model_names, artifact_names,
                            SMOTE_ratios,
                            aug_ratios, aug_technique, measure="sensitiviy", save_img=False, windowsOS=False):
    if windowsOS:
        slash = "\\"
    else:
        slash = "/"

    save_path = dir + slash + 'Plots' + slash + experiment

    colorlist = ["lightslategrey", "lightsteelblue", "darkcyan", "firebrick", "lightcoral"]
    # Plotting results
    art = len(artifact_names)

    performance_smote = []
    errors_smote = []

    for i in range(len(SMOTE_ratios)):
        performance_smote.append([lst[i] for lst in performance_list])
        errors_smote.append([lst[i] for lst in errors_list])

    performance_list_ofLists = performance_smote
    errors_list_ofLists = errors_smote

    for indv_art, name in enumerate(artifact_names):

        for j, performance_list in enumerate(performance_list_ofLists):
            errors_list = errors_list_ofLists[j]

            for i in range(len(performance_list)):
                performance_dict = performance_list[i]
                error_dict = errors_list[i]

                performance_vals = np.array(list(performance_dict.values())[:art])
                error_vals = np.array(list(error_dict.values())[:art])

                X_axis = np.arange(len(model_names)) - 0.3

                plt.bar(x=X_axis + 0.15 * i, height=performance_vals[indv_art, :], width=0.15, color=colorlist[i],
                        label="Aug. ratio = " + str(aug_ratios[i]))
                plt.errorbar(x=X_axis + 0.15 * i, y=performance_vals[indv_art, :], yerr=error_vals[indv_art, :],
                             fmt='.',
                             color='k')

            for i, model in enumerate(model_names):
                if model == 'baseline_perm':
                    model_names[i] = "base-\nline"
                elif model == 'AdaBoost':
                    model_names[i] = "Ada-\nBoost"

        plt.xticks(np.arange(len(model_names)), model_names, rotation=0)
        plt.ylim(0, 1)
        plt.title(
            f"{measure} with {aug_technique} augmentation on the '{name}'-class - SMOTE = {SMOTE_ratios[j] - 1}")
        # plt.xlabel("Model")
        plt.ylabel(measure)
        plt.legend(loc='center right', bbox_to_anchor=(1.36, 0.5))
        plt.subplots_adjust(bottom=0.2, right=0.775)

        if save_img:
            try:
                os.makedirs(save_path)
                print("New directory created!")
            except FileExistsError:
                pass

            plt.savefig(("{}{:s}{}_{}.png").format(save_path + slash + measure, slash, name, experiment))
        plt.show()

def plotResultsChooseMeasure(pickle_path, experiment_name, merged_file, windowsOS=False,
                             measure='sensitivity', save_img=False):
    # Loading statistically calculated results as dictionaries
    # For single files and their HO_trials
    # List of dictionaries of results. Each entry in the list is a results dictionary for one SMOTE ratio
    performance_list_ofLists, errors_list_ofLists, model_names, artifact_names, SMOTE_ratios, aug_ratios = tableResults_Augmentation(
        measure=measure, pickle_path=pickle_path, experiment_name=experiment_name,
        merged_file=merged_file, windowsOS=windowsOS)
    SMOTE_ratios.sort()

    # This function will plot results created in the augmentation experiment (with aug. ratio key in the dict)

    plotResultsAugmentation(performance_list=performance_list_ofLists, errors_list=errors_list_ofLists,
                                 experiment=experiment,
                                 model_names=model_names, artifact_names=artifact_names, SMOTE_ratios=SMOTE_ratios,
                                 aug_ratios=aug_ratios,
                                 save_img=save_img, aug_technique=Aug_technique, windowsOS=windowsOS,
                                 measure=measure)

def printResultsChooseMeasure(pickle_path, experiment_name, merged_file, windowsOS=False,
                              measure='sensitivity', LaTeX=False):
    performance_list_ofLists, errors_list_ofLists, model_names, artifact_names, SMOTE_ratios, aug_ratios = tableResults_Augmentation(
        measure=measure, pickle_path=pickle_path, experiment_name=experiment_name,
        merged_file=merged_file, windowsOS=windowsOS)
    SMOTE_ratios.sort()

    for j in range(len(aug_ratios)):
        print("\n\n")
        print(80 * "#")
        print("Results with augmentation rate set to {:2f}".format(aug_ratios[j]))
        print(80 * "#")

        performance_list = performance_list_ofLists[j]
        errors_list = errors_list_ofLists[j]

        for i, ratio in enumerate(SMOTE_ratios):

            if LaTeX:
                print("Implement LaTeX layout!")

            else:
                performance = performance_list[i]
                errors = errors_list[i]

                # Print dataframes
                df_eval = pd.DataFrame.from_dict(performance)
                df_eval.index = model_names
                print('\nOVERALL PERFORMANCE')
                print("SMOTE RATIO:" + str(ratio - 1) + "\n")
                print(np.round(df_eval * 100, 2))
                # print(df_eval)

                df_eval = pd.DataFrame.from_dict(errors)
                df_eval.index = model_names
                print('\nSTANDARD DEVIATIONS')
                print("SMOTE RATIO:" + str(ratio - 1) + "\n")
                print(np.round(df_eval * 100, 2))

                print("")
                print(80 * "#")


if __name__ == '__main__':
    dir = r"C:\Users\Albert Kjøller\Documents\GitHub\EEG_epilepsia"
    # dir = "/Users/philliphoejbjerg/Documents/GitHub/EEG_epilepsia"
    # pickle_path = r"/Users/Jacobsen/Documents/GitHub/EEG_epilepsia" + "/"
    windowsOS = True

    if windowsOS:
        slash = "\\"
    else:
        slash = "/"

    merged_file = True

    experiment = "smote_f2"  # "DataAug_white_noiseAdd_LR"
    experiment_name = '_smote_f2_GNB'  # end by either _Noise (or _color_Noise), _GAN or _MixUp if Augmentation
    experiment_name_merge = 'smote_f2_first_merge'

    Aug_technique = None  # "GAN"

    pickle_path = dir + slash + "results" + slash + "performance" + slash + experiment
    pickle_path_merge = dir + slash + "results" + slash + "merged_files" + slash + experiment

    # Code for manipulating a single file to i.e. add results from a single artifact.
    # GNB = 'pickle_path + '"GNB_for_merge_smote_f2"
    # manipulateFile(dir_name=GNB, file_name="results_smote_f2_GNB", main_file="results_smote_f2_GNB.npy",
    #               sec_file="results_smote_f2_GNBNull.npy", windowsOS=windowsOS, experiment=experiment)

    if merged_file:

        # Merge individual result-files
        mergeResultFiles(file_path=pickle_path, file_name=experiment_name_merge, windowsOS=windowsOS,
                         merge_smote_files=False, merge_aug_files=False)

        pickle_path = pickle_path_merge

    performance_list_ofLists, errors_list_ofLists, model_names, artifact_names, SMOTE_ratios, aug_ratios = tableResults_Augmentation(
        measure="sensitivity", pickle_path=pickle_path, experiment_name=experiment_name,
        merged_file=merged_file, windowsOS=windowsOS)
    SMOTE_ratios.sort()

    measures = ["sensitivity", "accuracy", "weighted_F2"]

    save_img = False
    LaTeX = False

    for measure in measures:
        plotResultsChooseMeasure(measure=measure,
                                 pickle_path=pickle_path,
                                 experiment_name=experiment_name,
                                 merged_file=merged_file,
                                 windowsOS=windowsOS,
                                 save_img=save_img)

        printResultsChooseMeasure(measure=measure,
                                  pickle_path=pickle_path,
                                  experiment_name=experiment_name,
                                  merged_file=merged_file,
                                  windowsOS=windowsOS,
                                  LaTeX=LaTeX)

    print("")
