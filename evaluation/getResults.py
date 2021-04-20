
import scipy.stats
import numpy as np
from prepData.dataLoader import *
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict
import glob
import pandas as pd

def mergeResultFiles(file_path, file_name="merged", windowsOS=False):
    #TODO: Kunne være fedt at lave, så man vælger hvilke filer der skal merges i stedet for directory

    if windowsOS:
        slash = "\\"
    else:
        slash = "/"

    # Nested dictionary
    all_results_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))


    file_names = [results_file.split(slash)[-1] for results_file in glob.glob(file_path + slash + "**")]

    for model_file in file_names:
        results = LoadNumpyPickles(file_path + slash, model_file, windowsOS=windowsOS)[()]
        smote_ratios = list(results.keys())
        folds = [key for key in results[smote_ratios[0]].keys() if type(key) == int]
        artifacts = list(results[smote_ratios[0]][folds[0]].keys())
        models = list(results[smote_ratios[0]][folds[0]][artifacts[0]].keys())

        for ratio in smote_ratios:
            for fold in folds:
                for artifact in artifacts:
                    for model in models:
                        if model == 'baseline_major':
                            pass
                        else:
                            all_results_dict[ratio][fold][artifact][model] = results[ratio][fold][artifact][model]

    # Save file in merged_files dir
    results_basepath = slash.join(file_path.split(slash)[:-2])

    # Reformating dictionary to avoid lambda call - to be able to save as pickle
    temp = defaultdict(dict)
    smote_ratios = list(all_results_dict.keys())
    for ratio in all_results_dict.keys():
        for fold in all_results_dict[smote_ratios[0]].keys():
            temp[ratio][fold] = all_results_dict[ratio][fold]

    exp = file_path.split(slash)[-1]
    print("\nNew file created!")
    SaveNumpyPickles(results_basepath + slash + "merged_files" + slash + exp, slash + file_name, temp, windowsOS)

def manipulateFile(file_path, file_name="merged", experiment="", main_file="", sec_file="", windowsOS=False):
    # TODO: Kunne være fedt at lave, så man vælger hvilke filer der skal merges i stedet for directory

    if windowsOS:
        slash = "\\"
    else:
        slash = "/"

    # Nested dictionary
    all_results_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    file_names = [results_file.split(slash)[-1] for results_file in glob.glob(file_path + slash + "**")]

    if file_names[0] != main_file:
        file_names = file_names[::-1]

    for model_file in file_names:
        results = LoadNumpyPickles(file_path + slash, model_file, windowsOS=windowsOS)[()]
        smote_ratios = list(results.keys())
        folds = [key for key in results[smote_ratios[0]].keys() if type(key) == int]
        artifacts = list(results[smote_ratios[0]][folds[0]].keys())
        models = list(results[smote_ratios[0]][folds[0]][artifacts[0]].keys())

        for ratio in smote_ratios:
            for fold in folds:
                for artifact in artifacts:
                    for model in models:
                        all_results_dict[ratio][fold][artifact][model] = results[ratio][fold][artifact][model]

    # Save file in merged_files dir
    results_basepath = slash.join(file_path.split(slash)[:-2])

    # Reformating dictionary to avoid lambda call - to be able to save as pickle
    temp = defaultdict(dict)
    smote_ratios = list(all_results_dict.keys())
    for ratio in all_results_dict.keys():
        for fold in all_results_dict[smote_ratios[0]].keys():
            temp[ratio][fold] = all_results_dict[ratio][fold]

    exp = experiment
    print("\nNew file created!")
    SaveNumpyPickles(results_basepath + slash + "performance" + slash + exp, slash + file_name, temp, windowsOS)


def tableResults(pickle_path, windows_OS, experiment_name, merged_file=False, windowsOS=False):
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
        results_all = LoadNumpyPickles(pickle_path=results_basepath + slash + "performance"+ slash + exp, file_name = slash + "results" + experiment_name +'.npy', windowsOS = windowsOS)
        results_all = results_all[()]
        # fold, artifact, model, hyperopt iterations
        HO_trials = LoadNumpyPickles(pickle_path=results_basepath + slash + "hyperopt" + slash + exp, file_name = slash + 'ho_trials' + experiment_name +'.npy', windowsOS = windowsOS)
        HO_trials = HO_trials[()]

    def mean_confidence_interval(data, confidence=0.95):
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
        return m-h, m, m+h

    table_list = []
    table_std_list = []

    for ratio in results_all:
        results = results_all[ratio]

        pd.set_option("display.max_rows", None, "display.max_columns", None)

        table = defaultdict(dict) #{}
        table_std = defaultdict(dict) #{}

        # want to average each fold
        #construct keys
        folds = [key for key in results.keys() if type(key)==int]
        artifacts = list(results[folds[0]].keys())
        models = list(results[folds[0]][artifacts[0]].keys())
        scores = list(results[folds[0]][artifacts[0]][models[0]].keys())

        # row-wise models, column-wise artifacts
        acc = np.zeros((len(models),len(artifacts)))
        acc_std = np.zeros((len(models),len(artifacts)))
        f1s = np.zeros((len(models),len(artifacts)))
        f1s_std = np.zeros((len(models),len(artifacts)))

        for idx_art, artifact in enumerate(artifacts):
            #table[artifact] = []
            store_model = [0]*len(models)
            sensitivity_std = [0]*len(models)

            for idx_mod, model in enumerate(models):

                store_scores = []

                temp_acc = []
                temp_f1 = []
                for fold in folds:
                    store_scores.append(results[fold][artifact][model]['sensitivity'])
                    temp_acc.append(results[fold][artifact][model]['accuracy'])
                    temp_f1.append(results[fold][artifact][model]['weighted_F1'])

                acc[idx_mod,idx_art] = np.mean(temp_acc)
                f1s[idx_mod,idx_art] = np.mean(temp_f1)

                # Standard deviation for acc. and F1 on each artifact
                acc_std[idx_mod, idx_art] = np.std(temp_acc)
                f1s_std[idx_mod, idx_art] = np.std(temp_f1)

                # Store standard deviation for sensitivies for each classifier/ model
                store_model[idx_mod] = np.mean(store_scores)
                sensitivity_std[idx_mod] = np.std(store_scores)

            table[artifact] = store_model
            table_std[artifact] = sensitivity_std


        table['avg. accuracy'] = np.mean(acc,axis=1)
        table['avg. weighted f1'] = np.mean(f1s,axis=1)

        # Mean standard deviation
        table_std['avg. accuracy'] = np.mean(acc_std, axis=1)
        table_std['avg. weighted f1'] = np.mean(f1s_std, axis=1)

        table_list.append(table)
        table_std_list.append(table_std)

    SMOTE_ratios = list(results_all.keys())

    return table_list, table_std_list, models, artifacts, SMOTE_ratios


def plotPerformanceModels(performance_dict, error_dict, experiment, model_names, artifact_names, ratio, save_img=False, windowsOS=False):

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
        plt.bar(x=artifact_names, height=performance_vals[indv_model,:], width=0.5, color="lightsteelblue")
        plt.errorbar(x=artifact_names, y=performance_vals[indv_model,:], yerr=error_vals[indv_model,:], fmt='.', color='k')
        plt.title(name + " - SMOTE RATIO:" + str(ratio-1))
        plt.ylim(0, 1)
        if save_img:
            plt.savefig(("{}{:s}{}_SMOTE_{}.png").format(save_path, slash, name, ratio-1))
        plt.show()

def plotPerformanceClasses(performance_dict, error_dict, experiment, model_names, artifact_names, ratio, save_img=False, windowsOS=False):

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
        plt.title(name + " - SMOTE RATIO:" + str(ratio-1))
        plt.xticks(rotation=25)
        plt.ylim(0,1)
        if save_img:
            plt.savefig(("{}{:s}{}_SMOTE_{}.png").format(save_path, slash, name, ratio-1))
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
                                   file_name=slash+"results" + experiment_name + '.npy', windowsOS=windowsOS)
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

        #TODO: Not completely functioning! It does not show the plots

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


def plotResultsSMOTE(performance_list, errors_list, experiment, model_names, artifact_names, SMOTE_ratios, save_img=False, windowsOS=False):
    if windowsOS:
        slash = "\\"
    else:
        slash = "/"

    save_path = dir + slash +'Plots' + slash + experiment

    colorlist = ["lightslategrey", "lightsteelblue", "darkcyan", "firebrick", "lightcoral"]
    # Plotting results
    art = len(artifact_names)
    for indv_art, name in enumerate(artifact_names):
        for i in range(len(performance_list)):
            performance_dict = performance_list[i]
            error_dict = errors_list[i]

            performance_vals = np.array(list(performance_dict.values())[:art])
            error_vals = np.array(list(error_dict.values())[:art])

            X_axis = np.arange(len(model_names)) - 0.3


            plt.bar(x=X_axis + 0.15 * i, height=performance_vals[indv_art, :], width=0.15, color=colorlist[i],
                    label="SMOTE = " + str(SMOTE_ratios[i]))
            plt.errorbar(x=X_axis + 0.15 * i, y=performance_vals[indv_art, :], yerr=error_vals[indv_art, :],
                         fmt='.',
                         color='k')


        for i, model in enumerate(model_names):
            if model == 'baseline_perm':
                model_names[i] = "base-\nline"
            elif model == 'AdaBoost':
                model_names[i] = "Ada-\nBoost"

        plt.xticks(np.arange(len(model_names)), model_names, rotation=0)
        plt.ylim(0,1)
        plt.title("Sensitiviy scores across models and SMOTEs on the '{:s}'-class".format(name))
        plt.xlabel("Model")
        plt.ylabel("Sensitivity")
        plt.legend(loc='center right', bbox_to_anchor=(1.36, 0.5))
        plt.subplots_adjust(bottom=0.2, right=0.775)

        if save_img:
            try:
                os.makedirs(save_path)
                print("New directory created!")
            except FileExistsError:
                pass

            plt.savefig(("{}{:s}{}_{}.png").format(save_path, slash, name, experiment))
        plt.show()


if __name__ == '__main__':
    dir = r"C:\Users\Albert Kjøller\Documents\GitHub\EEG_epilepsia"
    #dir = "/Users/philliphoejbjerg/Documents/GitHub/EEG_epilepsia"
    # pickle_path = r"/Users/Jacobsen/Documents/GitHub/EEG_epilepsia" + "/"
    windowsOS = True

    if windowsOS:
        slash = "\\"
    else:
        slash = "/"

    merged_file = True

    experiment = "smote"
    experiment_name = '_smote_easy_models'
    experiment_name_merge = 'smote_first_merge'


    pickle_path = dir + slash + "results" + slash + "performance" + slash + experiment
    pickle_path_merge = dir + slash + "results" + slash + "merged_files" + slash + experiment

    GNB = r"C:\Users\Albert Kjøller\Documents\GitHub\EEG_epilepsia\results\performance\GNB_for_merge"
    manipulateFile(file_path=GNB, file_name="results_smote_LR_GNB_baselines", main_file="results_smote_easy_models.npy",
                   sec_file="results_smote_GNB_null.npy", windowsOS=windowsOS, experiment=experiment)

    if merged_file:
        experiment_name = experiment_name_merge

        # Merge individual result-files
        mergeResultFiles(file_path=pickle_path, file_name="smote_first_merge", windowsOS=windowsOS)


    # Loading statistically calculated results as dictionaries
    # For single files and their HO_trials
    # List of dictionaries of results. Each entry in the list is a results dictionary for one SMOTE ratio
    performance_list, errors_list, model_names, artifact_names, SMOTE_ratios = tableResults(pickle_path=pickle_path, windows_OS=windowsOS, experiment_name=experiment_name, merged_file=merged_file, windowsOS=windowsOS)
    SMOTE_ratios.sort()

    # Save plots or not
    save_img = True

    plotResultsSMOTE(performance_list=performance_list, errors_list=errors_list, experiment=experiment,
                     model_names=model_names, artifact_names=artifact_names, SMOTE_ratios=SMOTE_ratios, save_img=save_img,
                     windowsOS=windowsOS)

    # For merged files
    #performance, errors, model_names, artifact_names = tableResults(pickle_path=pickle_path_merge, windows_OS=windowsOS, experiment_name=experiment_name_merge, merged_file=True)
    for i, ratio in enumerate(SMOTE_ratios):
        performance = performance_list[i]
        errors = errors_list[i]

        # Print dataframes
        df_eval = pd.DataFrame.from_dict(performance)
        df_eval.index = model_names
        print('\n\n\nOVERALL PERFORMANCE')
        print("SMOTE RATIO:" + str(ratio-1) +"\n")
        print(np.round(df_eval * 100, 2))
        # print(df_eval)
        """
        df_eval = pd.DataFrame.from_dict(errors)
        df_eval.index = model_names
        print('\nSTANDARD DEVIATIONS')
        print("SMOTE RATIO:" + str(ratio-1)+"\n")
        print(np.round(df_eval * 100, 2))
        """

        # Individual plotting commands! For now outcommented.

        #Across models
        #plotPerformanceModels(performance_dict=performance, error_dict=errors, experiment=experiment, model_names=model_names, artifact_names=artifact_names, ratio=ratio, save_img=save_img, windowsOS=windowsOS)

        #Across classes
        #plotPerformanceClasses(performance_dict=performance, experiment=experiment, error_dict=errors, model_names=model_names, artifact_names=artifact_names, ratio=ratio, save_img=save_img, windowsOS=windowsOS)



        #Plotting Hyperopt queries
        #TODO: BROKEN!
        #plotHyperopt(pickle_path=pickle_path, file_name=experiment_name, windowsOS=windowsOS)


    print("")


'''

model_names = [] # will be appended to

initial_data = {} # to construct dataframe

# if statements are used if the dict is originally empty
# we recursively add the rows for each model and end with overall performance
for model in CV_scores:
    model_names.append(model)

    #initialize for appending, momentary lists
    confidence = {} # confidence interval
    accuracies = []
    weighted_f1s = []

    for artifact in CV_scores[model]: # each artifact
        #add sensitivity to the data
        sensitivities = CV_scores[model][artifact]['sensitivity']

        if artifact_names[artifact] in initial_data:
            initial_data[artifact_names[artifact]].append(sensitivities.mean())
        else:
            initial_data[artifact_names[artifact]] = [sensitivities.mean()]

        minus, mean, plus = mean_confidence_interval(sensitivities)

        confidence[artifact_names[artifact]] = [minus, mean, plus]


        accuracies.append(CV_scores[model][artifact]['accuracy'].mean())
        weighted_f1s.append(CV_scores[model][artifact]['F1'].mean())

    #confidence
    minus, mean, plus = mean_confidence_interval(accuracies)
    confidence['accuracy'] = [minus, mean, plus]
    minus, mean, plus = mean_confidence_interval(weighted_f1s)
    confidence['weighted_f1'] = [minus, mean, plus]

    df_conf = pd.DataFrame.from_dict(confidence)
    df_conf.index = ['lower', 'mean', 'upper']
    df_conf.style.set_caption(model)
    print(df_conf)
    print('\n')

    # overall performance for all artifacts
    if 'accuracy' in initial_data:
        initial_data['accuracy'].append(np.mean(accuracies))
    else:
        initial_data['accuracy'] = [np.mean(accuracies)]

    if 'weighted_f1' in initial_data:
        initial_data['weighted_f1'].append(np.mean(weighted_f1s))
    else:
        initial_data['weighted_f1'] = [np.mean(weighted_f1s)]



#display all dataframe
pd.set_option("display.max_rows", None, "display.max_columns", None)

df_eval = pd.DataFrame.from_dict(initial_data)
df_eval.index = model_names
df_eval.style.set_caption("Overview")
print(df_eval)

print('\n\n')


'''

