from results import *


class ActiveResults:
    def __init__(self, dir, experiment, experiment_name, model='LR', smote_ratio=1, windowsOS=False):

        print("-" * 80)
        self.dir = dir
        self.windowsOS = windowsOS
        self.slash = "\\" if self.windowsOS == True else "/"

        self.experiment = experiment
        self.pickle_path = (self.slash).join([dir, "results", "performance", experiment])

        self.experiment_name = experiment_name
        self.basepath = self.slash.join(self.pickle_path.split(self.slash)[:-2])

        self.aug_ratios = [0, 0.5, 1, 1.5, 2]
        self.smote_ratios = [smote_ratio + 1] # + 1 since the experiment running with smote and result-layout are not consistent
        self.folds = [0, 1, 2, 3, 4]
        self.artifacts = ['eyem', 'chew', 'shiv', 'elpp', 'musc', 'null']
        self.models = ['LR']
        self.scores = ['balance', 'accuracy', 'F2', 'sensitivity']

        print(f"Created object for experiment: {experiment_name}\n")

        super(ActiveResults, self).__init__()

    def extractActiveResults(self):

        # Find files to be merged
        file_path = (self.slash).join([self.basepath, "performance", self.experiment, ""])
        file_names = [results_file.split(self.slash)[-1] for results_file in
                      glob.glob(file_path + "**")]

        if len(file_names) == 0:
            raise FileNotFoundError("There are no files in the specified directory.")

        i = 0
        all_results_dict = {}

        # Extract information from the original pickles from the experiments and save them in a new nested dictionary.
        #file_names.append(file_names[0])
        for model_file in file_names:
            results = LoadNumpyPickles(self.pickle_path + self.slash, model_file, windowsOS=self.windowsOS)[()]

            # Rereference the dictionary. The if-statements are created to ensure that the accumulated dictionary is not
            # overwritten when looping through either aug_ratios, smote_ratios, etc.

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

        return all_results_dict



def plotActiveResults(dictionary, init_percent, n_pr_query_percent, measures=['F2'], aug_ratios=None, control_values=None, control_std=None, aug_method=None):

    if aug_ratios == None:
        aug_ratios = [0,0.5,1,1.5,2]

    artifact_names = ['eyem', 'chew', 'shiv', 'elpp', 'musc', 'null']

    aug_ratios = aug_ratios
    smote_ratios = list(dictionary[aug_ratios[0]].keys())
    folds = list(dictionary[aug_ratios[0]][smote_ratios[0]].keys())
    artifacts = list(dictionary[aug_ratios[0]][smote_ratios[0]][folds[0]].keys())
    models = list(dictionary[aug_ratios[0]][smote_ratios[0]][folds[0]][artifacts[0]].keys())
    scores = list(dictionary[aug_ratios[0]][smote_ratios[0]][folds[0]][artifacts[0]][models[0]].keys())

    if control_values == None:
        control_values = len(artifacts) * [0]
    if control_std == None:
        control_std = len(artifacts) * [0]

    artifact = artifacts[0] #TODO: Currently saving the data without using this index since accumulated scores are wrongly appended to list in pipeline

    offset_errorbar = np.linspace(-0.002, 0.002, len(aug_ratios))

    dict_for_plots = defaultdict(lambda: defaultdict(dict))
    scores_results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for aug_ratio in aug_ratios:
        for smote_ratio in smote_ratios:
            for model in models:
                for score in scores[:3]:
                    for fold in folds:
                        results = dictionary[aug_ratio][smote_ratio][fold][artifact][model][score]
                        if results == {}:
                            pass
                        else:
                            check_artifact = results[0][0]
                            artifact_number = -1

                            for element in results:
                                # Check whether the results are on a new artifact - only necessary due to a f**kup in the pipeline...
                                if element[0] == check_artifact:
                                    i = 0
                                    artifact_number += 1
                                    art = artifact_names[artifact_number]

                                #if fold == 0 :
                                if dict_for_plots[art][init_percent + i * n_pr_query_percent] == {}:
                                    dict_for_plots[art][init_percent + i * n_pr_query_percent] = [element[1]]
                                else:
                                    dict_for_plots[art][init_percent + i * n_pr_query_percent].append(element[1])

                                i += 1


                    for artifact in artifacts:
                        mu, error = [], []
                        for percentage, score_list in dict_for_plots[artifact].items():
                            mu.append((percentage, np.mean(score_list)))
                            error.append((percentage, np.std(score_list)))

                        scores_results[artifact][score][aug_ratio]['mean'] = mu
                        scores_results[artifact][score][aug_ratio]['error'] = error

    color_list = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(aug_ratios)))

    for artifact in artifacts:
        for measure in measures:
            for c, aug_ratio in enumerate(aug_ratios):
                scores_list = scores_results[artifact][measure][aug_ratio]['mean']
                errors_list = scores_results[artifact][measure][aug_ratio]['error']
                errors = np.array([err[1] for err in errors_list])

                #plt.plot(*tuple(np.array(scores_list).T), color=color_list[c])

                xs, ys = tuple(np.array(scores_list).T)

                if aug_method != None:
                    plt.errorbar(xs + offset_errorbar[c], ys, yerr=errors, color=color_list[c], label=f"{aug_method}: {aug_ratio}")
                else:
                    plt.errorbar(xs + offset_errorbar[c], ys, yerr=errors, color=color_list[c], label=f"W/O augmentation")
                plt.ylim(bottom=0, top=1)

            plt.hlines(xmin=init_percent, xmax=init_percent + n_pr_query_percent * (i-1), y = control_values[artifact], colors = 'grey', linestyles = "--", label=f"Control exp. mean: {control_values[artifact]}")
            plt.gca().fill_between(xs, control_values[artifact] - 2 * control_std[artifact], control_values[artifact] + 2 * control_std[artifact], color='lightblue', alpha=0.5,
                                   label=fr"Control exp. dev.: 2 $\sigma$ ($\sigma$={control_std[artifact]})")
            plt.legend(loc='lower left')
            plt.xlabel("Percentage of pool used in training data")
            plt.ylabel(f"Performance: {measure}")
            plt.title(f"{artifact}, SMOTE: {smote_ratio-1}")
            savepath = ("\\").join(dir.split("\\")[:-1]) + r"\Plots" + r"\active_plots"
            os.makedirs(savepath, exist_ok=True)
            plt.savefig(savepath + fr"\SMOTE{smote_ratio}_init{init_percent}_query{n_pr_query_percent}_art{artifact}_metric{measure}.png")
            plt.show()


def plotActiveBalance(dictionary, init_percent, n_pr_query_percent, measures=['F2'], aug_method=None):

    if aug_method == None:
        raise AttributeError("Augmentation technique not specified!")

    artifact_names = ['eyem', 'chew', 'shiv', 'elpp', 'musc', 'null']

    aug_ratios = list(dictionary.keys())
    smote_ratios = list(dictionary[aug_ratios[0]].keys())
    folds = list(dictionary[aug_ratios[0]][smote_ratios[0]].keys())
    artifacts = list(dictionary[aug_ratios[0]][smote_ratios[0]][folds[0]].keys())
    models = list(dictionary[aug_ratios[0]][smote_ratios[0]][folds[0]][artifacts[0]].keys())
    scores = list(dictionary[aug_ratios[0]][smote_ratios[0]][folds[0]][artifacts[0]][models[0]].keys())

    artifact = artifacts[0]  # TODO: Currently saving the data without using this index since accumulated scores are wrongly appended to list in pipeline

    offset_errorbar = np.linspace(-0.002, 0.002, len(aug_ratios))

    dict_for_plots = defaultdict(lambda: defaultdict(dict))
    scores_results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    score = 'balance'

    for aug_ratio in aug_ratios:
        for smote_ratio in smote_ratios:
            for model in models:
                for fold in folds:
                    results = dictionary[aug_ratio][smote_ratio][fold][artifact][model][score]
                    if results == {}:
                        pass
                    else:
                        check_artifact = results[0][0]
                        artifact_number = -1

                        for element in results:
                            # Check whether the results are on a new artifact - only necessary due to a f**kup in the pipeline...
                            if element[0] == check_artifact:
                                i = 0
                                artifact_number += 1
                                art = artifact_names[artifact_number]

                            # if fold == 0 :
                            if dict_for_plots[art][init_percent + i * n_pr_query_percent] == {}:
                                dict_for_plots[art][init_percent + i * n_pr_query_percent] = [element[1]]
                            else:
                                dict_for_plots[art][init_percent + i * n_pr_query_percent].append(element[1])

                            i += 1

                for artifact in artifacts:
                    mu, error = [], []
                    for percentage, score_list in dict_for_plots[artifact].items():
                        mu.append((percentage, np.mean(score_list)))
                        error.append((percentage, np.std(score_list)))

                    scores_results[artifact][score][aug_ratio]['mean'] = mu
                    scores_results[artifact][score][aug_ratio]['error'] = error

    color_list = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(aug_ratios)))

    for artifact in artifacts:
        for c, aug_ratio in enumerate(aug_ratios):
            scores_list = scores_results[artifact][score][aug_ratio]['mean']
            errors_list = scores_results[artifact][score][aug_ratio]['error']
            errors = np.array([err[1] for err in errors_list])

            plt.plot(*tuple(np.array(scores_list).T), color=color_list[c])
            #plt.ylim(bottom=0, top=1)
            xs, ys = tuple(np.array(scores_list).T)

            plt.errorbar(xs + offset_errorbar[c], ys, yerr=errors, color=color_list[c],
                         label=f"{aug_method}: {aug_ratio}")

        plt.legend(loc='lower left')
        plt.xlabel("Percentage of pool used in training data")
        plt.ylabel(f"Balance (present / absent)")
        plt.title(f"{artifact}")
        plt.show()



if __name__ == '__main__':
    dir = r"C:\Users\Albert Kjøller\Documents\GitHub\EEG_epilepsia\true_pickles"  # dir = "/Users/philliphoejbjerg/Documents/GitHub/EEG_epilepsia"  # dir = r"/Users/Jacobsen/Documents/GitHub/EEG_epilepsia" + "/"

    # Example of merging fully created files from different models.
    experiment = "activeefficiency_experiment"  # directory containing the files we will look at
    experiment_name = '_ActiveImprovement_pilot'

    activeImprovement = ActiveResults(dir, experiment, experiment_name, smote_ratio=0, model='LR', windowsOS=True)
    active_dict = activeImprovement.extractActiveResults()

    control_values_artifact = {'eyem': 0.8, 'chew': 0.93, 'shiv': 0.91, 'elpp': 0.8, 'musc': 0.82, 'null': 0.76}
    control_std = {'eyem': 0.02, 'chew': 0.02, 'shiv': 0.03, 'elpp': 0.03, 'musc': 0.04, 'null': 0.03}
    plotActiveResults(active_dict, init_percent=0.1, n_pr_query_percent=0.01, aug_ratios=[0], control_values=control_values_artifact, control_std=control_std, measures=['F2'], aug_method=None)

    control_values_artifact = {'eyem': 0.72, 'chew': 0.81, 'shiv': 0.49, 'elpp': 0.49, 'musc': 0.6, 'null': 0.75}
    control_std = {'eyem': 0.04, 'chew': 0.06, 'shiv': 0.34, 'elpp': 0.11, 'musc': 0.11, 'null': 0.03}
    plotActiveResults(active_dict, init_percent=0.1, n_pr_query_percent=0.01, aug_ratios=[0], control_values=control_values_artifact, control_std=control_std, measures=['sensitivity'], aug_method=None)


    experiment = "act"
    experiment_name = "multiple_augRatios"
    activeImprovement = ActiveResults(dir, experiment, experiment_name, smote_ratio=1, model='LR', windowsOS=True)
    active_dict = activeImprovement.extractActiveResults()

    control_values_artifact = {'eyem': 0.8, 'chew': 0.93, 'shiv': 0.91, 'elpp': 0.8, 'musc': 0.82, 'null': 0.76}
    control_std = {'eyem': 0.02, 'chew': 0.02, 'shiv': 0.03, 'elpp': 0.03, 'musc': 0.04, 'null': 0.03}
    plotActiveResults(active_dict, init_percent=0.1, n_pr_query_percent=0.05,
                      control_values=control_values_artifact, control_std=control_std, measures=['F2'],
                      aug_method='MixUp')

    plotActiveBalance(active_dict, init_percent=0.1, n_pr_query_percent=0.05, measures=['F2'], aug_method='MixUp')

    experiment = "efficiency_experiment"  # directory containing the files we will look at
    experiment_name = '_Active_pilot'

    active = getResults(dir, experiment, experiment_name, merged_file=True, windowsOS=True)
    active.mergeResultFiles(file_name=experiment_name)

    # To work with the merged file we have to change the pickle path to the "merged" folder.
    active.changePicklePath()
    performances, errors = active.tableResults_Augmentation(experiment_name=experiment_name, measure="sensitivity")

    artifacts = active.artifacts
    smote_ratio = 1
    models = active.models  # 'GNB'

    y_pred_dict = active.getPredictions(models=models,
                                           aug_ratios=[0],
                                           smote_ratios=[smote_ratio],
                                           artifacts=artifacts)
    y_pred_dict = active.compressDict(y_pred_dict, smote_ratio=1, aug_ratio=0)

    # fullSMOTE.printScores(pred_dict=y_pred_dict, y_true_filename="y_true_5fold_randomstate_0", model='LDA',
    #                      artifacts=artifacts, ensemble=False, print_confusion=True)

    active.plot_multi_label_confusion(pred_dict=y_pred_dict, y_true_filename="y_true_5fold_randomstate_0",
                                         models=models,
                                         artifacts=active.artifacts, smote_ratio=smote_ratio - 1, ensemble=False)

    # Kør forsøg med n_queries på 0.25 % af poolen og op til 30% af poolen.

    print("")