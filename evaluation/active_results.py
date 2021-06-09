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



def plotActiveResults(dictionary, init_percent, n_pr_query_percent):

    artifact_names = ['eyem', 'chew', 'shiv', 'elpp', 'musc', 'null']

    aug_ratios = list(dictionary.keys())
    smote_ratios = list(dictionary[aug_ratios[0]].keys())
    folds = list(dictionary[aug_ratios[0]][smote_ratios[0]].keys())
    artifacts = list(dictionary[aug_ratios[0]][smote_ratios[0]][folds[0]].keys())
    models = list(dictionary[aug_ratios[0]][smote_ratios[0]][folds[0]][artifacts[0]].keys())
    scores = list(dictionary[aug_ratios[0]][smote_ratios[0]][folds[0]][artifacts[0]][models[0]].keys())

    artifact = artifacts[0] #TODO: Currently saving the data without using this index since accumulated scores are wrongly appended to list in pipeline

    dict_for_plots = defaultdict(lambda: defaultdict(dict))

    for aug_ratio in aug_ratios:
        for smote_ratio in smote_ratios:
            for model in models:

                for score in scores:
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

                                if fold == 0:
                                    dict_for_plots[art][init_percent + i * n_pr_query_percent] = [element[1]]
                                else:
                                    dict_for_plots[art][init_percent + i * n_pr_query_percent].append(element[1])

                                i += 1

                    scores_results = defaultdict(defaultdict(defaultdict(dict)))
                    for artifact in artifacts:
                        mu, error = [], []
                        for percentage, score_list in dict_for_plots[artifact].items():
                            mu.append((percentage, np.mean(score_list)))
                            error.append((percentage, np.std(score_list)))

                        scores_results[artifact][score][aug_ratio]['mean'] = mu
                        scores_results[artifact][score][aug_ratio]['error'] = error

                    print("CALCULATE MEAN AND STD.")

                    print("")

        acc_over_aug = []
        F2_over_aug = []
        sens_over_aug = []

    # Accumulated results with active learning
    plt.plot(*tuple(np.array(testacc_al).T))
    plt.plot(*tuple(np.array(testF2_al).T))
    plt.plot(*tuple(np.array(testSens_al).T))
    plt.legend(('Accuracy', 'F2', 'Sensitivity'))
    plt.show()

if __name__ == '__main__':
    dir = r"C:\Users\Albert Kjøller\Documents\GitHub\EEG_epilepsia\true_pickles"  # dir = "/Users/philliphoejbjerg/Documents/GitHub/EEG_epilepsia"  # dir = r"/Users/Jacobsen/Documents/GitHub/EEG_epilepsia" + "/"

    # Example of merging fully created files from different models.
    experiment = "activeefficiency_experiment"  # directory containing the files we will look at
    experiment_name = '_ActiveImprovement_pilot'

    activeImprovement = ActiveResults(dir, experiment, experiment_name, model='LR', windowsOS=True)
    active_dict = activeImprovement.extractActiveResults()


    plotActiveResults(active_dict, init_percent=0.1, n_pr_query_percent=0.2)

    experiment = "efficiency_experiment"  # directory containing the files we will look at
    experiment_name = '_Active_pilot'

    active = getResults(dir, experiment, experiment_name, merged_file=True, windowsOS=True)
    active.mergeResultFiles(file_name=experiment_name)

    # To work with the merged file we have to change the pickle path to the "merged" folder.
    fullSMOTE.changePicklePath()
    performances, errors = fullSMOTE.tableResults_Augmentation(experiment_name=experiment_name, measure="sensitivity")

    artifacts = fullSMOTE.artifacts
    smote_ratio = 1
    models = fullSMOTE.models  # 'GNB'

    y_pred_dict = fullSMOTE.getPredictions(models=models,
                                           aug_ratios=[0],
                                           smote_ratios=[smote_ratio],
                                           artifacts=artifacts)
    y_pred_dict = fullSMOTE.compressDict(y_pred_dict, smote_ratio=1, aug_ratio=0)

    # fullSMOTE.printScores(pred_dict=y_pred_dict, y_true_filename="y_true_5fold_randomstate_0", model='LDA',
    #                      artifacts=artifacts, ensemble=False, print_confusion=True)

    fullSMOTE.plot_multi_label_confusion(pred_dict=y_pred_dict, y_true_filename="y_true_5fold_randomstate_0",
                                         models=models,
                                         artifacts=fullSMOTE.artifacts, smote_ratio=smote_ratio - 1, ensemble=False)

    # Kør forsøg med n_queries på 0.25 % af poolen og op til 30% af poolen.

    print("")