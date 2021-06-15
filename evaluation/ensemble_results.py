from results import *


class getResultsEnsemble:
    def __init__(self, dir, experiments, experiment_name, windowsOS=False, merged_file=False):

        print("-" * 80)
        self.dir = dir
        self.windowsOS = windowsOS
        self.slash = "\\" if self.windowsOS == True else "/"
        self.experiments = experiments

        self.pickle_path = (self.slash).join([dir, "results", "performance"])
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

        super(getResultsEnsemble, self).__init__()

    def findNBestModels(self, dir, experiment_name, experiments, y_true_path, N_best=None, measure='balanced_acc',
                        windowsOS=True) -> object:
        # Initialize parameters and environment
        results_object = getResults(dir, list(experiments.keys())[0], experiment_name, merged_file=True, windowsOS=True)

        artifact_scores = {artifact: [] for artifact in results_object.artifacts}
        artifact_errors = {artifact: [] for artifact in results_object.artifacts}
        artifact_models = {artifact: [] for artifact in results_object.artifacts}
        artifact_technique = {artifact: [] for artifact in results_object.artifacts}
        artifact_augRatios = {artifact: [] for artifact in results_object.artifacts}
        artifact_smoteRatios = {artifact: [] for artifact in results_object.artifacts}

        for experiment, (smote_ratio, technique) in experiments.items():
            results_object = getResults(dir, experiment, experiment_name, merged_file=True, windowsOS=True)
            results_object.mergeResultFiles(file_name=experiment_name)

            results_object.changePicklePath()

            performances, errors = results_object.tableResults_Augmentation(experiment_name=experiment_name,
                                                                            smote_ratios=[smote_ratio],
                                                                            y_true_path=y_true_path, measure=measure)

            for artifact in results_object.artifacts:
                aug_ratiosCopy = results_object.aug_ratios.copy()
                if technique == 'control':
                    aug_ratiosCopy = [0]

                for aug_ratio in aug_ratiosCopy:
                    for i, model in enumerate(results_object.models):
                        score = performances[aug_ratio][smote_ratio][artifact][i]
                        error = errors[aug_ratio][smote_ratio][artifact][i]

                        artifact_scores[artifact].append(score)
                        artifact_errors[artifact].append(error)
                        artifact_models[artifact].append(model)
                        artifact_augRatios[artifact].append(aug_ratio)
                        artifact_smoteRatios[artifact].append(smote_ratio)
                        artifact_technique[artifact].append(technique)

        for artifact in results_object.artifacts:
            order = np.argsort(-np.array(artifact_scores[artifact]))

            if N_best == None:
                n = len(order)
            else:
                n = N_best

            best_scores = np.take(artifact_scores[artifact], order[:n])
            belonging_error = np.take(artifact_errors[artifact], order[:n])
            belonging_model = np.take(artifact_models[artifact], order[:n])
            belonging_aug = np.take(artifact_augRatios[artifact], order[:n])
            belonging_smote = np.take(artifact_smoteRatios[artifact], order[:n])
            belonging_technique = np.take(artifact_technique[artifact], order[:n])

            artifact_scores[artifact] = best_scores
            artifact_errors[artifact] = belonging_error
            artifact_models[artifact] = belonging_model
            artifact_augRatios[artifact] = belonging_aug
            artifact_smoteRatios[artifact] = belonging_smote
            artifact_technique[artifact] = belonging_technique

        output_dict = {'scores': artifact_scores, 'errors': artifact_errors,
                       'models': artifact_models, 'augRatios': artifact_augRatios,
                       'smote_ratios': artifact_smoteRatios, 'technique': artifact_technique}

        save_path = ("\\").join([dir, "results", experiment_name])
        os.makedirs(save_path, exist_ok=True)

        if N_best == None:
            N_best = "AllBest"

        SaveNumpyPickles(save_path, results_object.slash + f"orderedPredictions_{N_best}{measure}", output_dict,
                         windowsOS=windowsOS)

        return output_dict

    def getPredictionsEnsemble(self, best_pred_dict, experiments, artifacts=None, N_best=20, withFolds=False):

        if artifacts is None:
            artifacts = self.artifacts

        fileList = []
        new_dict = dict([(value, key) for key, value in experiments.items()])

        for artifact in artifacts:
            best_techs = best_pred_dict['technique'][artifact][:N_best]
            best_smotes = best_pred_dict['smote_ratios'][artifact][:N_best]

            for i in range(N_best):
                fileList.append(new_dict[(best_smotes[i], best_techs[i])])

        fileList = np.unique(fileList)

        helper = getResults(self.dir, self.experiment_name, self.experiment_name, merged_file=True, windowsOS=True)
        helper.mergeResultFiles(file_name=self.experiment_name, ensemble_files=fileList)

        #TODO: at merge virker ikke helt endnu - det er lidt dumt, da der ikke er en key til aug-method
        #TODO: så dictionary kommer til at override tidligere brugte modeller, hvis de bruger samme ratios...


        # Specifies basepath
        results_basepath = self.slash.join(self.pickle_path.split(self.slash)[:-1])
        exp = experiment_name

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

                        #TODO: Måske skal der en key der hedder aug_technique på her?
                        y_pred_dict[aug_ratio][smote_ratio][artifact][model] = y_pred

        return y_pred_dict

    def printNBestModels(self, bestDict, N_best=20, exclude_baseline=False, exclude_zero_aug=False):

        for artifact in self.artifacts:
            print("\n" + 100 * "-")
            print(f"Artifact: {artifact}\n")

            i = 0
            stop = 0
            while stop < N_best:
                tech = bestDict['technique'][artifact][i]
                if exclude_zero_aug == False:
                    if exclude_baseline:
                        if bestDict['models'][artifact][i] != 'baseline_perm':
                            if len(bestDict['models'][artifact][i]) > 2:
                                print(f"{stop})\t{bestDict['models'][artifact][i]} \t(SMOTE {bestDict['smote_ratios'][artifact][i] - 1}): "
                                      f"{tech}  \t\t({bestDict['augRatios'][artifact][i]}) = {bestDict['scores'][artifact][i]}")

                            else:
                                print(f"{stop})\t{bestDict['models'][artifact][i]} \t\t(SMOTE {bestDict['smote_ratios'][artifact][i] - 1}): "
                                      f"{tech} \t\t({bestDict['augRatios'][artifact][i]}) = {bestDict['scores'][artifact][i]}")

                            i += 1
                            stop += 1

                        else:
                            i += 1


                    else:
                        if len(bestDict['models'][artifact][i]) > 2:
                            print(f"{stop})\t{bestDict['models'][artifact][i]} \t(SMOTE {bestDict['smote_ratios'][artifact][i] - 1}): "
                                  f"{tech}  \t\t({bestDict['augRatios'][artifact][i]}) = {bestDict['scores'][artifact][i]}")
                        else:
                            print(f"{stop})\t{bestDict['models'][artifact][i]} \t\t(SMOTE {bestDict['smote_ratios'][artifact][i] - 1}): "
                                  f"{tech} \t\t({bestDict['augRatios'][artifact][i]}) = {bestDict['scores'][artifact][i]}")
                        i += 1
                        stop += 1
            else:
                if exclude_baseline:
                    if bestDict['augRatios'][artifact][i] == 0:
                        if bestDict['models'] == 'control':
                            if len(bestDict['models'][artifact][i]) > 2:
                                print(
                                    f"{stop})\t{bestDict['models'][artifact][i]} \t(SMOTE {bestDict['smote_ratios'][artifact][i] - 1}): "
                                    f"{tech}  \t({bestDict['augRatios'][artifact][i]}) = {bestDict['scores'][artifact][i]}")
                                stop += 1
                            else:
                                print(
                                    f"{stop}){bestDict['models'][artifact][i]} \t\t(SMOTE {bestDict['smote_ratios'][artifact][i] - 1}): "
                                    f"{tech} ({bestDict['augRatios'][artifact][i]}) = {bestDict['scores'][artifact][i]}")
                                stop += 1

                        i += 1
                    else:
                        if bestDict['models'][artifact][i] != 'baseline_perm':
                            if len(bestDict['models'][artifact][i]) > 2: #tabs / indentation
                                print(
                                    f"{stop})\t{bestDict['models'][artifact][i]} \t(SMOTE {bestDict['smote_ratios'][artifact][i] - 1}): "
                                    f"{tech}  \t\t({bestDict['augRatios'][artifact][i]}) = {bestDict['scores'][artifact][i]}")

                            else:
                                print(
                                    f"{stop})\t{bestDict['models'][artifact][i]} \t\t(SMOTE {bestDict['smote_ratios'][artifact][i] - 1}): "
                                    f"{tech} \t\t({bestDict['augRatios'][artifact][i]}) = {bestDict['scores'][artifact][i]}")

                            i += 1
                            stop += 1

                        else:
                            i += 1


                else:
                    if bestDict['augRatios'][artifact][i] == 0:
                        i += 1
                    else:
                        if len(bestDict['models'][artifact][i]) > 2:
                            print(f"{stop})\t{bestDict['models'][artifact][i]} \t(SMOTE {bestDict['smote_ratios'][artifact][i] - 1}): "
                                  f"{tech}  \t\t({bestDict['augRatios'][artifact][i]}) = {bestDict['scores'][artifact][i]}")
                        else:
                            print(f"{stop})\t{bestDict['models'][artifact][i]} \t\t(SMOTE {bestDict['smote_ratios'][artifact][i] - 1}): "
                                  f"{tech} \t\t({bestDict['augRatios'][artifact][i]}) = {bestDict['scores'][artifact][i]}")
                        i += 1
                        stop += 1

if __name__ == '__main__':
    dir = r"C:\Users\Albert Kjøller\Documents\GitHub\EEG_epilepsia"  # dir = "/Users/philliphoejbjerg/Documents/GitHub/EEG_epilepsia"  # dir = r"/Users/Jacobsen/Documents/GitHub/EEG_epilepsia" + "/"
    y_true_path = r"C:\Users\Albert Kjøller\Documents\GitHub\EEG_epilepsia\results\y_true\y_true_5fold_randomstate_0.npy"
    windowsOS = True
    if windowsOS:
        slash = "\\"
    else:
        slash = "/"

    # Experiments is a dictionary of experiment-name as key, the SMOTE ratio and abbreviation of aug-technique as
    # the values used in the experiment.
    experiments = {"SMOTE": (1, 'control'),
                   "augmentation_colorNoise": (1, 'color'), "augmentation_whiteNoise": (1, 'white'),
                   "augmentation_MixUp": (1, 'MixUp'), "augmentation_GAN": (1, 'GAN'),
                   "colorNoiseimprovement": (2, 'color'), "whiteNoiseimprovement": (2, 'white'),
                   "MixUpimprovement": (2, 'MixUp'), "GANimprovement": (2, 'GAN')}

    experiment_name = "_ensemble_experiment"
    ensembleExp = getResultsEnsemble(dir, experiments=experiments, experiment_name=experiment_name, merged_file=False, windowsOS=windowsOS)
    measure = 'balanced_acc'
    N_best = None

    loadedBestDictName = slash + "orderedPredictions_AllBestbalanced_acc.npy"
    bestDictPicklepath = (slash).join([dir, "results", experiment_name])

    if loadedBestDictName == None:
        bestDict = ensembleExp.findNBestModels(dir=dir, experiment_name=experiment_name,
                                               experiments=experiments, y_true_path=y_true_path,
                                               N_best=N_best,
                                               measure=measure, windowsOS=windowsOS)
    else:
        bestDict = LoadNumpyPickles(pickle_path=bestDictPicklepath, file_name=loadedBestDictName, windowsOS=windowsOS)[()]

    ensembleExp.printNBestModels(bestDict=bestDict, N_best=20, exclude_baseline=True)

    y_pred_dict = ensembleExp.getPredictionsEnsemble(best_pred_dict=bestDict, experiments=experiments, N_best=20, artifacts=['eyem'])

    y_pred_dict = fullSMOTE.compressDict(y_pred_dict, smote_ratio=1, aug_ratio=0)

    corr_matrix = fullSMOTE.getCorrelation(artifact='eyem')
    MI = fullSMOTE.getMutualInformation(artifact='eyem')
    print("Mutual Information:\n" + str(MI))

    y_pred_dict_sub = fullSMOTE.getPredictions(  # models=['LDA', 'GNB', 'MLP', 'LR', 'SGD'],
        aug_ratios=[0],
        withFolds=False)

    # Choose smote and aug-ratio
    y_pred_dict_sub = fullSMOTE.compressDict(y_pred_dict_sub, smote_ratio=1, aug_ratio=0)
    # fullSMOTE.printScores(pred_dict=y_pred_dict_sub, model='LDA', y_true_filename="y_true_randomstate_0")

    # The input should be a list of models, a list of aug_ratios for each model and a list of smote_ratios for each
    # model. For future experiments it should take a list with augmentation_techniques in as well.
    ensemble_pred_dict = fullSMOTE.EnsemblePredictions(['LDA', 'GNB', 'MLP', 'LR', 'SGD'], [0, 0, 0, 0, 0],
                                                       [1, 1, 1, 1, 1], withFolds=False)

    # TODO: Implement function to calculate standard error of the ensemble method!
    fullSMOTE.printScores(pred_dict=ensemble_pred_dict, y_true_filename="y_true_randomstate_0", ensemble=True)
