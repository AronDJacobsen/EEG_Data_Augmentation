from results import *
from tqdm import tqdm
import seaborn as sns


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
                        windowsOS=True, withPreds=False, withFolds=False) -> object:
        # Initialize parameters and environment
        results_object = getResults(dir, list(experiments.keys())[0], experiment_name, merged_file=True, windowsOS=True)

        artifact_scores = {artifact: [] for artifact in results_object.artifacts}
        artifact_errors = {artifact: [] for artifact in results_object.artifacts}
        artifact_models = {artifact: [] for artifact in results_object.artifacts}
        artifact_technique = {artifact: [] for artifact in results_object.artifacts}
        artifact_augRatios = {artifact: [] for artifact in results_object.artifacts}
        artifact_smoteRatios = {artifact: [] for artifact in results_object.artifacts}
        artifact_predictions = {artifact: [] for artifact in results_object.artifacts}

        for experiment, (smote_ratio, technique) in tqdm(experiments.items()):
            results_object = getResults(dir, experiment, experiment_name, merged_file=True, windowsOS=True)
            results_object.mergeResultFiles(file_name=experiment_name)

            results_object.changePicklePath()

            performances, errors, predictions = results_object.tableResults_Augmentation(
                experiment_name=experiment_name,
                smote_ratios=[smote_ratio], store_preds=withPreds,
                y_true_path=y_true_path, measure=measure, withFolds=withFolds)

            for artifact in results_object.artifacts:
                aug_ratiosCopy = results_object.aug_ratios.copy()
                if technique == 'control':
                    aug_ratiosCopy = [0]

                for aug_ratio in aug_ratiosCopy:
                    if aug_ratio != 0 or technique == 'control':
                        for i, model in enumerate(results_object.models):
                            score = performances[aug_ratio][smote_ratio][artifact][i]
                            error = errors[aug_ratio][smote_ratio][artifact][i]
                            preds = predictions[aug_ratio][smote_ratio][artifact][i]

                            artifact_scores[artifact].append(score)
                            artifact_errors[artifact].append(error)
                            artifact_models[artifact].append(model)
                            artifact_augRatios[artifact].append(aug_ratio)
                            artifact_smoteRatios[artifact].append(smote_ratio)
                            artifact_technique[artifact].append(technique)
                            artifact_predictions[artifact].append(preds)

        for artifact in tqdm(results_object.artifacts):
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
            belonging_preds = np.take(artifact_predictions[artifact], order[:n], axis=0)

            artifact_scores[artifact] = best_scores
            artifact_errors[artifact] = belonging_error
            artifact_models[artifact] = belonging_model
            artifact_augRatios[artifact] = belonging_aug
            artifact_smoteRatios[artifact] = belonging_smote
            artifact_technique[artifact] = belonging_technique
            artifact_predictions[artifact] = belonging_preds

        output_dict = {'scores': artifact_scores, 'errors': artifact_errors,
                       'models': artifact_models, 'augRatios': artifact_augRatios,
                       'smote_ratios': artifact_smoteRatios, 'technique': artifact_technique,
                       'predictions': artifact_predictions}

        save_path = ("\\").join([dir, "results", experiment_name])
        os.makedirs(save_path, exist_ok=True)

        if N_best == None:
            N_best = "AllBest"

        SaveNumpyPickles(save_path, results_object.slash + f"orderedPredictions_{N_best}{measure}_folds{withFolds}",
                         output_dict,
                         windowsOS=windowsOS)

        return output_dict

    def getPredictionsEnsemble(self, best_pred_dict, experiments, artifacts=None, N_best=20, withFolds=False):

        if artifacts is None:
            artifacts = self.artifacts

        experimentList = []
        new_dict = dict([(value, key) for key, value in experiments.items()])

        for artifact in artifacts:
            best_techs = best_pred_dict['technique'][artifact]  # [:N_best]
            best_smotes = best_pred_dict['smote_ratios'][artifact]  # [:N_best]

            for i in range(N_best):
                experimentList.append(new_dict[(best_smotes[i], best_techs[i])])

        experimentList = np.unique(experimentList)

        helper = getResults(self.dir, self.experiment_name, self.experiment_name, merged_file=True, windowsOS=True)
        ensemble_results = helper.mergeResultFiles(file_name=self.experiment_name, ensemble_experiments=experimentList)

        # For all folds to get predictions of all data points.
        y_pred_dict = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))))
        for ensemble_exp in experimentList:
            # aug_ratios = list(ensemble_results[ensemle_exp].keys())
            for aug_ratio in self.aug_ratios:
                for smote_ratio in [1, 2]:
                    for artifact in self.artifacts:
                        for model in self.models:
                            y_pred = []

                            technique = experiments[ensemble_exp][1]

                            model_pos = np.where(best_pred_dict['models'][artifact] == model)[0]
                            augPos = np.where(best_pred_dict['augRatios'][artifact] == aug_ratio)[0]
                            smotePos = np.where(best_pred_dict['smote_ratios'][artifact] == smote_ratio)[0]
                            tech_pos = np.where(best_pred_dict['technique'][artifact] == technique)[0]

                            temp = np.intersect1d(model_pos, augPos)
                            temp = np.intersect1d(temp, smotePos)
                            temp = np.intersect1d(temp, tech_pos)

                            if temp < N_best:
                                for fold in self.folds:
                                    # if results_all[aug_ratio][smote_ratio][fold][artifact][model] != defaultdict(dict): # if model is not in merged files

                                    y_pred_fold = \
                                        ensemble_results[ensemble_exp][aug_ratio][smote_ratio][fold][artifact][model][
                                            'y_pred']

                                    y_pred.append(y_pred_fold)

                                    if fold == 0:
                                        name = (artifact, model, smote_ratio - 1, technique, aug_ratio)

                            else:
                                pass

                            try:
                                trial = np.concatenate(y_pred)
                                if withFolds == False:
                                    y_pred = trial

                            except ValueError:
                                y_pred = np.nan

                            if temp < N_best:
                                # y_pred_dict[technique][aug_ratio][smote_ratio][artifact][model] = y_pred
                                y_pred_dict[artifact][technique][aug_ratio][smote_ratio][model] = y_pred
                                print(name)

        return y_pred_dict

    def compressDict(self, pred_dict):

        pred_dict_new = defaultdict(dict)

        for artifact in self.artifacts:
            techniques = list(pred_dict[artifact].keys())
            name_pred_list = []
            for technique in techniques:
                for aug_ratio in self.aug_ratios:
                    for smote_ratio in self.smote_ratios:
                        for model in self.models:
                            try:
                                y_pred = pred_dict[artifact][technique][aug_ratio][smote_ratio][model]

                                if np.any(y_pred == {}):
                                    pass
                                else:
                                    name = f"{model}_{technique}{aug_ratio}SMOTE{smote_ratio}"
                                    name_pred_list.append((name, y_pred))
                                    # np.all(y_pred_old == y_pred) # TO CHECK SOMETHING WITH PREDICs.
                                    # np.all(name_old == name)

                                    name_old = name
                                    y_pred_old = y_pred


                            except KeyError:
                                pass
            pred_dict_new[artifact] = name_pred_list
            # pred_dict_new[technique][artifact] = pred_dict[technique][artifact]

        return pred_dict_new

    def getCorrelation(self, bestDict, artifact=None, N_best=20, latex=False, withFolds=False):

        if artifact != None:

            preds = bestDict['predictions'][artifact][:N_best]
            model_names = np.arange(N_best) + 1

            if withFolds:
                pred_folds = np.transpose(preds)
                folds = len(pred_folds)

                corr_matrixList = [np.corrcoef(pred_folds[i].tolist()) for i in range(folds)]
                corr_matrix = pd.DataFrame(np.mean(corr_matrixList, axis=0), columns=model_names, index=model_names)

            else:
                corr_matrix = pd.DataFrame(np.corrcoef(preds), columns=model_names, index=model_names)

            sns.heatmap(corr_matrix, cmap='Blues')
            plt.title(artifact)
            plt.show()

            print("#" * 80)
            print(f"Correlation matrix: class = {artifact}")
            if latex:
                print(corr_matrix.to_latex())
            else:
                print(corr_matrix)

            return corr_matrix

        else:
            corr_matrix_dict = defaultdict(dict)

            fig, axs = plt.subplots(2, 3)  # , sharex=True, sharey=True)

            cbar_ax = fig.add_axes([0.89, .3, .03, .4])
            plt.subplots_adjust(left=0.1, right=0.85)

            for i, artifact in enumerate(self.artifacts):
                j = 0

                if i > 2:
                    j = 1

                preds = bestDict['predictions'][artifact][:N_best]
                model_names = np.arange(N_best) + 1

                if withFolds:
                    pred_folds = np.transpose(preds)
                    folds = len(pred_folds)

                    corr_matrixList = [np.corrcoef(pred_folds[i].tolist()) for i in range(folds)]
                    corr_matrix = pd.DataFrame(np.mean(corr_matrixList, axis=0), columns=model_names, index=model_names)

                else:
                    corr_matrix = pd.DataFrame(np.corrcoef(preds), columns=model_names, index=model_names)

                corr_matrix_dict[artifact] = corr_matrix
                corr_matrix = sns.heatmap(corr_matrix, annot=False, cmap=plt.cm.Blues,
                                          ax=axs[j, i % 3], cbar_ax=cbar_ax,
                                          vmin=0, vmax=1)
                corr_matrix.set_aspect('equal', 'box')

                axs[j, 0].tick_params()
                axs[j, 1].tick_params()

                if i % 3 == 0:
                    corr_matrix.set_ylabel("Sorted best models")
                if j == 1:
                    corr_matrix.set_xlabel("Sorted best models")

                corr_matrix.set_title(artifact)

            fig.suptitle(f"Correlation matrices of {N_best} best classifiers on each artifact")

            save_path = (self.slash).join([self.dir, "Plots", self.experiment_name])

            img_path = f"{save_path}{self.slash}correlation_matrix.png"
            os.makedirs((self.slash).join(img_path.split(self.slash)[:-1]), exist_ok=True)
            fig.savefig(img_path)

            fig.show()

            return corr_matrix_dict

    def getMutualInformation(self, bestDict, artifact=None, withFolds=False):

        if artifact == None:
            print("Please specify artifact!")
        else:
            corr_matrix = self.getCorrelation(bestDict=bestDict, artifact='eyem', withFolds=withFolds)

            # Formula from here: https://lips.cs.princeton.edu/correlation-and-mutual-information/
            I = -1 / 2 * np.log(1 - np.round(corr_matrix, 4) ** 2)

            print("#" * 80)
            print(f"Mutual information matrix: class = {artifact}")
            print(I)

        return I

    def EnsemblePredictions(self, bestDict, select_models, select_aug_ratios, select_smote_ratios,
                            select_aug_techniques, artifact=None,
                            withFolds=True):

        #select_smote_ratios = np.array(select_smote_ratios)
        if artifact is None:
            raise AttributeError("Please specify artifact!")
        else:

            n_classifiers = len(select_models)
            preds = []
            print("-"*80)
            print(f"\nCreating ensemble predictions for {artifact}")

            for i in range(n_classifiers):
                model_pos = np.where(bestDict['models'][artifact] == select_models[i])
                tech_pos = np.where(bestDict['technique'][artifact] == select_aug_techniques[i])
                aug_pos = np.where(bestDict['augRatios'][artifact] == select_aug_ratios[i])
                smote_pos = np.where(bestDict['smote_ratios'][artifact] == select_smote_ratios[i])

                temp = np.intersect1d(model_pos, tech_pos)
                temp = np.intersect1d(temp, aug_pos)
                temp = np.intersect1d(temp, smote_pos)

                preds.append(bestDict['predictions'][artifact][temp])
                summary = (bestDict['models'][artifact][temp][0],
                           bestDict['technique'][artifact][temp][0],
                           bestDict['augRatios'][artifact][temp][0],
                           bestDict['smote_ratios'][artifact][temp][0] - 1)

                print(f"No. {temp[0] + 1}) {summary}")


            if withFolds == False:
                ensemble_preds = []

                preds = np.array([arr[0] for arr in preds])
                preds = np.transpose(preds)
                for arr in tqdm(preds):
                    ensemble_preds.append(Counter(arr.tolist()).most_common(1)[0][0])

                return np.array(ensemble_preds)

            else:
                ensemble_preds = defaultdict(dict)
                for i in self.folds:
                    ensemble_preds_fold = []

                    preds_fold = np.array([preds[model][0][i] for model in range(n_classifiers)])
                    preds_fold = np.transpose(preds_fold)
                    for arr in tqdm(preds_fold):
                        ensemble_preds_fold.append(Counter(arr.tolist()).most_common(1)[0][0])

                    ensemble_preds[i] = np.array(ensemble_preds_fold)

                return ensemble_preds

    def plotAugTechnique(self, bestDict, mean=True, max_Aug=False, smote_ratio=None, measure='Balanced acc',
                         artifacts=None, exclude_baseline=True):

        smote_ratio_old = smote_ratio
        if mean == max_Aug:
            raise AttributeError("Only one of either mean or max_Aug can be chosen!")

        if artifacts == None:
            artifacts = self.artifacts

        Aug_comparisonDict = defaultdict(dict)

        for a, artifact in enumerate(artifacts):
            techniques = np.unique(bestDict['technique'][artifact])
            techniques = np.array(['control', 'MixUp', 'GAN', 'color', 'white'])
            for i, technique in enumerate(techniques):
                if technique == "control":
                    smote_ratio_old = smote_ratio
                    smote_ratio = 1

                if exclude_baseline:
                    pos_tech1 = np.where(bestDict['technique'][artifact] == technique)[0]
                    pos_tech2 = np.where(bestDict['models'][artifact] != 'baseline_perm')[0]
                    pos_tech3 = np.where(bestDict['smote_ratios'][artifact] == smote_ratio)[0]

                    if smote_ratio != None:
                        pos_tech = np.intersect1d(pos_tech1, pos_tech2)
                        pos_tech = np.intersect1d(pos_tech, pos_tech3)
                    else:
                        pos_tech = np.intersect1d(pos_tech1, pos_tech2)
                else:
                    pos_tech = np.where(bestDict['technique'][artifact] == technique)
                    if smote_ratio != None:
                        pos_tech3 = np.where(bestDict['smote_ratios'][artifact] == smote_ratio)[0]
                        pos_tech = np.intersect1d(pos_tech3, pos_tech)

                scores_tech = bestDict['scores'][artifact][pos_tech]
                Aug_comparisonDict[technique] = scores_tech

                number = len(techniques)
                cmap = plt.get_cmap('coolwarm')
                colorlist = [cmap(i) for i in np.linspace(0, 1, number)]

                w = 0.15
                w = 0.75 / len(techniques)
                if len(techniques) == 1:
                    X_axis = np.arange(len(self.models))
                else:
                    X_axis = a - 0.3

                if mean:
                    height = np.mean(Aug_comparisonDict[technique])
                    yerr = np.std(Aug_comparisonDict[technique])
                elif max_Aug:
                    height = Aug_comparisonDict[technique][0]
                    yerr = bestDict['errors'][artifact][pos_tech][0]

                if smote_ratio == None:
                    plt.bar(x=X_axis + w * i,
                            height=height,
                            width=w,
                            color=colorlist[i],
                            label=f"{technique}")
                else:
                    plt.bar(x=X_axis + w * i,
                            height=height,
                            width=w,
                            color=colorlist[i],
                            label=f"{technique}")

                plt.errorbar(x=X_axis + w * i, y=height,
                             yerr=yerr,
                             fmt='.', color='k')

                smote_ratio = smote_ratio_old

            plt.xticks(np.arange(len(artifacts)), artifacts, rotation=0)
            plt.ylim(0, 1)

            if mean:
                plt.title(f"Mean across aug. methods, Baseline excluded: {exclude_baseline}")
            if max_Aug:
                plt.title(f"Max performance within aug. methods, Baseline excluded: {exclude_baseline}")

            plt.xlabel("Artifacts")
            plt.ylabel(measure)

            if a == 0:
                plt.legend(loc='center right', bbox_to_anchor=(1.33, 0.5))
                plt.subplots_adjust(right=0.775)

        if smote_ratio == None:
            img_path = f"{(self.slash).join([self.dir, 'Plots', self.experiment_name, measure])}{self.slash}augTechniquesComparison_SMOTEmixed_Mean={mean}.png"
        else:
            img_path = f"{(self.slash).join([self.dir, 'Plots', self.experiment_name, measure])}{self.slash}augTechniquesComparison_SMOTE{smote_ratio - 1}Mean={mean}.png"
        os.makedirs((self.slash).join(img_path.split(self.slash)[:-1]), exist_ok=True)
        plt.savefig(img_path)
        plt.show()

        for i, technique in enumerate(techniques):
            smote_ratio_old = smote_ratio
            if technique == "control":
                smote_ratio = 1

            scores = Aug_comparisonDict[technique]
            if smote_ratio == None:
                plt.plot(np.arange(len(scores)) / len(scores), scores, label=f"{technique} (SMOTE: mixed)",
                         color=colorlist[i])

            else:
                plt.plot(np.arange(len(scores)) / len(scores), scores, label=f"{technique} (SMOTE {smote_ratio - 1})",
                         color=colorlist[i])
            smote_ratio = smote_ratio_old

        plt.ylim(0, 1)
        plt.title(f"Sorted performance of aug. methods")
        plt.xlabel("Percentage of models (within aug. method) evaluated")
        plt.ylabel(measure)
        plt.legend()

        if smote_ratio == None:
            img_path = f"{(self.slash).join([self.dir, 'Plots', self.experiment_name, measure])}{self.slash}sortedScores_SMOTEmixed_Mean{mean}.png"
        else:
            img_path = f"{(self.slash).join([self.dir, 'Plots', self.experiment_name, measure])}{self.slash}sortedScores_SMOTE{smote_ratio - 1}Mean{mean}.png"
        os.makedirs((self.slash).join(img_path.split(self.slash)[:-1]), exist_ok=True)
        plt.savefig(img_path)
        plt.show()

    def plotImprovementExp(self, bestDict, smote_ratio, measure='balanced_acc'):
        self.augtechniques = ['control', 'MixUp', 'GAN', 'white', 'color']

        control_max = defaultdict(dict)
        control_err = defaultdict(dict)
        ax_control = 0

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize = (14, 8))
        axList = [ax_control, ax1, ax2, ax3, ax4]

        for ax, aug_technique in enumerate(self.augtechniques):

            bestForEachModel = {model: [] for model in self.models}
            errorForEachModel = {model: [] for model in self.models}

            for a, artifact in enumerate(self.artifacts):
                tech_pos = np.where(bestDict['technique'][artifact] == aug_technique)

                for model in self.models:
                    model_pos = np.where(bestDict['models'][artifact] == model)
                    temp_pos = np.intersect1d(tech_pos, model_pos)

                    if aug_technique == 'control':
                        control_max[(artifact, model)] = bestDict['scores'][artifact][temp_pos][0]
                        control_err[(artifact, model)] = bestDict['errors'][artifact][temp_pos][0]

                    else:
                        controlAUG_pos = np.where(bestDict['augRatios'][artifact] == 0)
                        conrtolSMOTE_pos = np.where(bestDict['smote_ratios'][artifact] == 1)
                        control_pos = np.intersect1d(controlAUG_pos, conrtolSMOTE_pos)
                        temp_pos = np.setdiff1d(temp_pos, control_pos)

                        smote_pos = np.where(bestDict['smote_ratios'][artifact] == smote_ratio)
                        temp_pos = np.intersect1d(temp_pos, smote_pos)

                        bestForEachModel[model].append(bestDict['scores'][artifact][temp_pos][0])
                        errorForEachModel[model].append(bestDict['errors'][artifact][temp_pos][0])

            if aug_technique == 'control':
                pass
            else:
                number = len(self.models)
                cmap = plt.get_cmap('coolwarm')
                colorlist = [cmap(i) for i in np.linspace(0, 1, number)]

                w = 0.15
                w = 0.75 / len(self.models)
                X_axis = np.arange(len(self.artifacts)) - 0.3

                i = 0

                for model in self.models:
                    control_maxList = np.array([control_max[(artifact, model)] for artifact in self.artifacts])
                    control_errList = np.array([control_err[(artifact, model)] for artifact in self.artifacts])

                    height = np.array(bestForEachModel[model]) - control_maxList
                    pooledSTD = np.sqrt((np.array(errorForEachModel[model]) ** 2 + control_errList ** 2) / 2)

                    axList[ax].bar(x=X_axis + w * i,
                            height=height,
                            width=w,
                            color=colorlist[i],
                            label=model)

                    axList[ax].errorbar(x=X_axis + w * i, y=height,
                                 yerr=pooledSTD,
                                 fmt='.', color='k')
                    i += 1

                    axList[ax].tick_params()
                    axList[ax].tick_params()

                    axList[ax].set_xticks(np.arange(len(self.artifacts)))
                    axList[ax].set_xticklabels(self.artifacts)
                    axList[ax].set_title(f"{aug_technique}, SMOTE: {smote_ratio - 1}")#, {measure}")
                    if measure == 'sensitivity':
                        axList[ax].set_ylim(-0.4, 0.4)
                    else:
                        axList[ax].set_ylim(-0.2, 0.4)

                if ax == 4:
                    #fig.suptitle(measure)
                    plt.legend(loc='center right', bbox_to_anchor=(1.36, 1.1))
                    plt.subplots_adjust(left=0.05, right=0.87, bottom=0.05)


                    img_path = f"{(self.slash).join([self.dir, 'Plots', self.experiment_name, measure])}{self.slash}_SMOTE{smote_ratio}_{measure}.png"
                    os.makedirs((self.slash).join(img_path.split(self.slash)[:-1]), exist_ok=True)
                    plt.savefig(img_path)

                    plt.show()

    def printNBestModels(self, bestDict, N_best=20, exclude_baseline=False, exclude_zero_aug=False):

        for artifact in self.artifacts:
            print("\n" + 100 * "-")
            print(f"Artifact: {artifact}\n")

            i = 0
            stop = 1
            while stop < N_best:
                tech = bestDict['technique'][artifact][i]
                if exclude_zero_aug == False:
                    if exclude_baseline:
                        if bestDict['models'][artifact][i] != 'baseline_perm':
                            if len(bestDict['models'][artifact][i]) > 2:
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
                        if len(bestDict['models'][artifact][i]) > 2:
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
                            if len(bestDict['models'][artifact][i]) > 2:  # tabs / indentation
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
                            print(
                                f"{stop})\t{bestDict['models'][artifact][i]} \t(SMOTE {bestDict['smote_ratios'][artifact][i] - 1}): "
                                f"{tech}  \t\t({bestDict['augRatios'][artifact][i]}) = {bestDict['scores'][artifact][i]}")
                        else:
                            print(
                                f"{stop})\t{bestDict['models'][artifact][i]} \t\t(SMOTE {bestDict['smote_ratios'][artifact][i] - 1}): "
                                f"{tech} \t\t({bestDict['augRatios'][artifact][i]}) = {bestDict['scores'][artifact][i]}")
                        i += 1
                        stop += 1

    def plotEnsembleMulitLabelCM(self, ensemble_preds, y_true):

        fig, axs = plt.subplots(2, 3)  # , sharex=True, sharey=True)

        cbar_ax = fig.add_axes([0.89, .3, .03, .4])
        plt.subplots_adjust(left=0.1, right=0.85)

        for i, artifact in enumerate(self.artifacts):
            j = 0

            if i > 2:
                j = 1

            conf_folds = [confusion_matrix(y_pred=ensemble_preds[artifact][fold], y_true=y_true[artifact][fold]) for fold in self.folds]
            sum_conf_matrix = np.sum(conf_folds, axis=0)
            sum_conf_matrix = sum_conf_matrix.astype('float') / sum_conf_matrix.sum(axis=1)[:, np.newaxis]
            conf_matrix = sns.heatmap(np.round(sum_conf_matrix, 2), annot=True, cmap=plt.cm.Blues,
                                      ax=axs[j, i % 3], cbar_ax=cbar_ax,
                                      vmin=0, vmax=1)
            conf_matrix.set_aspect('equal', 'box')

            axs[j, 0].tick_params()
            axs[j, 1].tick_params()

            if i % 3 == 0:
                conf_matrix.set_ylabel("Actual label")
            if j == 1:
                conf_matrix.set_xlabel("Predicted label")

            conf_matrix.set_title(f"{artifact}")

        fig.suptitle(f"Confusion matrices on each artifact, Ensemble classifier")

        save_path = (self.slash).join([self.dir, "Plots", self.experiment_name])

        img_path = f"{save_path}{self.slash}confusion_matrix.png"
        os.makedirs((self.slash).join(img_path.split(self.slash)[:-1]), exist_ok=True)
        fig.savefig(img_path)

        fig.show()


if __name__ == '__main__':
    dir = r"C:\Users\Albert Kjøller\Documents\GitHub\EEG_epilepsia"  # dir = "/Users/philliphoejbjerg/Documents/GitHub/EEG_epilepsia"  # dir = r"/Users/Jacobsen/Documents/GitHub/EEG_epilepsia" + "/"
    y_true_path = r"C:\Users\Albert Kjøller\Documents\GitHub\EEG_epilepsia\results\y_true"
    y_true_file = r"\y_true_5fold_randomstate_0.npy"

    # dir = r"/Users/Jacobsen/Documents/GitHub/EEG_epilepsia"  # dir = "/Users/philliphoejbjerg/Documents/GitHub/EEG_epilepsia"  # dir = r"/Users/Jacobsen/Documents/GitHub/EEG_epilepsia" + "/"
    # y_true_path = r"C:\Users\Albert Kjøller\Documents\GitHub\EEG_epilepsia\results\y_true\y_true_5fold_randomstate_0.npy"

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
    ensembleExp = getResultsEnsemble(dir, experiments=experiments, experiment_name=experiment_name, merged_file=False,
                                     windowsOS=windowsOS)
    measure = 'balanced_acc' #'balanced_acc'
    N_best = 20
    withFolds = True

    loadedBestDictName = slash + "orderedPredictions_100balanced_acc_foldsTrue.npy"  # Either None or file-name
    bestDictPicklepath = (slash).join([dir, "results", experiment_name])

    if loadedBestDictName == None:
        bestDict = ensembleExp.findNBestModels(dir=dir, experiment_name=experiment_name,
                                               experiments=experiments, y_true_path=y_true_path+y_true_file,
                                               N_best=N_best, withPreds=True, withFolds=withFolds,
                                               measure=measure, windowsOS=windowsOS)
    else:
        bestDict = LoadNumpyPickles(pickle_path=bestDictPicklepath, file_name=loadedBestDictName, windowsOS=windowsOS)[
            ()]

    ensembleExp.printNBestModels(bestDict=bestDict, N_best=N_best, exclude_baseline=True)
    ensembleExp.plotAugTechnique(bestDict=bestDict, mean=False, max_Aug=True, measure=measure, exclude_baseline=True)

    # y_pred_dict = ensembleExp.getPredictionsEnsemble(best_pred_dict=bestDict, experiments=experiments, N_best=N_best, artifacts=None)
    # y_pred_dict = ensembleExp.compressDict(y_pred_dict)

    N_best = 20
    corr_matrix = ensembleExp.getCorrelation(bestDict=bestDict, N_best=N_best,
                                                 withFolds=withFolds)  # , latex=True)

    n_classifiers = 5

    leastCorrelatedModelIdxs = {artifact: np.argsort(np.mean(corr_matrix[artifact].to_numpy(), axis=0))[:n_classifiers] for artifact in ensembleExp.artifacts}
    leastCorrelatedModelsDict = defaultdict(dict)

    # Finding least correlated models based on their mean with the remaining ones
    for artifact in ensembleExp.artifacts:
        Idxs = leastCorrelatedModelIdxs[artifact]
        leastCorrelatedModelsDict[artifact] = {'models': bestDict['models'][artifact][Idxs].tolist(),
                                    'techniques': bestDict['technique'][artifact][Idxs].tolist(),
                                    'aug_ratios': bestDict['augRatios'][artifact][Idxs].tolist(),
                                    'smote_ratios': bestDict['smote_ratios'][artifact][Idxs].tolist()}

    # getting y_true predicitons
    results_y_true = LoadNumpyPickles(pickle_path=y_true_path, file_name=r"\y_true_5fold_randomstate_0.npy",
                                      windowsOS=windowsOS)
    results_y_true = results_y_true[()]
    y_true_dict = {}
    for artifact in tqdm(ensembleExp.artifacts):
        y_true_art = []
        for i in ensembleExp.folds:
            y_true_art.append(results_y_true[i][artifact]['y_true'])
        y_true_dict[artifact] = y_true_art

    fold_art_table = np.zeros((5, 6))

    for fold in ensembleExp.folds:
        for i, artifact in enumerate(ensembleExp.artifacts):
            present = y_true_dict[artifact][fold].sum()
            all = len(y_true_dict[artifact][fold])

            fold_art_table[fold, i] = present / all

    df = pd.DataFrame(np.round(fold_art_table, 4) * 100, columns=ensembleExp.artifacts, index=np.arange(5) + 1)
    print(df.to_latex())

    # The input should be a list of models, a list of aug_ratios for each model and a list of smote_ratios for each
    # model. For future experiments it should take a list with augmentation_techniques in as well.
    ensemble_preds = defaultdict(dict)
    bacc_dict = defaultdict(dict)
    sens_dict = defaultdict(dict)
    bestModelsDict = leastCorrelatedModelsDict
    for artifact in ensembleExp.artifacts:
        ensemble_preds_artifact = ensembleExp.EnsemblePredictions(bestDict=bestDict,
                                                                  artifact=artifact,
                                                                  select_models=bestModelsDict[artifact]['models'],
                                                                  select_aug_techniques=bestModelsDict[artifact]['techniques'],
                                                                  select_aug_ratios=bestModelsDict[artifact]['aug_ratios'],
                                                                  select_smote_ratios=bestModelsDict[artifact]['smote_ratios'],
                                                                  withFolds=withFolds)
        ensemble_preds[artifact] = ensemble_preds_artifact

        bacc = []
        sens = []
        for fold in ensembleExp.folds:
            actual = y_true_dict[artifact][fold]
            predictions = ensemble_preds_artifact[fold]
            bacc.append(balanced_accuracy_score(y_true=actual, y_pred=predictions))

            conf_matrix = confusion_matrix(y_true=actual, y_pred=predictions, labels=[0, 1])

            TP = conf_matrix[1][1]
            TN = conf_matrix[0][0]
            FP = conf_matrix[0][1]
            FN = conf_matrix[1][0]

            if TP == 0 and FN == 0:
                print("No TP or FN found.")
                FN = 1  # Random number to account for division by zero
            sensitivity = (TP / float(TP + FN))
            sens.append(sensitivity)


        bacc_dict[artifact] = (np.mean(bacc), np.std(bacc))
        sens_dict[artifact] = (np.mean(sens), np.std(sens))
        print(f"\nB.acc.{artifact}: {bacc_dict[artifact]}")
        print(f"\nSens. {artifact}: {sens_dict[artifact]}")

    ensembleExp.plotEnsembleMulitLabelCM(ensemble_preds=ensemble_preds, y_true=y_true_dict)

    exps = ['control', 'ensemble']

    control_sens_manual = {'scores': {'eyem': 0.69,
                                      'chew': 0.81,
                                      'shiv': 0.49,
                                      'elpp': 0.49,
                                      'musc': 0.55,
                                      'null': 0.76},

                           'errors': {'eyem': 0.04,
                                      'chew': 0.06,
                                      'shiv': 0.34,
                                      'elpp': 0.11,
                                      'musc': 0.10,
                                      'null': 0.03}}

    cmap = plt.get_cmap('Blues')
    colorlist = [cmap(i) for i in np.linspace(0.3, 0.8, len(exps))]
    #colorlist = ["lightsteelblue", "darkcyan"]
    fig, ((ax1,ax2)) = plt.subplots(1, 2, figsize=(10,5))
    axList = [ax1, ax2]
    for m, measure in enumerate(['balanced_acc', 'sensitivity']):
        for indv_art, artifact in enumerate(ensembleExp.artifacts):
            for i, exp in enumerate(exps):

                if exp == 'control':
                    if measure == 'balanced_acc':
                        pos = np.where(bestDict['technique'][artifact] == exp)[0][0]
                        score = bestDict['scores'][artifact][pos]
                        error = bestDict['errors'][artifact][pos]
                    elif measure == 'sensitivity':
                        score = control_sens_manual['scores'][artifact]
                        error = control_sens_manual['errors'][artifact]

                if exp == 'ensemble':
                    if measure == 'balanced_acc':
                        score, error = bacc_dict[artifact]
                    elif measure == 'sensitivity':
                        score, error = sens_dict[artifact]

                if indv_art == 0:
                    label = f"{exp}"
                else:
                    label = ""

                # TODO: FIX THIS AXIS-thing for a nice bar plot!!
                X_axis = np.arange(len(ensembleExp.artifacts)) - 0.35/2
                axList[m].bar(x=X_axis[indv_art] + 0.35 * i,
                        height=score,
                        width=0.35,
                        color=colorlist[i],
                        label=label)

                axList[m].errorbar(x=X_axis[indv_art] + 0.35 * i,
                             y=score,
                             yerr=error,
                             fmt='.', color='k')

        artifacts = ensembleExp.artifacts

        axList[m].set_xticks(np.arange(len(artifacts)))
        axList[m].set_xticklabels(artifacts)
        axList[m].set_ylim(0, 1)
        #plt.title()

        axList[m].set_xlabel("Artifacts")
        axList[m].set_ylabel(measure)
    plt.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))
    plt.subplots_adjust(right=0.85)
    plt.show()



    print("")
