from results import *


def findNBestModels(dir, experiment_name, experiments, y_true_path, N_best=None, measure='balanced_acc',
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
            for aug_ratio in results_object.aug_ratios:
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
    experiments = {"augmentation_colorNoise": (1, 'color'), "augmentation_whiteNoise": (1, 'white'),
                   "augmentation_MixUp": (1, 'MixUp'), "augmentation_GAN": (1, 'GAN'),
                   "colorNoiseimprovement": (2, 'color'), "whiteNoiseimprovement": (2, 'white'),
                   "MixUpimprovement": (2, 'MixUp'), "GANimprovement": (2, 'GAN')}

    experiment_name = "_ensemble_experiment"
    measure = 'sensitivity'
    N_best = None

    bestDict = findNBestModels(dir=dir, experiment_name=experiment_name,
                               experiments=experiments, y_true_path=y_true_path,
                               N_best=N_best,
                               measure=measure, windowsOS=windowsOS)
    
    y_pred_dict = fullSMOTE.getPredictions(models=models,
                                           aug_ratios=[0],
                                           smote_ratios=[smote_ratio],
                                           artifacts=artifacts)
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
