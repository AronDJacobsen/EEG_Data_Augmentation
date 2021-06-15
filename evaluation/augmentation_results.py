from results import *
from ensemble_results import *

if __name__ == '__main__':
    dir = r"C:\Users\Albert Kjøller\Documents\GitHub\EEG_epilepsia"  # dir = "/Users/philliphoejbjerg/Documents/GitHub/EEG_epilepsia"  # dir = r"/Users/Jacobsen/Documents/GitHub/EEG_epilepsia" + "/"
    windowsOS = True

    y_true_path = r"C:\Users\Albert Kjøller\Documents\GitHub\EEG_epilepsia\results\y_true\y_true_5fold_randomstate_0.npy"

    experiment_name = "_ensemble_experiment"
    experiments = {"SMOTE": (1, 'control'),
                   "augmentation_colorNoise": (1, 'color'), "augmentation_whiteNoise": (1, 'white'),
                   "augmentation_MixUp": (1, 'MixUp'), "augmentation_GAN": (1, 'GAN'),
                   "colorNoiseimprovement": (2, 'color'), "whiteNoiseimprovement": (2, 'white'),
                   "MixUpimprovement": (2, 'MixUp'), "GANimprovement": (2, 'GAN')}
    ensembleExp = getResultsEnsemble(dir, experiments=experiments, experiment_name=experiment_name, merged_file=False,
                                     windowsOS=windowsOS)
    files = ["orderedPredictions_AllBestsensitivity.npy", "orderedPredictions_AllBestbalanced_acc.npy"]
    for i, measure in enumerate(['sensitivity', 'balanced_acc']):
        N_best = None
        slash = ensembleExp.slash

        loadedBestDictName = slash + files[i]
        bestDictPicklepath = (slash).join([dir, "results", experiment_name])

        bestDict = LoadNumpyPickles(pickle_path=bestDictPicklepath, file_name=loadedBestDictName, windowsOS=windowsOS)[()]

        ensembleExp.plotAugTechnique(bestDict=bestDict, measure=measure, smote_ratio=1, mean=True, max_Aug=False, exclude_baseline=True)
        ensembleExp.plotAugTechnique(bestDict=bestDict, measure=measure, smote_ratio=2, mean=True, max_Aug=False, exclude_baseline=True)
        ensembleExp.plotAugTechnique(bestDict=bestDict, measure=measure, smote_ratio=None, mean=True, max_Aug=False, exclude_baseline=True)

        ensembleExp.plotAugTechnique(bestDict=bestDict, measure=measure, smote_ratio=1, mean=False, max_Aug=True, exclude_baseline=True)
        ensembleExp.plotAugTechnique(bestDict=bestDict, measure=measure, smote_ratio=2, mean=False, max_Aug=True, exclude_baseline=True)
        ensembleExp.plotAugTechnique(bestDict=bestDict, measure=measure, smote_ratio=None, mean=True, max_Aug=False, exclude_baseline=True)

    experiments = ["augmentation_colorNoise", "augmentation_whiteNoise", "augmentation_GAN", "augmentation_MixUp"]
    aug_techniques = ["colored noise", "white noise", "GAN", "MixUp"]
    LaTeX = False
    save_img = True
    smote = 0

    for i, experiment in enumerate(experiments):
        experiment_name = "_" + experiment + "_merged_allModels"
        augExp = getResults(dir, experiment, experiment_name, merged_file=True, windowsOS=True)
        augExp.mergeResultFiles(file_name=experiment_name)

        augExp.changePicklePath()

        aug_technique = aug_techniques[i]
        """for measure in ["weighted_F2", "sensitivity", "balanced_acc"]:
            augExp.plotAllModelsBestAug(measure=measure,
                                        experiment_name=experiment_name,
                                        y_true_path=y_true_path,
                                        aug_technique=aug_technique,
                                        smote_ratios=[smote + 1],
                                        aug_ratios=augExp.aug_ratios,
                                        save_img=save_img)"""

        augExp.plotResultsImprovementExp(experiment_name=experiment_name,
                                         smote_ratio=smote + 1,
                                         y_true_path=y_true_path,
                                         aug_technique=aug_technique,
                                         sens_control=[0.72, 0.81, 0.7, 0.69, 0.7, 0.76],
                                         sens_std_control=[0.04, 0.06, 0.04, 0.04, 0.04, 0.03],
                                         bacc_control=[0.76, 0.86, 0.69, 0.63, 0.7, 0.69],
                                         bacc_std_control=[0.02, 0.02, 0.16, 0.04, 0.05, 0.02],
                                         save_img=True)

    experiments = ["colorNoiseimprovement", "whiteNoiseimprovement", "GANimprovement", "MixUpimprovement"]
    aug_techniques = ["colored noise", "white noise", "GAN", "MixUp"]
    LaTeX = False
    save_img = True
    smote = 1

    for i, experiment in enumerate(experiments):
        experiment_name = "_" + experiment + "_merged_allModels"
        augExp = getResults(dir, experiment, experiment_name, merged_file=True, windowsOS=True)
        augExp.mergeResultFiles(file_name=experiment_name)

        augExp.changePicklePath()

        aug_technique = aug_techniques[i]
        """for measure in ["weighted_F2", "sensitivity", "balanced_acc"]:
            augExp.plotAllModelsBestAug(measure=measure,
                                        experiment_name=experiment_name,
                                        y_true_path=y_true_path,
                                        aug_technique=aug_technique,
                                        smote_ratios=[smote + 1],
                                        aug_ratios=augExp.aug_ratios,
                                        save_img=save_img)"""

        augExp.plotResultsImprovementExp(experiment_name=experiment_name,
                                         smote_ratio=smote + 1,
                                         y_true_path=y_true_path,
                                         aug_technique=aug_technique,
                                         sens_control=[0.72, 0.81, 0.7, 0.69, 0.7, 0.76],
                                         sens_std_control=[0.04, 0.06, 0.04, 0.04, 0.04, 0.03],
                                         bacc_control=[0.76, 0.86, 0.69, 0.63, 0.7, 0.69],
                                         bacc_std_control=[0.02, 0.02, 0.16, 0.04, 0.05, 0.02],
                                         save_img=True)



    print("")


    """# Next we wish to examine F2!
    augExp.printResults(measure="weighted_F2",
                        experiment_name=experiment_name,
                        y_true_path=y_true_path,
                        smote_ratios=[1],
                        aug_ratios=[0.5],
                        printSTDTable=True,
                        across_SMOTE=False,
                        LaTeX=False)"""

