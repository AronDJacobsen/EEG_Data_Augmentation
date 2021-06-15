from results import *

if __name__ == '__main__':
    dir = r"C:\Users\Albert Kjøller\Documents\GitHub\EEG_epilepsia"  # dir = "/Users/philliphoejbjerg/Documents/GitHub/EEG_epilepsia"  # dir = r"/Users/Jacobsen/Documents/GitHub/EEG_epilepsia" + "/"

    y_true_path = r"C:\Users\Albert Kjøller\Documents\GitHub\EEG_epilepsia\results\y_true\y_true_5fold_randomstate_0.npy"

    """ # Example of merging fully created files from different models.
    experiment = "augmentation_colorNoise"  # directory containing the files we will look at
    experiment_name = '_augmentation_colorNoise_merged_allModels'
    augExp = getResults(dir, experiment, experiment_name, merged_file=True, windowsOS=True)
    augExp.mergeResultFiles(file_name=experiment_name)

    # To work with the merged file we have to change the pickle path to the "merged" folder.
    augExp.changePicklePath()

    # Initialize global parameters?
    aug_technique = "colored noise"  # "GAN" # "MixUp" #Noise Addition --> mainly for naming of the plots
    LaTeX = False
    save_img = True """

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
                                         save_img=True)

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
        for measure in ["weighted_F2", "sensitivity", "balanced_acc"]:
            augExp.plotAllModelsBestAug(measure=measure,
                                        experiment_name=experiment_name,
                                        y_true_path=y_true_path,
                                        aug_technique=aug_technique,
                                        smote_ratios=[smote + 1],
                                        aug_ratios=augExp.aug_ratios,
                                        save_img=save_img)

        augExp.plotResultsImprovementExp(experiment_name=experiment_name,
                                         smote_ratio=smote_ratio,
                                         y_true_path=y_true_path,
                                         aug_technique=aug_technique,
                                         save_img=True)



    """# Next we wish to examine F2!
    augExp.printResults(measure="weighted_F2",
                        experiment_name=experiment_name,
                        y_true_path=y_true_path,
                        smote_ratios=[1],
                        aug_ratios=[0.5],
                        printSTDTable=True,
                        across_SMOTE=False,
                        LaTeX=False)"""

