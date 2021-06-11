from results import *


if __name__ == '__main__':
    dir = r"C:\Users\Albert Kjøller\Documents\GitHub\EEG_epilepsia"  # dir = "/Users/philliphoejbjerg/Documents/GitHub/EEG_epilepsia"  # dir = r"/Users/Jacobsen/Documents/GitHub/EEG_epilepsia" + "/"

    """ # Example of merging fully created files from different models.
    experiment = "augmentation_colorNoise"  # directory containing the files we will look at
    experiment_name = '_augmentation_colorNoise_merged_allModels'
    fullSMOTE = getResults(dir, experiment, experiment_name, merged_file=True, windowsOS=True)
    fullSMOTE.mergeResultFiles(file_name=experiment_name)

    # To work with the merged file we have to change the pickle path to the "merged" folder.
    fullSMOTE.changePicklePath()

    # Initialize global parameters?
    aug_technique = "colored noise"  # "GAN" # "MixUp" #Noise Addition --> mainly for naming of the plots
    LaTeX = False
    save_img = True """

    experiments = ["augmentation_colorNoise", "augmentation_whiteNoise", "augmentation_GAN", "augmentation_MixUp"]
    aug_techniques = ["colored noise", "white noise", "GAN", "MixUp"]
    LaTeX = False
    save_img = True
    smote = 0
    y_true_path = r"C:\Users\Albert Kjøller\Documents\GitHub\EEG_epilepsia\results\y_true\y_true_5fold_randomstate_0.npy"


    for i, experiment in enumerate(experiments):
        experiment_name = "_" + experiment + "_merged_allModels"
        fullSMOTE = getResults(dir, experiment, experiment_name, merged_file=True, windowsOS=True)
        fullSMOTE.mergeResultFiles(file_name=experiment_name)

        fullSMOTE.changePicklePath()

        aug_technique = aug_techniques[i]

        fullSMOTE.plotResults(measure="weighted_F2",
                              experiment_name=experiment_name,
                              y_true_path=y_true_path,
                              aug_technique=aug_technique,
                              smote_ratios=[smote+1],
                              aug_ratios=fullSMOTE.aug_ratios,
                              across_SMOTE=False,
                              save_img=save_img)

        fullSMOTE.plotResults(measure="sensitivity",
                              experiment_name=experiment_name,
                              y_true_path=y_true_path,
                              aug_technique=aug_technique,
                              smote_ratios=[smote+1],
                              aug_ratios=fullSMOTE.aug_ratios,
                              across_SMOTE=False,
                              save_img=save_img)


    experiments = ["colorNoiseimprovement", "whiteNoiseimprovement", "GANimprovement", "MixUpimprovement"]
    aug_techniques = ["colored noise", "white noise", "GAN", "MixUp"]
    LaTeX = False
    save_img = True
    smote = 1

    for i, experiment in enumerate(experiments):
        experiment_name = "_" + experiment + "_merged_allModels"
        fullSMOTE = getResults(dir, experiment, experiment_name, merged_file=True, windowsOS=True)
        fullSMOTE.mergeResultFiles(file_name=experiment_name)

        fullSMOTE.changePicklePath()

        aug_technique = aug_techniques[i]

        fullSMOTE.plotResults(measure="weighted_F2",
                              experiment_name=experiment_name,
                              y_true_path=y_true_path,
                              aug_technique=aug_technique,
                              smote_ratios=[smote + 1],
                              aug_ratios=fullSMOTE.aug_ratios,
                              across_SMOTE=False,
                              save_img=save_img)

        fullSMOTE.plotResults(measure="sensitivity",
                              experiment_name=experiment_name,
                              y_true_path=y_true_path,
                              aug_technique=aug_technique,
                              smote_ratios=[smote + 1],
                              aug_ratios=fullSMOTE.aug_ratios,
                              across_SMOTE=False,
                              save_img=save_img)

    # Next we wish to examine F2!
    fullSMOTE.printResults(measure="weighted_F2",
                           experiment_name=experiment_name,
                           y_true_path=y_true_path,
                           smote_ratios=[1],
                           aug_ratios=[0.5],
                           printSTDTable=True,
                           across_SMOTE=False,
                           LaTeX=False)


    # Example of merging fully created files from different models.
    experiment = "augmentation_whiteNoise"  # directory containing the files we will look at
    experiment_name = '_augmentation_whiteNoise_merged_allModels'
    fullSMOTE = getResults(dir, experiment, experiment_name, merged_file=True, windowsOS=True)
    fullSMOTE.mergeResultFiles(file_name=experiment_name)

    # To work with the merged file we have to change the pickle path to the "merged" folder.
    fullSMOTE.changePicklePath()

    # Initialize global parameters?
    aug_technique = "white noise"  # "GAN" # "MixUp" #Noise Addition --> mainly for naming of the plots
    LaTeX = False
    save_img = True

    # Next we wish to examine F2!
    fullSMOTE.printResults(measure="weighted_F2",
                           experiment_name=experiment_name,
                           smote_ratios=[1],
                           aug_ratios=[0.5],
                           printSTDTable=True,
                           across_SMOTE=False,
                           LaTeX=False)

    fullSMOTE.plotResults(measure="weighted_F2",
                          experiment_name=experiment_name,
                          aug_technique=aug_technique,
                          smote_ratios=[1],
                          aug_ratios=fullSMOTE.aug_ratios,
                          across_SMOTE=False,
                          save_img=save_img)





    # TODO: Don't know how we will work with the augmented files

    # Example of merging result-files created with same models but different Augmentation-ratios
    # TODO: Not tried yet as we have not conducted the experiment
    # experiment = "LR_for_merge_MixUp"
    # experiment_name = '_MixUp_LR'
    # LR_MixUp = getResults(dir, experiment, experiment_name, merged_file=True, windowsOS=True)
    # LR_MixUp.mergeResultFiles(file_name=experiment_name)

    # across_SMOTE has to be False when doing it with augmentation
    """
    # For augmentation
    fullSMOTE.plotResults(measure="sensitivity",
                          experiment_name=experiment_name,
                          aug_technique=aug_technique,
                          smote_ratios=fullSMOTE.smote_ratios,
                          aug_ratios=[0],
                          across_SMOTE=False,
                          save_img=save_img)
    """
