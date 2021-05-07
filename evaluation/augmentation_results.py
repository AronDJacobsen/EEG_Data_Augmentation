from results import *


if __name__ == '__main__':
    dir = r"C:\Users\Albert Kjøller\Documents\GitHub\EEG_epilepsia"  # dir = "/Users/philliphoejbjerg/Documents/GitHub/EEG_epilepsia"  # dir = r"/Users/Jacobsen/Documents/GitHub/EEG_epilepsia" + "/"

    # Example of merging fully created files from different models.
    experiment = "augmentation_colorNoise"  # directory containing the files we will look at
    experiment_name = '_augmentation_colorNoise_merged_allModels'
    fullSMOTE = getResults(dir, experiment, experiment_name, merged_file=True, windowsOS=True)
    fullSMOTE.mergeResultFiles(file_name=experiment_name)

    # To work with the merged file we have to change the pickle path to the "merged" folder.
    fullSMOTE.changePicklePath()

    # Initialize global parameters?
    aug_technique = "colored noise"  # "GAN" # "MixUp" #Noise Addition --> mainly for naming of the plots
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
