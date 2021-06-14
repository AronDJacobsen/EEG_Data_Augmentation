from results import *

if __name__ == '__main__':
    dir = r"C:\Users\Albert Kjøller\Documents\GitHub\EEG_epilepsia"  # dir = "/Users/philliphoejbjerg/Documents/GitHub/EEG_epilepsia"  # dir = r"/Users/Jacobsen/Documents/GitHub/EEG_epilepsia" + "/"

    # Example of merging fully created files from different models.
    experiment = "SMOTE"  # directory containing the files we will look at
    experiment_name = '_smote_merged'
    fullSMOTE = getResults(dir, experiment, experiment_name, merged_file=True, windowsOS=True)
    fullSMOTE.mergeResultFiles(file_name=experiment_name)

    # To work with the merged file we have to change the pickle path to the "merged" folder.
    fullSMOTE.changePicklePath()

    # Initialize global parameters?
    aug_technique = None  # "GAN" # "MixUp" #Noise Addition --> mainly for naming of the plots
    LaTeX = True
    save_img = True
    y_true_path = r"C:\Users\Albert Kjøller\Documents\GitHub\EEG_epilepsia\results\y_true\y_true_5fold_randomstate_0.npy"

    # Next we wish to examine F2!
    fullSMOTE.printResults(measure="weighted_F2",
                           experiment_name=experiment_name,
                           y_true_path=y_true_path,
                           smote_ratios=[1],
                           aug_ratios=[0],
                           printSTDTable=True,
                           LaTeX=True)

    #fullSMOTE.plotResultsPlainExp(experiment_name=experiment_name,
    #                              across_SMOTE=True, y_true_path=y_true_path,
    #                              save_img=True)


    # No augmentation or smote at all!
    """fullSMOTE.plotResults(measure="balanced_acc",
                          experiment_name=experiment_name,
                          aug_technique=aug_technique,
                          y_true_path=y_true_path,
                          smote_ratios=fullSMOTE.smote_ratios,
                          aug_ratios=[0],
                          across_SMOTE=True,
                          save_img=True)#save_img))"""
    fullSMOTE.plotResults(measure="weighted_F2",
                          experiment_name=experiment_name,
                          aug_technique=aug_technique,
                          y_true_path=y_true_path,
                          smote_ratios=[1],#fullSMOTE.smote_ratios,
                          aug_ratios=[0],
                          across_SMOTE=True,
                          save_img=True)#save_img)
    fullSMOTE.plotResults(measure="accuracy",
                          experiment_name=experiment_name,
                          aug_technique=aug_technique,
                          y_true_path=y_true_path,
                          smote_ratios=[1],  # fullSMOTE.smote_ratios,
                          aug_ratios=[0],
                          across_SMOTE=True,
                          save_img=True)  # save_img)
    fullSMOTE.plotResults(measure="sensitivity",
                          experiment_name=experiment_name,
                          aug_technique=aug_technique,
                          y_true_path=y_true_path,
                          smote_ratios=[1],  # fullSMOTE.smote_ratios,
                          aug_ratios=[0],
                          across_SMOTE=True,
                          save_img=True)  # save_img)

    """fullSMOTE.plotResults(measure="balanced_acc",
                          experiment_name=experiment_name,
                          aug_technique=aug_technique,
                          smote_ratios=fullSMOTE.smote_ratios,
                          aug_ratios=[0],
                          across_SMOTE=True,
                          save_img=True)  # save_img)"""

    fullSMOTE.plotResults(measure="weighted_F2",
                          experiment_name=experiment_name,
                          aug_technique=aug_technique,
                          y_true_path=y_true_path,
                          smote_ratios=fullSMOTE.smote_ratios,
                          aug_ratios=[0],
                          across_SMOTE=True,
                          save_img=True)  # save_img)

    fullSMOTE.plotResults(measure="accuracy",
                          experiment_name=experiment_name,
                          aug_technique=aug_technique,
                          y_true_path=y_true_path,
                          smote_ratios=fullSMOTE.smote_ratios,
                          aug_ratios=[0],
                          across_SMOTE=True,
                          save_img=save_img)

    fullSMOTE.plotResults(measure='sensitivity',
                          experiment_name=experiment_name,
                          aug_technique=aug_technique,
                          y_true_path=y_true_path,
                          smote_ratios=fullSMOTE.smote_ratios,
                          aug_ratios=[0],
                          across_SMOTE=True,
                          save_img=save_img)

    print("Break")

