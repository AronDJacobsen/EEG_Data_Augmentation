from results import *


if __name__ == '__main__':



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

    raise NotImplementedError("Run augmentation experiments!")