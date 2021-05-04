from results import *


if __name__ == '__main__':
    dir = r"C:\Users\Albert Kjøller\Documents\GitHub\EEG_epilepsia"  # dir = "/Users/philliphoejbjerg/Documents/GitHub/EEG_epilepsia"  # dir = r"/Users/Jacobsen/Documents/GitHub/EEG_epilepsia" + "/"

    experiment = "smote_f2"  # "DataAug_white_noiseAdd_LR"
    experiment_name = '_smote_f2_GNB'  # end by either _Noise (or _color_Noise), _GAN or _MixUp if Augmentation
    experiment_object = getResults(dir, experiment, experiment_name, merged_file=False, windowsOS=True)

    # Example of manipulating a single model by adding results from a single artifact, i.e. GNB Null added to GNB.
    GNB = "GNB_for_merge_smote_f2"
    experiment_object.addSingleArtifact(dir_name=GNB, new_file_name="results_smote_f2_GNB",
                                        main_file="results_smote_f2_GNB.npy",
                                        sec_file="results_smote_f2_GNBNull.npy",
                                        experiment=experiment)