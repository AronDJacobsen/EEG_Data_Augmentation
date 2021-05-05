

from results import *



if __name__ == '__main__':
    dir = r"C:\Users\Albert Kjøller\Documents\GitHub\EEG_epilepsia"  # dir = "/Users/philliphoejbjerg/Documents/GitHub/EEG_epilepsia"  # dir = r"/Users/Jacobsen/Documents/GitHub/EEG_epilepsia" + "/"

    # Example of merging fully created files from different models.
    experiment = "smote_f2"  # directory containing the files we will look at
    experiment_name = '_smote_f2_withoutKNN'
    fullSMOTE = getResults(dir, experiment, experiment_name, merged_file=True, windowsOS=True)
    fullSMOTE.mergeResultFiles(file_name=experiment_name)

    # To work with the merged file we have to change the pickle path to the "merged" folder.
    fullSMOTE.changePicklePath()
    fullSMOTE.merged_file = True

    # Creates a dictionary for the predictions.
    y_pred_dict = fullSMOTE.getPredictions()  # models = ['LDA']
    corr_matrix = fullSMOTE.getCorrelation(artifact='eyem')
    MI = fullSMOTE.getMutualInformation(artifact='eyem')
    print("Mutual Information:\n" + str(MI))

    y_pred_dict_sub = fullSMOTE.getPredictions(#models=['LDA', 'GNB', 'MLP', 'LR', 'SGD'],
                                               aug_ratios=[0],
                                               withFolds=False)

    # Choose smote and aug-ratio
    y_pred_dict_sub = fullSMOTE.compressDict(y_pred_dict_sub, smote_ratio=1, aug_ratio=0)
    #fullSMOTE.printScores(pred_dict=y_pred_dict_sub, model='LDA', y_true_filename="y_true_randomstate_0")


    # The input should be a list of models, a list of aug_ratios for each model and a list of smote_ratios for each
    # model. For future experiments it should take a list with augmentation_techniques in as well.
    ensemble_pred_dict = fullSMOTE.EnsemblePredictions(['LDA', 'GNB', 'MLP', 'LR', 'SGD'], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1], withFolds=False)

    #TODO: Implement function to calculate standard error of the ensemble method!
    fullSMOTE.printScores(pred_dict=ensemble_pred_dict, y_true_filename = "y_true_randomstate_0", ensemble=True)
