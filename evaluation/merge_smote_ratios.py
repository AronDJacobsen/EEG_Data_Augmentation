
from results import *

if __name__ == '__main__':
    dir = r"C:\Users\Albert Kjøller\Documents\GitHub\EEG_epilepsia"  # dir = "/Users/philliphoejbjerg/Documents/GitHub/EEG_epilepsia"  # dir = r"/Users/Jacobsen/Documents/GitHub/EEG_epilepsia" + "/"


    # Example of merging result-files created with same models but different SMOTE-ratios
    experiment = 'AdaBoost_for_merge_smote_f2'
    experiment_name = '_smote_f2_AdaBoost'  # end of the name of the new file to be created or the file(s) to be loaded
    AdaBoost_smote = getResults(dir, experiment, experiment_name, merged_file=True, windowsOS=True)
    AdaBoost_smote.mergeResultFiles(file_name=experiment_name)
