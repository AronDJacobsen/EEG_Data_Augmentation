
from dataLoader import createSubjectDict, PicklePrepData
import os


### Create pickles from preprocessed data based on the paths below. Unmuted when pickles exist. Takes quite
### a while to run

pickle_folder = r"\example"
pickle_path = r"C:\Users\Albert Kjøller\Documents\GitHub\EEG_epilepsia" + pickle_folder
os.makedirs(pickle_path, exist_ok=True)

prep_dir = r"C:\Users\Albert Kjøller\Documents\GitHub\TUAR_full_data\clean_data" + "\\"
windowsOS = True

subject_dict = createSubjectDict(prep_directory=prep_dir, windowsOS=windowsOS)
PicklePrepData(subjects_dict=subject_dict, prep_directory=prep_dir, save_path=pickle_path, windowsOS=windowsOS)