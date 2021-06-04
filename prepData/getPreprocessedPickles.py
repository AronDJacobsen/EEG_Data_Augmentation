
from dataLoader import *
import os


### Create pickles from already preprocessed data based on the paths below. Unmuted when pickles exist. Takes quite
### a while to run

pickle_folder = r"\true_pickles"#r"\whiteNoise_pickles"
pickle_path = r"C:\Users\Albert Kjøller\Documents\GitHub\EEG_epilepsia" + pickle_folder
os.makedirs(pickle_path, exist_ok=True)

prep_dir = r"C:\Users\Albert Kjøller\Documents\GitHub\TUAR_full_data\whitenoise_covarOne_true" + "\\"
windowsOS = True

createNewPickles = False

if createNewPickles:
    subject_dict = createSubjectDict(prep_directory=prep_dir, windowsOS=windowsOS)
    PicklePrepData(subjects_dict=subject_dict, prep_directory=prep_dir, save_path=pickle_path, windowsOS=windowsOS)


# Clean data for NaNs
X_file = r"\X.npy"
y_file = r"\y.npy"
ID_file = r"\ID_frame.npy"

X = LoadNumpyPickles(pickle_path=pickle_path, file_name=X_file, windowsOS=windowsOS)
y = LoadNumpyPickles(pickle_path=pickle_path, file_name=y_file, windowsOS=windowsOS)
ID_frame = LoadNumpyPickles(pickle_path=pickle_path, file_name=ID_file, windowsOS=windowsOS)

clean_nonAugmented = False

if clean_nonAugmented:
    DeleteNan(X=X, y=y, ID_frame=ID_frame, save_path=pickle_path, windowsOS=windowsOS)
else:
    #pickle_folder = r"\whiteNoise_pickles"
    pickle_folder = r"\colorNoise_pickles"
    pickle_path = r"C:\Users\Albert Kjøller\Documents\GitHub\EEG_epilepsia" + pickle_folder
    os.makedirs(pickle_path, exist_ok=True)

    X_noise = LoadNumpyPickles(pickle_path=pickle_path, file_name=X_file, windowsOS=windowsOS)
    y_noise = LoadNumpyPickles(pickle_path=pickle_path, file_name=y_file, windowsOS=windowsOS)
    ID_frame_noise = LoadNumpyPickles(pickle_path=pickle_path, file_name=ID_file, windowsOS=windowsOS)

    DeleteNanNoise(X=X, y=y, ID_frame=ID_frame, X_noise=X_noise, y_noise=y_noise, ID_frame_noise=ID_frame_noise, save_path=pickle_path, windowsOS=windowsOS)

print("Breakpoint")