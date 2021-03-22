import mne, torch, time, glob, pickle, os #, re
from itertools import repeat
import pandas as pd
import numpy as np
from collections import defaultdict
import prepData.loadData as loadData
from prepData.preprocessPipeline import TUH_rename_ch, readRawEdf, pipeline, spectrogramMake, slidingWindow

def createSubjectDict(prep_directory, windowsOS=False):
    # Setting directory
    if windowsOS:
        subdirs = [sub_dir.split("\\")[-1] for sub_dir in glob.glob(prep_directory + "**")]
    else:
        subdirs = [sub_dir.split("/")[-1] for sub_dir in glob.glob(prep_directory + "**")]
    subjects = defaultdict(dict)

    # Loop through all session to load window-tensors into the "subjects"-dictionary
    for session_test in subdirs:
        subjectID = session_test.split("_")[0]
        if windowsOS:
            dim_tensors = [dim_t.reshape(-1) for tensor_file in glob.glob(prep_directory + session_test + "\\" + "**")
                           for dim_t in torch.load(tensor_file)[0]]
        else:
            dim_tensors = [dim_t.reshape(-1) for tensor_file in glob.glob(prep_directory + session_test + "/" + "**") for dim_t in torch.load(tensor_file)[0]]

        # Assigning window-tensors to their subjectID and session key.
        if subjectID in subjects.keys():
            subjects[subjectID][session_test] = dim_tensors
        else:
            subjects[subjectID] = {session_test: dim_tensors}

    return subjects


def loadPrepData(subjects_dir, prep_directory, windowsOS = False):

    #Setting start-values
    input_loader = []
    label_loader = []
    ID_loader = []
    # Encoding rules for the 6 classes.
    label_encoder = {"eyem":0, "chew":1, "shiv":2, "elpp":3, "musc":4, "null":5}

    # Looping through all session-folders to get the filepaths to all the preprocessed windows.
    if windowsOS:
        pt_inputs = [pt_dir for ID in subjects_dir for session in subjects_dir[ID].keys() for pt_dir in
                     glob.glob(prep_directory + session + "\\" + "**")]
    else:
        pt_inputs = [pt_dir for ID in subjects_dir for session in subjects_dir[ID].keys() for pt_dir in glob.glob(prep_directory + session + "/"+ "**")]

    # Loading all preprocessed files and dividing them into data, X, and target variable, y.
    error_id = set() # set for keeping track of subjects with errors in data.
    for count, pt_dir in enumerate(pt_inputs):
        pt_loaded = torch.load(pt_dir)
        pt_label = np.zeros(len(label_encoder))

        # Encoding labels as a one-hot encoding.
        for label in pt_loaded[1]:
            pt_label[label_encoder[label]] = 1
        # Rearranging tensors to flattened numpy arrays, while checking for correct shape.
        if len(pt_loaded[0].numpy().flatten().astype(np.float16)) != 4579:
            if windowsOS:
                error_id.add(pt_inputs[count].split("\\")[5].split("_")[0])
            else:
                error_id.add(pt_inputs[count].split("/")[5].split("_")[0])
            pass
        else:
            input_loader.append(pt_loaded[0].numpy().flatten().astype(np.float16))
            label_loader.append(pt_label)

            # Load subject ID for current window.
            if windowsOS:
                ID_loader.append(pt_dir.split("\\")[-2].split("_")[0])
            else:
                ID_loader.append(pt_dir.split("/")[-2].split("_")[0])

    # Stacking data to matrices.
    X = np.stack(input_loader)
    y = np.stack(label_loader)
    ID_frame = np.stack(ID_loader)

    return X, y, ID_frame, error_id


# Hvis man arbejder med fulde datasæt er det megget smartere først at lave pickles, og så hente det ned bagefter med LoadPickles
def CreatePickles(windowsOS=False):
    # Ensuring correct path
    os.chdir(os.getcwd())


    # What is your execute path? #

    if windowsOS:
        save_dir = r"C:\Users\TUAR_full_data" + "\\"
        TUAR_dir = r"data_TUH_EEG\TUH_EEG_CORPUS\artifact_dataset" + "\\"
        prep_dir = r"tempData" + "\\"
    else:
        save_dir = r"/Users/AlbertoK/Desktop/EEG/subset" + "/"
        TUAR_dir = r"data_TUH_EEG/TUH_EEG_CORPUS/artifact_dataset" + "/"  # \**\01_tcp_ar #\100\00010023\s002_2013_02_21
        prep_dir = r"tempData" + "/"

        # Albert path mac: save_dir = r"/Users/AlbertoK/Desktop/EEG/subset" + "/"
        # Aron:
        # Phillip: save_dir = r"/Users/philliphoejbjerg/NovelEEG" + "/"

    jsonDir = r"tmp.json"

    jsonDataDir = save_dir + jsonDir
    TUAR_dirDir = save_dir + TUAR_dir
    prep_dirDir = save_dir + prep_dir

    # Creating directory for subjects, sessions, windows for easy extraction of tests in loadPrepData
    subjects = createSubjectDict(prep_dirDir, windowsOS=windowsOS)

    # X = Number of windows, 19*241    y = Number of windows, 6 categories     ID_frame = subjectID for each window
    X, y, ID_frame, error_id = loadPrepData(subjects, prep_dirDir, windowsOS=windowsOS)


    pickle.dump(X, open("X.pkl", "wb"))
    pickle.dump(y, open("y.pkl", "wb"))
    pickle.dump(ID_frame, open("ID_frame.pkl", "wb"))

    return X, y, ID_frame, error_id

def LoadPickles(DelNan = False):

    # put the pickles in the prepData folder
    os.chdir(os.getcwd())

    X = pickle.load(open("X.pkl", "rb"))
    y = pickle.load(open("y.pkl", "rb"))
    ID_frame = pickle.load(open("ID_frame.pkl", "rb"))


    # Deletes rows with NaN values.
    if DelNan == True:
        X, y, ID_frame = DeleteNan(X, y, ID_frame)



    return X, y, ID_frame

def DeleteNan(X, y, ID_frame):
    # NanList in decreasing order, shows window-index with NaN.
    NanList = []
    for i in range(len(X)):
        windowVals = np.isnan(X[i])

        if np.any(windowVals==True):
            print(np.any(windowVals==True))
            NanList.append(i)

    # TODO: DelNan - no Nans in current data
    # NanList = [47698, 47687, 47585, 47569, 47490, 47475, 47436, 47409, 47339, 35919, 35914, 35759, 14819, 14815, 14802, 14787, 14786, 14781, 14776, 14770, 14765, 14758, 14752, 14745, 14741, 14726, 14717, 2246, 2242, 2064]

    for ele in NanList:
        X = np.delete(X, (ele), axis = 0)
        y = np.delete(y, (ele), axis=0)
        ID_frame = np.delete(ID_frame, (ele))

    return X, y, ID_frame