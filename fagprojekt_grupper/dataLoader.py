import mne, torch, time, glob #, os, re
from itertools import repeat
import pandas as pd
import numpy as np
from collections import defaultdict
import preprocessing.loadData as loadData
from preprocessing.preprocessPipeline import TUH_rename_ch, readRawEdf, pipeline, spectrogramMake, slidingWindow


def processRawData(data_path, save_path, file_selected):
    # Redefining so we don't have to change variables through all of the function
    TUAR_dir = data_path
    save_dir = save_path

    TUAR_data = loadData.findEdf(path=TUAR_dir, selectOpt=False, saveDir=save_dir)

    # prepare TUAR output
    counter = 0  # debug counter
    tic = time.time()

    subjects = defaultdict(dict)
    for edf in file_selected:  # TUAR_data:
        subject_ID = edf.split('_')[0]
        if subject_ID in subjects.keys():
            subjects[subject_ID][edf] = TUAR_data[edf].copy()
        else:
            subjects[subject_ID] = {edf: TUAR_data[edf].copy()}

        # debug counter for subject error
        counter += 1
        print("\n\n%s is patient: %i\n\n" % (edf, counter))

        # initialize hierarchical dict
        proc_subject = subjects[subject_ID][edf]
        proc_subject = readRawEdf(proc_subject, saveDir=save_dir, tWindow=10, tStep=10 * .5,
                                  read_raw_edf_param={'preload': True})  # ,
        # "stim_channel": ['EEG ROC-REF', 'EEG LOC-REF', 'EEG EKG1-REF',
        #                  'EEG T1-REF', 'EEG T2-REF', 'PHOTIC-REF', 'IBI',
        #                  'BURSTS', 'SUPPR']})

        # find data labels
        labelPath = subjects[subject_ID][edf]['path'][-1].split(".edf")[0]
        proc_subject['annoDF'] = loadData.label_TUH_full(annoPath=labelPath + ".tse", window=[0, 50000],
                                                         saveDir=save_dir)

        # Makoto + PREP processing steps
        proc_subject["rawData"] = TUH_rename_ch(proc_subject["rawData"])
        ch_TPC = mne.pick_channels(proc_subject["rawData"].info['ch_names'],
                                   include=['Fp1', 'F7', 'T3', 'T5', 'F3', 'C3', 'P3', 'O1', 'Cz', 'Fp2', 'F4', 'C4',
                                            'P4', 'O2', 'F8', 'T4', 'T6', 'A1', 'A2'],
                                   exclude=['Fz', 'Pz', 'ROC', 'LOC', 'EKG1', 'T1', 'T2', 'BURSTS', 'SUPPR', 'IBI',
                                            'PHOTIC'])
        mne.pick_info(proc_subject["rawData"].info, sel=ch_TPC, copy=False)
        pipeline(proc_subject["rawData"], type="standard_1005", notchfq=60, downSam=100)

        # Generate output windows for (X,y) as (tensor, label)
        proc_subject["preprocessing_output"] = slidingWindow(proc_subject, t_max=proc_subject["rawData"].times.max(),
                                                             tStep=proc_subject["tStep"], FFToverlap=0.75, crop_fq=24,
                                                             annoDir=save_dir,
                                                             localSave={"sliceSave": True, "saveDir": save_dir,
                                                                        "local_return": True})  # r"C:\Users\anden\PycharmProjects"+"\\"})
        # except:
        #     print("sit a while and listen: %s" % subjects[subject_ID][edf]['path'])

    toc = time.time()
    print("\n~~~~~~~~~~~~~~~~~~~~\n"
          "it took %imin:%is to run preprocess-pipeline for %i patients\n with window length [%.2fs] and t_step [%.2fs]"
          "\n~~~~~~~~~~~~~~~~~~~~\n"
          % (int((toc - tic) / 60), int((toc - tic) % 60), len(subjects), subjects[subject_ID][edf]["tWindow"],
             subjects[subject_ID][edf]["tStep"]))



def createSubjectDict(prep_directory):
    # Setting directory
    subdirs = [sub_dir.split("/")[-1] for sub_dir in glob.glob(prep_directory + "**")]
    subjects = defaultdict(dict)

    # Loop through all session to load window-tensors into the "subjects"-dictionary
    for session_test in subdirs:
        subjectID = session_test.split("_")[0]
        dim_tensors = [dim_t.reshape(-1) for tensor_file in glob.glob(prep_directory + session_test + "/" + "**") for dim_t in torch.load(tensor_file)[0]]

        # Assigning window-tensors to their subjectID and session key.
        if subjectID in subjects.keys():
            subjects[subjectID][session_test] = dim_tensors
        else:
            subjects[subjectID] = {session_test: dim_tensors}

    return subjects


def loadPrepData(subjects_dir, prep_directory):

    #Setting start-values
    input_loader = []
    label_loader = []
    ID_loader = []
    # Encoding rules for the 6 classes.
    label_encoder = {"eyem":0, "chew":1, "shiv":2, "elpp":3, "musc":4, "null":5}

    # Looping through all session-folders to get the filepaths to all the preprocessed windows.
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
            error_id.add(pt_inputs[count].split("/")[5].split("_")[0])
            pass
        else:
            input_loader.append(pt_loaded[0].numpy().flatten().astype(np.float16))
            label_loader.append(pt_label)

            # Load subject ID for current window.
            ID_loader.append(pt_dir.split("/")[-2].split("_")[0])

    # Stacking data to matrices.
    X = np.stack(input_loader)
    y = np.stack(label_loader)
    ID_frame = np.stack(ID_loader)

    return X, y, ID_frame, error_id


#Overflødigt nu - skal vi bare slette det?
"""
    subdirs = [sub_dir.split("/")[-1] for sub_dir in glob.glob(prep_directory + "**")]

    subjects = defaultdict(dict)
    test_tensors = []
    all_labels = []
    indiv = []
    for session_test in subdirs:
        subjectID = session_test.split("_")[0]
        dim_tensors = [dim_t.reshape(-1) for tensor_file in glob.glob(prep_directory + session_test + "/" + "**") for dim_t in torch.load(tensor_file)[0]]
        target_labels = [torch.load(tensor_file)[1] for tensor_file in glob.glob(prep_directory + session_test + "/" + "**")]

        #TODO: Tjek lige om det her dictionary stadig er fedt!
        if subjectID in subjects.keys():
            subjects[subjectID][session_test] = dim_tensors
        else:
            subjects[subjectID] = {session_test: dim_tensors} #TODO: MÅske skal dim_tensors ændres?

        # Stacking tensors across dimensions / electrodes - instead of across tests
        el_tensors = []
        for i in range(19):
            el_tensors.append(torch.stack(dim_tensors[i::19]))

        test_tensors.append(torch.stack(el_tensors))
        #test_tensor = torch.stack(el_tensors)

        #TODO: Omformatér label-listen før vi kan bruge den som target - den skal encodes.
        all_labels.extend(target_labels)
        # Tilføjer subjectID nummeret til en liste, ligeså mange gange som target_labels.
        indiv.extend([subjectID] * len(target_labels))



    #TODO: Skift navnet på variablen test ud - det passer ikke ind i train/test-navngivning

    final_tensor = torch.cat(test_tensors, dim=1)
    torch.mean(final_tensor, dim=1)  # Mean across eletrodes, respectively
    torch.std(final_tensor, dim=1)  # Standard dev. across eletrodes, respectively

    return subjects, final_tensor, all_labels, indiv
"""
