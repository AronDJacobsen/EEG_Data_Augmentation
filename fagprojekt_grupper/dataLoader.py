import mne, torch, time, glob #, os, re
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

# TODO  -  Find på et andet navn
def loadPrepData(prep_directory):
    subdirs = [sub_dir.split("/")[-1] for sub_dir in glob.glob(prep_directory + "**")]

    subjects = defaultdict(dict)
    test_tensors = []
    all_labels = []
    for test in subdirs:
        subjectID = test.split("_")[0]
        dim_tensors = [dim_t.reshape(-1) for tensor_file in glob.glob(prep_directory + test + "/" + "**") for dim_t in torch.load(tensor_file)[0]]
        test_labels = [torch.load(tensor_file)[1] for tensor_file in glob.glob(prep_directory + test + "/" + "**")]

        #TODO: Tjek lige om det her dictionary stadig er fedt!
        if subjectID in subjects.keys():
            subjects[subjectID][test] = dim_tensors
        else:
            subjects[subjectID] = {test: dim_tensors} #TODO: MÅske skal dim_tensors ændres?

        # Stacking tensors across dimensions / electrodes - instead of across tests
        el_tensors = []
        for i in range(19):
            el_tensors.append(torch.stack(dim_tensors[i::19]))

        test_tensors.append(torch.stack(el_tensors))
        #test_tensor = torch.stack(el_tensors)

        #TODO: Omformatér label-listen før vi kan bruge den som target - den skal encodes.
        all_labels.extend(test_labels)

    #TODO: Skift navnet på variablen test ud - det passer ikke ind i train/test-navngivning

    final_tensor = torch.cat(test_tensors, dim=1)
    torch.mean(final_tensor, dim=1)  # Mean across eletrodes, respectively
    torch.std(final_tensor, dim=1)  # Standard dev. across eletrodes, respectively

    return subjects, final_tensor, all_labels

