import os, time, mne
import numpy as np
from collections import defaultdict
import prepData.loadData as loadData
from prepData.preprocessPipeline2 import TUH_rename_ch, readRawEdf, pipeline, spectrogramMake, slidingWindow

def processRawData(data_path, save_path, file_selected, windowsOS=False):
    # Redefining so we don't have to change variables through all of the function
    TUAR_dir = data_path
    save_dir = save_path

    TUAR_data = loadData.findEdf(path=TUAR_dir, selectOpt=False, saveDir=save_dir, windowsOS=windowsOS)

    # prepare TUAR output
    counter = 0  # debug counter
    tic = time.time()

    subjects = defaultdict(dict)
    all_subject_gender = {"male": [], "female": [], "other": []}
    all_subject_age = []
    subjects_marked = []

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
        proc_subject = readRawEdf(proc_subject, saveDir=save_dir, tWindow=1, tStep=1 * .5,
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
        pipeline(proc_subject["rawData"], type="standard_1005", notchfq=60, downSam=150)

        # Generate output windows for (X,y) as (tensor, label)
        proc_subject["preprocessing_output"] = slidingWindow(proc_subject, t_max=proc_subject["rawData"].times.max(),
                                                             tStep=proc_subject["tStep"], FFToverlap=0.75, crop_fq=24,
                                                             annoDir=save_dir,
                                                             localSave={"sliceSave": True, "saveDir": save_dir,
                                                                        "local_return": True})  # r"C:\Users\anden\PycharmProjects"+"\\"})
        # except:
        #     print("sit a while and listen: %s" % subjects[subject_ID][edf]['path'])

        if subject_ID not in subjects_marked:
            # catch age and gender for descriptive statistics
            if subjects[subject_ID][edf]["gender"].lower() == 'm':
                #all_subject_gender["male"].append(subjects[subject_ID][edf]["gender"].lower())
                all_subject_gender["male"].append(subjects[subject_ID][edf]["age"])
                # gender[0].append(subjects[id][edf]["gender"].lower())
            elif subjects[subject_ID][edf]["gender"].lower() == 'f':
                #all_subject_gender["female"].append(subjects[subject_ID][edf]["gender"].lower())
                all_subject_gender["female"].append(subjects[subject_ID][edf]["age"])
                # gender[1].append(subjects[id][edf]["gender"].lower())
            else:
                # all_subject_gender["other"].append(subjects[subject_ID][edf]["gender"].lower())
                all_subject_gender["other"].append(subjects[subject_ID][edf]["age"])
                # print(subjects[id][edf]["gender"].lower())
            all_subject_age.append(subjects[subject_ID][edf]["age"])
            # except:
            #     print("sit a while and listen: %s" % subjects[subject_ID][edf]['path'])
            subjects_marked.append(subject_ID)

    all_subject_age = np.array(all_subject_age)

    toc = time.time()


    print("\n~~~~~~~~~~~~~~~~~~~~\n"
          "it took %imin:%is to run preprocess-pipeline for %i patients\n with window length [%.2fs] and t_step [%.2fs]"
          "\n~~~~~~~~~~~~~~~~~~~~\n"
          % (int((toc - tic) / 60), int((toc - tic) % 60), len(subjects), subjects[subject_ID][edf]["tWindow"],
             subjects[subject_ID][edf]["tStep"]))


    return all_subject_age, all_subject_gender



if __name__ == '__main__':
    # Ensuring correct path
    os.chdir(os.getcwd())

    windowsOS = True

    # What is your execute path? #

    if windowsOS:
        save_dir = r"C:\Users\Albert Kjøller\Documents\GitHub\TUAR_full_data" + "\\"
        TUAR_dir = r"TUH_EEG_CORPUS\artifact_dataset" + "\\"
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



    # Which files should be processed - a subset or all?
    TUAR_data = loadData.findEdf(path=TUAR_dir, selectOpt=False, saveDir=save_dir, windowsOS=windowsOS)
    files_selected = TUAR_data.copy()


    # CALLING THE PREPROCESSING
    age, gender = processRawData(TUAR_dir, save_dir, files_selected, windowsOS=windowsOS)
    #all_subject_age, all_subject_gender = processRawData(TUAR_dir, save_dir, files_selected)

    #all_subject_age_hist = np.histogram(all_subject_age, range=(0,100))
    #plt.hist(all_subject_age, range=(0,100))
    #plt.show()
