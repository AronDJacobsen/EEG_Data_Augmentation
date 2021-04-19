import os, time, mne, shutil, glob
import numpy as np
from collections import defaultdict
import prepData.loadData as loadData
from prepData.preprocessPipeline import TUH_rename_ch, readRawEdf, pipeline, spectrogramMake, slidingWindow
from scipy.signal import butter, lfilter, freqz

### BUTTERWORTH LOWPASS FILTERS TAKEN FROM
# https://stackoverflow.com/questions/25191620/creating-lowpass-filter-in-scipy-understanding-methods-and-units
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


####

def generateNoisyData(data_path, save_path, file_selected, windowsOS=False, cutoff_freq=None, sample_rate=150, order=5):
    # Redefining so we don't have to change variables through all of the function
    TUAR_dir = data_path
    save_dir = save_path

    TUAR_data = loadData.findEdf(path=TUAR_dir, selectOpt=False, saveDir=save_dir, windowsOS=windowsOS)

    # prepare TUAR output
    counter = 0  # debug counter
    tic = time.time()

    subjects = defaultdict(dict)
    #all_subject_gender = {"male": [], "female": [], "other": []}
    #all_subject_age = []
    #subjects_marked = []

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
        proc_subject = readRawEdf(proc_subject, saveDir=save_dir, tWindow=1, tStep=1 * .25, # 75% temporalt overlap
                                  read_raw_edf_param={'preload': True})  # ,
        # "stim_channel": ['EEG ROC-REF', 'EEG LOC-REF', 'EEG EKG1-REF',
        #                  'EEG T1-REF', 'EEG T2-REF', 'PHOTIC-REF', 'IBI',
        #                  'BURSTS', 'SUPPR']})


        # NOISE ADDITION STEP!
        orig_object = proc_subject["rawData"]
        raw_signals = proc_subject["rawData"].get_data()
        n_chan, n_obs = raw_signals.shape
        noisy_signals = np.empty((n_chan, n_obs))


        # TODO: Zero mean with variance as parameter - maybe covariance?
        for i, chan_signal in enumerate(raw_signals):
            mean = 0
            std = np.std(chan_signal)
            white_noise = np.random.normal(mean, std, n_obs)

            if cutoff_freq is not None:
                # TODO: Lav lav-pass filtrering af white noise. TJEK OM DET VIRKER!
                colored_noise = butter_lowpass_filter(white_noise, cutoff_freq, sample_rate, order)
                noise = colored_noise

            else:
                noise = white_noise

            # GAUSSIAN NOISE FUNCTION I MNE (KAN NOK IKKE HÅNDTERE COLORED NOISE
            noisy_signals[i] = chan_signal + noise

        orig_object._data = noisy_signals
        proc_subject["rawData"] = orig_object

        # TODO: CHANGE FILE NAME!
        # Create directories for saving noisy files similarly to normal files
        filename = orig_object.filenames[0]
        filename = filename.split("\\")
        folder_pos = [i for i, m in enumerate(filename) if m == "TUH_EEG_CORPUS"][0]
        orig_path = ("\\").join(filename[:-1])
        filename[folder_pos] = "NoisyTUH"
        try:
            os.makedirs(("\\").join(filename[:-1]))
        except FileExistsError:
            print("Directory is already created!")

        fileList = [i for i in glob.glob(orig_path + "\\" + "**")]
        for item in fileList:
            src = item

            dst = filename[:-1]
            dst.append(item.split("\\")[-1])
            dst = ("\\").join(dst)

            shutil.copyfile(src, dst)

        proc_subject['path'] = [("\\").join(filename[folder_pos:])]

        proc_subject = readRawEdf(proc_subject, saveDir=save_dir, tWindow=1, tStep=1 * .25,  # 75% temporalt overlap
                                  read_raw_edf_param={'preload': True})  # ,

        # find data labels
        labelPath = subjects[subject_ID][edf]['path'][-1].split(".edf")[0]
        proc_subject['annoDF'] = loadData.label_TUH_full(annoPath=labelPath + ".tse", window=[0, 50000],
                                                         saveDir=save_dir)

        # Makoto + PREP processing steps
        proc_subject["rawData"] = TUH_rename_ch(proc_subject["rawData"])
        # ch_TPC = mne.pick_channels(proc_subject["rawData"].info['ch_names'],
        #                           include=['Fp1', 'F7', 'T3', 'T5', 'F3', 'C3', 'P3', 'O1', 'Cz', 'Fp2', 'F4', 'C4',
        #                                    'P4', 'O2', 'F8', 'T4', 'T6', 'A1', 'A2'],
        #                           exclude=['Fz', 'Pz', 'ROC', 'LOC', 'EKG1', 'T1', 'T2', 'BURSTS', 'SUPPR', 'IBI',
        #                                    'PHOTIC'])
        # mne.pick_info(proc_subject["rawData"].info, sel=ch_TPC, copy=False)

        # Hardcoding channel pick to avoid false referencing
        proc_subject["rawData"].pick_channels(ch_names=['Fp1', 'F7', 'T3', 'T5', 'F3', 'C3', 'P3', 'O1', 'Cz',
                                                        'Fp2', 'F4', 'C4', 'P4', 'O2', 'F8', 'T4', 'T6', 'A1', 'A2'])

        # downSampling must be above the filter frequency in FIR to avoid aliasing
        pipeline(proc_subject["rawData"], type="standard_1005", notchfq=60, downSam=150) # TO avoid aliasing for the FIR-filter

        # Generate output windows for (X,y) as (tensor, label)
        proc_subject["preprocessing_output"] = slidingWindow(proc_subject, t_max=proc_subject["rawData"].times.max(),
                                                             tStep=proc_subject["tStep"], FFToverlap=0.75, crop_fq=24,
                                                             annoDir=save_dir,
                                                             localSave={"sliceSave": True, "saveDir": save_dir,
                                                                        "local_return": False})  # saving to harddrive as float16 type
        # except:
        #     print("sit a while and listen: %s" % subjects[subject_ID][edf]['path'])


    toc = time.time()
    mne.simulation.add_noise(proc_subject["rawData"], )

    print("\n~~~~~~~~~~~~~~~~~~~~\n"
          "it took %imin:%is to run preprocess-pipeline for %i patients\n with window length [%.2fs] and t_step [%.2fs]"
          "\n~~~~~~~~~~~~~~~~~~~~\n"
          % (int((toc - tic) / 60), int((toc - tic) % 60), len(subjects), subjects[subject_ID][edf]["tWindow"],
             subjects[subject_ID][edf]["tStep"]))


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

    # Choose which files we wish to augment!
    # TODO: How will we do this? Randomly select EDF's or choose over subjects?
    # TODO: I believe it should only train on this augmented data? Maybe we will just create a folder with preprocessed
    # TODO: data (like tempData) and draw on the files for specific ID's when training? I think files_selected can be passed
    # on to the noisyData-function.

    # Select EDF-files to be augmented with noise addition.
    how_many = 10

    EDF_files = list(TUAR_data.keys())
    np.random.choice(EDF_files, how_many)
    files_selected = TUAR_data.copy()


    # CALLING THE PREPROCESSING to get noisy preprocessed data files
    # TODO: Determine the sample_rate and the order of the Butterworth-filter!
    generateNoisyData(TUAR_dir, save_dir, files_selected, windowsOS=windowsOS, cutoff_freq=30.0, sample_rate=150, order=5)