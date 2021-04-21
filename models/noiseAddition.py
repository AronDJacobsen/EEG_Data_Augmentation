import os, time, mne, shutil, glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import prepData.loadData as loadData
from prepData.preprocessPipeline import TUH_rename_ch, readRawEdf, pipeline, spectrogramMake, slidingWindow
from scipy.signal import butter, lfilter, freqz
from prepData.dataLoader import *

### BUTTERWORTH LOWPASS FILTERS TAKEN FROM
# https://stackoverflow.com/questions/25191620/creating-lowpass-filter-in-scipy-understanding-methods-and-units
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data, axis=1)
    return y


####

def generateNoisyData(data_path, save_path, file_selected, variance, use_covariance=False, windowsOS=False, cutoff_freq=None, sample_rate=150, order=5, save_fig=False):
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



        # Create directories for saving noisy files similarly to normal files in another directory
        filename = proc_subject["rawData"].filenames[0]
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

        #proc_subject = readRawEdf(proc_subject, saveDir=save_dir, tWindow=1, tStep=1 * .25,  # 75% temporalt overlap
        #                          read_raw_edf_param={'preload': True})  # ,

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



        # NOISE ADDITION START SETUP!
        orig_object = proc_subject["rawData"]
        raw_signals = proc_subject["rawData"].get_data()
        n_chan, n_obs = raw_signals.shape
        noisy_signals = np.empty((n_chan, n_obs))

        covar = mne.compute_raw_covariance(orig_object)

        if save_fig:
            fig = orig_object.plot()

            base = ("\\").join(os.getcwd().split("\\")[:-1])
            freq_str = str(cutoff_freq).split(".")[0]
            figure_name = base + r"\Plots\NoiseAddition_visualization\clean.png"
            fig.savefig(figure_name)

            covar.plot(orig_object.info, proj=True)
            covar.plot_topomap(orig_object.info, proj=True)

        if use_covariance:
            Sigma = covar.data * variance #TODO: Giver det mening med variance her?
            N = Sigma.shape[0]
            white_noise = np.random.multivariate_normal(np.zeros(N), Sigma, n_obs).T


            # TODO: Tjek om AXIS er korrekt i butter_lowpass_filter når man kalder lfilter
            if cutoff_freq is not None:
                colored_noise = butter_lowpass_filter(white_noise, cutoff_freq, sample_rate, order)
                noise = colored_noise

            else:
                noise = white_noise

            noisy_signals = raw_signals + noise

        else:
            for i, chan_signal in enumerate(raw_signals):
                mean = 0
                std = np.sqrt(variance)
                white_noise = np.random.normal(mean, std, n_obs)

                if cutoff_freq is not None:
                    colored_noise = butter_lowpass_filter(white_noise, cutoff_freq, sample_rate, order)
                    noise = colored_noise
                else:
                    noise = white_noise

                noisy_signals[i] = chan_signal + noise


        orig_object._data = noisy_signals

        if save_fig:
            fig = orig_object.plot(title="Cutoff freq: " + str(cutoff_freq))

            base = ("\\").join(os.getcwd().split("\\")[:-1])
            freq_str = str(cutoff_freq).split(".")[0]
            figure_name = base + r"\Plots\NoiseAddition_visualization\freq{}_var{}.png".format(freq_str, variance)
            fig.savefig(figure_name)

        proc_subject["rawData"] = orig_object



        # Generate output windows for (X,y) as (tensor, label)
        proc_subject["preprocessing_output"] = slidingWindow(proc_subject, t_max=proc_subject["rawData"].times.max(),
                                                             tStep=proc_subject["tStep"], FFToverlap=0.75, crop_fq=24,
                                                             annoDir=save_dir,
                                                             localSave={"sliceSave": True, "saveDir": save_dir,
                                                                        "local_return": False})  # saving to harddrive as float16 type
        # except:
        #     print("sit a while and listen: %s" % subjects[subject_ID][edf]['path'])


    toc = time.time()

    print("\n~~~~~~~~~~~~~~~~~~~~\n"
          "it took %imin:%is to run preprocess-pipeline for %i patients\n with window length [%.2fs] and t_step [%.2fs]"
          "\n~~~~~~~~~~~~~~~~~~~~\n"
          % (int((toc - tic) / 60), int((toc - tic) % 60), len(subjects), subjects[subject_ID][edf]["tWindow"],
             subjects[subject_ID][edf]["tStep"]))

    filepath = r"C:\Users\Albert Kjøller\Documents\GitHub\EEG_epilepsia"
    SaveNumpyPickles(filepath, r"\subjects_var{}_cutoff{}".format(variance, cutoff_freq), subjects, windowsOS=True)
    PicklePrepData(subjects, r"C:\Users\Albert Kjøller\Documents\GitHub\TUAR_full_data\tempData", filepath, windowsOS=True)


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

    # Select EDF-files to be augmented with noise addition.
    files_selected = TUAR_data.copy()

    EDFs = np.array(list(files_selected.keys()))

    # Choose one EDF for visualization of noise
    # N = len(EDFs)
    N = 1
    use_subset = False

    if use_subset:
        files_selected_sub = defaultdict(dict)
        np.random.seed = 20
        chosen = np.random.choice(len(EDFs), N)
        chosen = [0]

        for key in list(EDFs[chosen]):
            files_selected_sub[key] = files_selected[key]

        files_selected = files_selected_sub

    # CALLING THE PREPROCESSING to get noisy preprocessed data files
    # Max cutoff_freq is half the sample rate!
    cutoff_freq = None

    # How much should the covariance have as impact
    variance = 1
    use_covar = True
    save_fig = False

    # TODO: Add edf-name to the plot!
    generateNoisyData(TUAR_dir, save_dir, files_selected, variance=variance, use_covariance=use_covar, windowsOS=windowsOS, cutoff_freq=cutoff_freq, sample_rate=150, order=5, save_fig=save_fig)