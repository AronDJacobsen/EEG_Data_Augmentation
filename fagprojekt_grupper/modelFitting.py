import os, glob, torch, time
import numpy as np
#from sklearn...
from fagprojekt_grupper.dataLoader import processRawData, loadPrepData


os.chdir(os.getcwd())

save_dir = r"/Users/AlbertoK/Desktop/EEG/subset" + "/"  # ~~~ What is your execute path?
TUAR_dir = r"data_TUH_EEG/TUH_EEG_CORPUS/artifact_dataset" + "/"  # \**\01_tcp_ar #\100\00010023\s002_2013_02_21
jsonDir = r"tmp.json"
prep_dir = r"tempData" + "/"

jsonDataDir = save_dir + jsonDir
TUAR_dirDir = save_dir + TUAR_dir
prep_dirDir = save_dir + prep_dir


subset = ["00010418_s008_t000.edf", "00010079_s004_t002.edf", "00009630_s001_t001.edf", '00007952_s001_t001.edf',
          '00009623_s008_t004.edf', '00009623_s008_t005.edf', '00009623_s010_t000.edf',
          '00001006_s001_t001.edf', '00006501_s001_t000.edf', '00006514_s008_t001.edf', '00006514_s020_t001.edf',
          '00002348_s014_t008.edf', '00003573_s003_t000.edf', '00006224_s002_t001.edf', '00007020_s001_t001.edf',
          '00007981_s001_t001.edf', '00008476_s001_t000.edf', '00008527_s001_t000.edf', '00004473_s001_t000.edf',
          '00005672_s001_t001.edf', '00003849_s001_t000.edf', '00007647_s001_t001.edf', '00003008_s006_t006.edf']

# for all subjects run as: file_selected = TUAR_data
file_selected = subset.copy()

processRawData(TUAR_dir,save_dir,file_selected)

subdirs = [sub_dir.split("/")[-1] for sub_dir in glob.glob(prep_dirDir + "**")]
subjects = loadPrepData(prep_dirDir)
p_inspect = subdirs[-1]
subjects[p_inspect.split("_")[0]][p_inspect]