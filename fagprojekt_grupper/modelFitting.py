import os, glob, torch, time, random
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from fagprojekt_grupper.dataLoader import processRawData, loadPrepData


os.chdir(os.getcwd())

save_dir = r"/Users/philliphoejbjerg/NovelEEG" + "/"  # ~~~ What is your execute path? # /Users/philliphoejbjerg/NovelEEG # "/Users/AlbertoK/Desktop/EEG/subset" + "/"  # ~~~ What is your execute path? # /Users/philliphoejbjerg/NovelEEG

TUAR_dir = r"data_TUH_EEG/TUH_EEG_CORPUS/artifact_dataset" + "/"  # \**\01_tcp_ar #\100\00010023\s002_2013_02_21
jsonDir = r"tmp.json"
prep_dir = r"tempData" + "/"

jsonDataDir = save_dir + jsonDir
TUAR_dirDir = save_dir + TUAR_dir
prep_dirDir = save_dir + prep_dir

"""
subset = ["00010418_s008_t000.edf", "00010079_s004_t002.edf", "00009630_s001_t001.edf", '00007952_s001_t001.edf',
          '00009623_s008_t004.edf', '00009623_s008_t005.edf', '00009623_s010_t000.edf',
          '00001006_s001_t001.edf', '00006501_s001_t000.edf', '00006514_s008_t001.edf', '00006514_s020_t001.edf',
          '00002348_s014_t008.edf', '00003573_s003_t000.edf', '00006224_s002_t001.edf', '00007020_s001_t001.edf',
          '00007981_s001_t001.edf', '00008476_s001_t000.edf', '00008527_s001_t000.edf', '00004473_s001_t000.edf',
          '00005672_s001_t001.edf', '00003849_s001_t000.edf', '00007647_s001_t001.edf', '00003008_s006_t006.edf']

# for all subjects run as: file_selected = TUAR_data
file_selected = subset.copy()
processRawData(TUAR_dir,save_dir,file_selected)
"""


subdirs = [sub_dir.split("/")[-1] for sub_dir in glob.glob(prep_dirDir + "**")]
indiv = list({subdir.split("_")[0] for subdir in subdirs})
train_indiv = random.sample(indiv, 10)
test_indiv = np.setdiff1d(indiv, train_indiv)

subjects, X, y_list, indiv = loadPrepData(prep_dirDir)
p_inspect = subdirs[-1]
subjects[p_inspect.split("_")[0]][p_inspect]

#TODO: Does not work yet!
"""
# Encoding of labels
#labels = y_list
#labels = ['null', 'chew', 'eyem', 'elpp', 'musc', 'shiv']
le = LabelEncoder()
targets = le.fit_transform(labels)
targets = torch.as_tensor(targets)
"""



subjects_list = list(subjects.keys())
train_subjects = subjects_list[:10]
test_subjects = subjects_list[10:20]

example = '00004625_s003_t001'

#new = pd.DataFrame.from_dict(subjects[example.split("_")[0]][example][i][0])