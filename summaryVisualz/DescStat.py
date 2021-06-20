import numpy as np
import os

from prepData.dataLoader import *

pickle_path = r'/Users/philliphoejbjerg/Documents/GitHub/EEG_epilepsia/'
#X = LoadNumpyPickles(pickle_path, r'X_clean.npy', True)
y = LoadNumpyPickles(pickle_path, r'y_clean.npy', True)


def DescriptiveStats_labels(labels):
    # Artifacts
    n_labels = {i: np.unique(labels[:, i], return_counts=True)[1][1] for i in range(len(labels[0]))}
    N = sum(n_labels.values())
    N_windows = 1344862
    label_ratio = np.array(list(n_labels.values()) )/ N_windows
    return n_labels, N, label_ratio

def DescriptiveStats_person(tempdata_path = '/Users/philliphoejbjerg/NovelEEG/tempData'):
    # Subject
    import os
    time_pers = dict()
    for file in os.listdir(tempdata_path):
        session = os.listdir(tempdata_path + '/' + file)
        if file.split('_')[0] in time_pers.keys():
            time_pers[file.split('_')[0]] += max([float(session[i].split('_')[2][:-4]) for i in range(len(session))])
        else:
            time_pers[file.split('_')[0]] = max([float(session[i].split('_')[2][:-4]) for i in range(len(session))])
    return time_pers


n_labels, N, label_ratio = DescriptiveStats_labels(y)
#time_pers = DescriptiveStats_person()

