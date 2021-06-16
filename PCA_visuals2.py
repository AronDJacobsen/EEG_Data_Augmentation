import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#from bioinfokit.visuz import cluster
import matplotlib
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from prepData.dataLoader import LoadNumpyPickles
#from collections import defaultdict, Counter
#from sklearn.model_selection import train_test_split
from models.balanceData import *
#import plotly.express as px
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import os
import itertools

#from models.GAN import GAN
from summaryVisualz.GAN import GAN
from models.mixup import mixup

import pandas as pd


def cm_to_inch(value):
    return value/2.54

def subsample(X, y, n):
    indices = np.random.choice(len(X), n)
    return X[indices, :], y[indices], indices

def runPCA(X, y, n_components=0):
    #Xmean, Xerr = np.mean(X, axis=0), np.std(X, axis=0)
    #Xstd = (X - Xmean) / Xerr

    print("\nRunning PCA...")
    scaler = StandardScaler()
    scaler.fit(X)
    Xstd = scaler.transform(X)
    if n_components == 0:
        pca = PCA()
    else:
        pca = PCA(n_components=n_components)
    pca.fit(Xstd)

    print("PCA done!")
    return pca, Xstd




def undersample(X, y, ratio):
    #undersample
    label_size = Counter(y)
    major = max(label_size, key=label_size.get)
    decrease = label_size[1 - major] * ratio
    label_size[major] = int(np.round(decrease, decimals=0))
    Xnew, ynew = rand_undersample(X, y, arg=label_size, state=random_state_val, multi=False)
    return Xnew, ynew




def investigateBalancing(pca, X_test, y_test, p, colors, artifact, method, save, show):

    #projecting
    X_test = pca.transform(X_test)


    #combinations of principal components
    combi = list(itertools.combinations(p, 2))
    plt.figure(figsize=(cm_to_inch(42),cm_to_inch(10)))
    plt.subplots_adjust(left = 0.05, right=0.90)

    for i, (p1, p2) in enumerate(combi):
        plt.subplot(1, len(combi), i+1)
        #for test absent
        idx = [j for j, label in enumerate(y_test) if label == 0]
        xd, yd = X_test[:, p1][idx], X_test[:, p2][idx]
        plt.scatter(xd, yd, s=30, c=colors[0], label='Absent', alpha=0.3)

        #for test present
        idx = [j for j, label in enumerate(y_test) if label == 1]
        xd, yd = X_test[:, p1][idx], X_test[:, p2][idx]
        plt.scatter(xd, yd, s=30, c=colors[1], label='Present', alpha=0.9)

        plt.xlabel('Eigenvector '+str(p1+1))
        plt.ylabel('Eigenvector '+str(p2+1))

    #plt.title(method)
    plt.legend(bbox_to_anchor=(1.35, 1))

    plt.gcf()
    if save:
        if file_out:
            my_path = pickle_path+saving+'/other'
        else:
            my_path = os.path.dirname(os.path.abspath(__file__))+saving+'/other'
        my_file = method + "_" + artifact
        path = os.path.join(my_path, my_file)
        plt.savefig(path, dpi=800)
    if show:
        plt.show()

    plt.clf()








if __name__ == '__main__':
    random_state_val = 0
    np.random.seed(random_state_val)


    windowsOS = False
    #pickle_path = r"C:\Users\Albert Kjøller\Documents\GitHub\EEG_epilepsia"
    pickle_path = r"/Users/Jacobsen/Documents/GitHub/EEG_epilepsia" + "/"
    #pickle_path = r"/zhome/96/8/147177/Desktop/Fagprojekt3/EEG_epilepsia-main/"

    #show figs, save figs
    show = False
    save = True

    under_n = 1000 # for undersampling
    smote_n = 600 # for smote
    subsample_n = 100 # for augmentation

    smote_ratio = 10 # augmented ratios
    aug_smote_ratio = 2
    aug_ratio = 1
    colors = ["steelblue", "lightcoral"]

    principals = [0, 1, 2]
    n_components = 6

    fastrun = True
    sub = 100000

    file_out = True # if file in main map
    if file_out:
        saving = 'summaryVisualz'
    else:
        saving = '/summaryVisualz'

    if save == True and show == False:
        matplotlib.use('Agg')
        #stop=0

    # loading data - define which pickles to load (with NaNs or without)
    X_file = r"\X_clean.npy"  # X_file = r"\X.npy"
    y_file = r"\y_clean.npy"  # y_file = r"\y_clean.npy"
    ID_file = r"\ID_frame_clean.npy"  # ID_file = r"\ID_frame.npy"

    X = LoadNumpyPickles(pickle_path=pickle_path, file_name=X_file, windowsOS=windowsOS)
    y = LoadNumpyPickles(pickle_path=pickle_path, file_name=y_file, windowsOS=windowsOS)
    ID_frame = LoadNumpyPickles(pickle_path=pickle_path, file_name=ID_file, windowsOS=windowsOS)

    X, y, ID_frame = binary(X, y, ID_frame)

    artifact_names = ['eyem', 'chew', 'shiv', 'elpp', 'musc', 'null']
    #artifact_names = ['eyem']

    ####################################

    if fastrun:
        X, y, indices = subsample(X, y, sub)

    pca_unbal, Xstd = runPCA(X, y)


    #we want an outside subsample, test sample to project
    X_sample, y_sample, indices = subsample(X, y, subsample_n)
    X = np.delete(X, indices, axis=0)
    y = np.delete(y, indices, axis=0)



    #for every artifact
    for artifact in artifact_names:
        print('Artifact: ' + artifact)
        #extracting artifact
        artifact_pos = [i for i, name in enumerate(artifact_names) if name==artifact][0]
        y_arti = y[:, artifact_pos] # full data
        y_sample_arti = y_sample[:, artifact_pos] # sample data


        #Undersampling for this artifact
        X_under, y_under = rand_undersample(X, y_arti, arg='majority', state=random_state_val, multi=False)
        #extracting samples
        X_under_sample, y_under_sample, indices = subsample(X_under, y_under, subsample_n)
        X_under = np.delete(X_under, indices, axis=0)
        y_under = np.delete(y_under, indices, axis=0)
        #running pca
        pca_under, Xstd_under = runPCA(X_under, y_under)
        #finding standardizing values
        Xmean_under, Xerr_under = np.mean(X_under, axis=0), np.std(X_under, axis=0)
        #standardizing sample
        X_under_sample = (X_under_sample - Xmean_under) / Xerr_under


        #SMOTE for this artifact
        X_smote, y_smote = balanceData(X, y_arti, ratio=aug_smote_ratio , random_state_val=random_state_val)
        #extracting samples
        X_smote_sample, y_smote_sample, indices = subsample(X_smote, y_smote, subsample_n)
        X_smote = np.delete(X_smote, indices, axis=0)
        y_smote = np.delete(y_smote, indices, axis=0)
        #running pca
        pca_smote, Xstd_smote = runPCA(X_smote, y_smote)
        #finding standardizing values
        Xmean_smote, Xerr_smote = np.mean(X_smote, axis=0), np.std(X_smote, axis=0)
        #standardizing sample
        X_smote_sample = (X_smote_sample - Xmean_smote) / Xerr_smote


        #undersampling
        method = 'under'
        investigateBalancing(pca_under, X_under_sample, y_under_sample, principals, colors, artifact, method, save, show)


        #undersampling
        method = 'smote'
        investigateBalancing(pca_smote, X_smote_sample, y_smote_sample, principals, colors, artifact, method, save, show)

        print("Done with: " + artifact)


