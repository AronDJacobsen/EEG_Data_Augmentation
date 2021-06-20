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


def plotPCA(X, y, p1, p2, colors, method):
    classes = ['Absent', 'Present']
    for val, name in enumerate(classes):
        indices = [j for j, label in enumerate(y) if label == val]
        if val == 0:
            transparency = 0.3
        else:
            transparency = 0.9
        plt.scatter(X[:, p1][indices], X[:, p2][indices], s=50, c=colors[val], label=name, alpha=transparency)

    plt.xlabel('Eigenvector '+str(p1+1))
    plt.ylabel('Eigenvector '+str(p2+1))
    #plt.title("{:s} - {:s}".format(artifact, method))
    plt.title(method)
    plt.legend(loc=1)



def under(X, y, pca, balancing_n, colors, artifact, random_state_val, show, save):

    #sampling
    X_before, y_before, idx = subsample(X, y, balancing_n)
    #projecting
    X_p_b = pca.transform(X_before) #X_sample outside of pca basis

    plt.figure(figsize=(cm_to_inch(20),cm_to_inch(10)))
    ax1 = plt.subplot(1, 2, 1)
    method = 'Unbalanced data'
    plotPCA(X_p_b, y_before, 0, 1, colors, method)

    #undersampling
    Xnew, ynew = undersample(X_before, y_before, 1)
    X_p_a = pca.transform(Xnew) #X_sample outside of pca basis
    #plotting sample on base
    ax2 = plt.subplot(1, 2, 2, sharex=ax1, sharey=ax1)
    method = 'Balanced data'
    plotPCA(X_p_a, ynew, 0, 1, colors, method)
    plt.gcf()
    if save:
        if file_out:
            my_path = pickle_path+saving+'/other'
        else:
            my_path = os.path.dirname(os.path.abspath(__file__))+saving+'/other'
        my_file = 'undersampling' + "_" + artifact + "_example"
        path = os.path.join(my_path, my_file)
        plt.savefig(path, dpi=800)
    if show:
        plt.show()
    plt.clf()



def undersample(X, y, ratio):
    #undersample
    label_size = Counter(y)
    major = max(label_size, key=label_size.get)
    decrease = label_size[1 - major] * ratio
    label_size[major] = int(np.round(decrease, decimals=0))
    Xnew, ynew = rand_undersample(X, y, arg=label_size, state=random_state_val, multi=False)
    return Xnew, ynew


def investigateSMOTE(X, y, pca, balancing_n, ratio, random_state_val, artifact, show, save, n_components):

    # højere ratio for større effekt
    ### using original basis (unbalanced) ###

    #we want an outside subsample
    X_unbal, y_unbal, indices = subsample(X, y, balancing_n)
    X_unbal_p = pca.transform(X_unbal)


    plt.figure(figsize=(cm_to_inch(30),cm_to_inch(10)))
    ax1 = plt.subplot(1, 3, 1)
    method = 'Unbalanced data'
    plotPCA(X_unbal_p, y_unbal, 0, 1, colors, method)

    X_unbal_down, y_unbal_down = undersample(X_unbal, y_unbal, ratio)
    X_unbal_down_p = pca.transform(X_unbal_down)
    ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
    method = 'Downsampled data'
    plotPCA(X_unbal_down_p, y_unbal_down, 0, 1, colors, method)

    #only three neighbors
    if not np.sum(y_unbal_down==1) < 4:
        X_smote, y_smote = smote(X_unbal_down, y_unbal_down, multi = False, state = random_state_val, k_neighbors=3)
        X_smote_p = pca.transform(X_smote)
        ax3 = plt.subplot(1, 3, 3, sharex=ax2, sharey=ax2)
        method = 'Balanced data'
        plotPCA(X_smote_p, y_smote, 0, 1, colors, method)
    plt.gcf()
    method = 'SMOTE'
    if save:
        if file_out:
            my_path = pickle_path+saving+'/other'
        else:
            my_path = os.path.dirname(os.path.abspath(__file__))+saving+'/other'
        my_file = artifact + "_" + method + "_example"
        path = os.path.join(my_path, my_file)
        plt.savefig(path, dpi=800)
    if show:
        plt.show()

    plt.clf()



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
        my_file = artifact + "_" + method
        path = os.path.join(my_path, my_file)
        plt.savefig(path, dpi=800)
    if show:
        plt.show()

    plt.clf()






def investigateAugmentation(pca, X_test, y_test, X_aug, y_aug, p, colors, artifact, method, subsample_n, aug_ratio):


    X_aug, y_aug, indices = subsample(X_aug, y_aug, aug_ratio*subsample_n)

    #projecting
    X_test_p = pca.transform(X_test)
    X_aug_p = pca.transform(X_aug)

    #combinations of principal components
    combi = list(itertools.combinations(p, 2))
    plt.figure(figsize=(cm_to_inch(42),cm_to_inch(10)))
    plt.subplots_adjust(left = 0.05, right=0.85)

    for i, (p1, p2) in enumerate(combi):
        plt.subplot(1, len(combi), i+1)
        #for test absent
        idx = [j for j, label in enumerate(y_test) if label == 0]
        xd, yd = X_test_p[:, p1][idx], X_test_p[:, p2][idx]
        plt.scatter(xd, yd, s=30, c=colors[0], label='Original-Absent', alpha=0.3)

        #for test present
        idx = [j for j, label in enumerate(y_test) if label == 1]
        xd, yd = X_test_p[:, p1][idx], X_test_p[:, p2][idx]
        plt.scatter(xd, yd, s=30, c=colors[1], label='Original-Present', alpha=0.9)

        #for augmented absent
        idx = [j for j, label in enumerate(y_aug) if label == 0]
        xd, yd = X_aug_p[:, p1][idx], X_aug_p[:, p2][idx]
        plt.scatter(xd, yd, s=50, c=colors[0], marker="+", label='Augmented-Absent', alpha=0.5)

        #for augmented present
        idx = [j for j, label in enumerate(y_aug) if label == 1]
        xd, yd = X_aug_p[:, p1][idx], X_aug_p[:, p2][idx]
        plt.scatter(xd, yd, s=50, c=colors[1], marker="+", label='Augmented-Present', alpha=0.9)
        plt.xlabel('Eigenvector '+str(p1+1))
        plt.ylabel('Eigenvector '+str(p2+1))

    #plt.title(method)
    plt.legend(bbox_to_anchor=(1.05, 1))

    plt.gcf()
    if save:
        if file_out:
            my_path = pickle_path+saving+'/principals'
        else:
            my_path = os.path.dirname(os.path.abspath(__file__))+saving+'/principals'
        my_file = artifact + "_" + method
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
    bal_n = 2*subsample_n

    smote_ratio = 10 # augmented ratios
    aug_smote_ratio = 2
    aug_ratio = 1
    colors = ["steelblue", "lightcoral"]

    principals = [0, 1, 2]
    n_components = 6

    fastrun = False
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


    ### NOISE ADDITION white noise###
    # Noise addition - most of this code is just taken from the pipeline...
    if windowsOS:
        pickle_path_aug = pickle_path + r"\augmentation_pickles"
        noise_experiment = r"\whitenoise_covarOne"
    else:
        pickle_path_aug = pickle_path + "augmentation_pickles" + "/"
        noise_experiment = "whitenoise_covarOne" + "/"

    X_white = LoadNumpyPickles(pickle_path_aug + noise_experiment, file_name=X_file, windowsOS=windowsOS)
    y_white = LoadNumpyPickles(pickle_path_aug + noise_experiment, file_name=y_file, windowsOS=windowsOS)

    if fastrun:
        X_white, y_white, indices = subsample(X_white, y_white, sub)


    ### NOISE ADDITION colored noise###
    # Noise addition - most of this code is just taken from the pipeline...
    if windowsOS:
        pickle_path_aug = pickle_path + r"\augmentation_pickles"
        noise_experiment = r"\colornoise30Hz_covarOne"
    else:
        pickle_path_aug = pickle_path + "augmentation_pickles" + "/"
        noise_experiment = "colornoise30Hz_covarOne" + "/"

    #experiment = 'DataAug_white_noiseAdd_LR'  # 'DataAug_color_noiseAdd_LR'
    #experiment_name = "_DataAug_white_Noise"  # "_DataAug_color_Noise" added to saving files

    X_color = LoadNumpyPickles(pickle_path_aug + noise_experiment, file_name=X_file, windowsOS=windowsOS)
    y_color = LoadNumpyPickles(pickle_path_aug + noise_experiment, file_name=y_file, windowsOS=windowsOS)

    if fastrun:
        X_color, y_color, indices = subsample(X_color, y_color, sub)



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
        X_under_test, y_under_test, indices = subsample(X_under, y_under, bal_n)
        X_under = np.delete(X_under, indices, axis=0)
        y_under = np.delete(y_under, indices, axis=0)
        #running pca
        pca_under, Xstd_under = runPCA(X_under, y_under)
        #finding standardizing values
        Xmean_under, Xerr_under = np.mean(X_under, axis=0), np.std(X_under, axis=0)
        #standardizing sample
        X_under_test = (X_under_test - Xmean_under) / Xerr_under
        #extracting samples
        X_under_sample, y_under_sample, indices = subsample(X_under_test, y_under_test, subsample_n)

        #SMOTE for this artifact
        X_smote, y_smote = balanceData(X, y_arti, ratio=aug_smote_ratio , random_state_val=random_state_val)
        #extracting samples
        X_smote_test, y_smote_test, indices = subsample(X_smote, y_smote, bal_n)
        X_smote = np.delete(X_smote, indices, axis=0)
        y_smote = np.delete(y_smote, indices, axis=0)
        #running pca
        pca_smote, Xstd_smote = runPCA(X_smote, y_smote)
        #finding standardizing values
        Xmean_smote, Xerr_smote = np.mean(X_smote, axis=0), np.std(X_smote, axis=0)
        #standardizing sample
        X_smote_test = (X_smote_test - Xmean_smote) / Xerr_smote
        #extracting samples
        X_smote_sample, y_smote_sample, indices = subsample(X_smote_test, y_smote_test, subsample_n)


        ### Explained variance ###
        #unbalanced
        print('Explained variance')
        plt.plot(np.cumsum(pca_unbal.explained_variance_ratio_))
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        plt.grid()
        plt.gcf()
        if save:
            if file_out:
                my_path = pickle_path+saving+'/other'
            else:
                my_path = os.path.dirname(os.path.abspath(__file__))+saving+'/other'
            my_file = artifact + '_EV_unbal'
            path = os.path.join(my_path, my_file)
            plt.savefig(path, dpi=800)
        if show:
            plt.show()
        plt.clf()
        #Undersampling
        plt.plot(np.cumsum(pca_under.explained_variance_ratio_))
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        plt.grid()
        plt.gcf()
        if save:
            if file_out:
                my_path = pickle_path+saving+'/other'
            else:
                my_path = os.path.dirname(os.path.abspath(__file__))+saving+'/other'
            my_file = artifact + '_EV_under'
            path = os.path.join(my_path, my_file)
            plt.savefig(path, dpi=800)
        if show:
            plt.show()
        plt.clf()
        #SMOTE
        plt.plot(np.cumsum(pca_smote.explained_variance_ratio_))
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        plt.grid()
        plt.gcf()
        if save:
            if file_out:
                my_path = pickle_path+saving+'/other'
            else:
                my_path = os.path.dirname(os.path.abspath(__file__))+saving+'/other'
            my_file = artifact + '_EV_smote'
            path = os.path.join(my_path, my_file)
            plt.savefig(path, dpi=800)
        if show:
            plt.show()
        plt.clf()


        #undersampling
        method = 'under'
        investigateBalancing(pca_under, X_under_test, y_under_test, principals, colors, artifact, method, save, show)


        #undersampling
        method = 'smote'
        investigateBalancing(pca_smote, X_smote_test, y_smote_test, principals, colors, artifact, method, save, show)


        ### Undersampling ###
        print('Undersampling')
        under(X, y_arti, pca_unbal, under_n, colors, artifact, random_state_val, show, save)


        ### SMOTE ###
        print('SMOTE')
        #we want unbalanced data
        investigateSMOTE(X, y_arti, pca_unbal, smote_n, smote_ratio, random_state_val, artifact, show, save, n_components)



        ### NOISE ADDITION white noise###
        # UNDERSAMPLING
        print('Noise addition - white - under')
        #extracting class
        y_white_arti = y_white[:, artifact_pos]
        X_white_arti, y_white_arti = rand_undersample(X_white, y_white_arti, arg='majority', state=random_state_val, multi=False)
        # The function can therefore be called quite easily
        #standardizing
        X_white_arti_under = (X_white_arti - Xmean_under) / Xerr_under
        method = "white_under"
        investigateAugmentation(pca_under, X_under_sample, y_under_sample, X_white_arti_under, y_white_arti, principals, colors, artifact, method, subsample_n, aug_ratio)
        #SMOTE
        print('Noise addition - white - smote')
        #standardizing
        X_white_arti_smote = (X_white_arti - Xmean_smote) / Xerr_smote
        method = "white_smote"
        investigateAugmentation(pca_smote, X_smote_sample, y_smote_sample, X_white_arti_smote, y_white_arti, principals, colors, artifact, method, subsample_n, aug_ratio)




        ### NOISE ADDITION color noise###
        # UNDERSAMPLING
        print('Noise addition - color - under')
        #extracting class
        y_color_arti = y_color[:, artifact_pos]
        X_color_arti, y_color_arti = rand_undersample(X_color, y_color_arti, arg='majority', state=random_state_val, multi=False)
        # The function can therefore be called quite easily
        #standardizing
        X_color_arti_under = (X_color_arti - Xmean_under) / Xerr_under
        method = "color_under"
        investigateAugmentation(pca_under, X_under_sample, y_under_sample, X_color_arti_under, y_color_arti, principals, colors, artifact, method, subsample_n, aug_ratio)
        #SMOTE
        print('Noise addition - color - smote')
        # UNDERSAMPLING
        #standardizing
        X_color_arti_smote = (X_color_arti - Xmean_smote) / Xerr_smote
        method = "color_smote"
        investigateAugmentation(pca_smote, X_smote_sample, y_smote_sample, X_color_arti_smote, y_color_arti, principals, colors, artifact, method, subsample_n, aug_ratio)


        ### MixUP ###
        print('MixUp - under')
        # Onehot-encoding for mixup to work
        y_onehot_encoded = OneHotEncoder(sparse=False).fit_transform(y_under.reshape(len(y_under), 1))
        # Running mixup
        mix_X, mix_y, _ = mixup(X_under, y_onehot_encoded, aug_ratio)
        # Undoing the onehot-encoding
        mix_y = np.argmax(mix_y, axis=1)
        #standardizing
        mix_X = (mix_X - Xmean_under) / Xerr_under
        method = "MixUp_under"
        investigateAugmentation(pca_under, X_under_sample, y_under_sample, mix_X, mix_y, principals, colors, artifact, method, subsample_n, aug_ratio)

        print('MixUp - smote')
        # Onehot-encoding for mixup to work
        y_onehot_encoded = OneHotEncoder(sparse=False).fit_transform(y_smote.reshape(len(y_smote), 1))
        # Running mixup
        mix_X, mix_y, _ = mixup(X_smote, y_onehot_encoded, aug_ratio)
        # Undoing the onehot-encoding
        mix_y = np.argmax(mix_y, axis=1)
        #standardizing
        mix_X = (mix_X - Xmean_smote) / Xerr_smote
        method = "MixUp_smote"
        investigateAugmentation(pca_smote, X_smote_sample, y_smote_sample, mix_X, mix_y, principals, colors, artifact, method, subsample_n, aug_ratio)


        ### GAN ###
        print('GAN - under')
        method = "GAN"
        GAN_epochs = 100
        class_size = int(sum(y_under))  # Sum of all the ones. Since data is balanced, the other class is same size
        # Existing data for class 0 and 1 (Since not yet shuffled)
        class0 = X_under[:class_size]
        class1 = X_under[class_size:]
        # GAN-augmented data, generated from existing data of each class.
        GAN_class0 = GAN(class0, method+"-absent", NtoGenerate=int(aug_ratio * class_size), epochs=GAN_epochs)
        print("GAN class 0 complete")
        GAN_class1 = GAN(class1, method+"present", NtoGenerate=int(aug_ratio * class_size), epochs=GAN_epochs)
        print("GAN class 1 complete")
        X_GAN = np.concatenate((GAN_class0, GAN_class1))
        y_GAN = np.concatenate((np.zeros(int(aug_ratio * class_size)), np.ones(int(aug_ratio * class_size))))
        #standardizing
        X_GAN = (X_GAN - Xmean_under) / Xerr_under
        method = "GAN_under"
        investigateAugmentation(pca_under, X_under_sample, y_under_sample, X_GAN, y_GAN, principals, colors, artifact, method, subsample_n, aug_ratio)

        print('GAN - smote')
        method = "GAN"
        GAN_epochs = 100
        class_size = int(sum(y_smote))  # Sum of all the ones. Since data is balanced, the other class is same size
        # Existing data for class 0 and 1 (Since not yet shuffled)
        class0 = X_smote[:class_size]
        class1 = X_smote[class_size:]
        # GAN-augmented data, generated from existing data of each class.
        GAN_class0 = GAN(class0, method+"-absent", NtoGenerate=int(aug_ratio * class_size), epochs=GAN_epochs)
        print("GAN class 0 complete")
        GAN_class1 = GAN(class1, method+"present", NtoGenerate=int(aug_ratio * class_size), epochs=GAN_epochs)
        print("GAN class 1 complete")
        X_GAN = np.concatenate((GAN_class0, GAN_class1))
        y_GAN = np.concatenate((np.zeros(int(aug_ratio * class_size)), np.ones(int(aug_ratio * class_size))))
        #standardizing
        X_GAN = (X_GAN - Xmean_smote) / Xerr_smote
        method = "GAN_smote"
        investigateAugmentation(pca_smote, X_smote_sample, y_smote_sample, X_GAN, y_GAN, principals, colors, artifact, method, subsample_n, aug_ratio)


        print("Done with: " + artifact)


