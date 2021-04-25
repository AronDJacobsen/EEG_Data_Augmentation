import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from bioinfokit.visuz import cluster
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from prepData.dataLoader import LoadNumpyPickles
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
from models.balanceData import *
import plotly.express as px
from sklearn.preprocessing import OneHotEncoder

from models.GAN import GAN
from models.mixup import mixup

import pandas as pd


def balancedData(X, y, ID_frame):

    labels = ['eyem','chew','shiv','elpp','musc','null']
    count_labels = [np.unique(y[:,i],return_counts=True)[1][1] for i in range(len(labels))]
    ratio_labels = np.array(count_labels) / len(y)


    # Fining out how many of the files have multiple labels and how many labels these files have.
    multiple_labels = [np.sum(y[i,:]) for i in range(len(y))]
    multiple_labels_count = np.unique(multiple_labels, return_counts=True)

    # Initializing dictionaries for finding where we have multiple labels
    indices_2 = []
    mult_dict_2 = {'eyem':0, 'chew':0 ,'shiv':0, 'elpp':0, 'musc':0,'null':0}
    indices_3 = []
    mult_dict_3 = {'eyem':0, 'chew':0 ,'shiv':0, 'elpp':0, 'musc':0,'null':0}

    for i, y_single in enumerate(y):
        pos = np.where(y_single == 1)[0]
        if len(pos) == 2:
            indices_2.append(i)
            for pos_idx in pos:
                mult_dict_2[labels[pos_idx]] += 1
        if len(pos) > 2:
            indices_3.append(i)
            for pos_idx in pos:
                mult_dict_3[labels[pos_idx]] += 1

    print("\n{:d} windows contain two different labels. These are distributed as follows: ".format(len(indices_2)) + str(mult_dict_2))

    print("\n{:d} windows contain three different labels. These are distributed as follows: ".format(len(indices_3)) + str(mult_dict_3))
    print("The labelling of the files holding three labels are seen below: \n" + str(y[indices_3]))


    # colors = "lightcoral" or "lightsteelblue" or lightslategrey"
    plt.bar(np.arange(len(labels)), ratio_labels, color="lightsteelblue", tick_label=labels)
    plt.show()
    print("Ratio of labels, following " + str(labels) + ": \t" + str(np.round(ratio_labels*100, 3)))



def runPCA(X, y, n_components):
    #Xmean, Xerr = np.mean(X, axis=0), np.std(X, axis=0)
    #Xstd = (X - Xmean) / Xerr

    scaler = StandardScaler()
    scaler.fit(X)
    Xstd = scaler.transform(X)

    pca = PCA(n_components=n_components)

    print("\nRunning PCA...")
    pca_out = pca.fit(Xstd)
    loadings = pca_out.components_

    print("PCA done!")
    return pca, scaler


def plotPCABig(pca_scores, pca, aug_label_list):

    # Code taken from https://plotly.com/python/pca-visualization/
    labels = {
        str(i): f"PC {i + 1} ({var:.1f}%)"
        for i, var in enumerate(pca.explained_variance_ratio_ * 100)
    }

    fig = px.scatter_matrix(
        pca_scores,
        labels=labels,
        dimensions=range(4),
        color=aug_label_list
    )
    fig.update_traces(diagonal_visible=False)
    fig.show()

def plotPCAScatter3D(pca_scores, pca, aug_label_list, colors, artifact, method):
    y_text = []
    for number in aug_label_list:
        if number == 1:
            y_text.append("Present")
        elif number == 0:
            y_text.append("Absent")

    if y_text != []:
        aug_label_list = y_text

    distinct = np.unique(aug_label_list)

    fig = plt.figure(1, figsize=(8, 6))
    ax = Axes3D(fig, elev=-150, azim=110)

    label_size = Counter(aug_label_list)
    major = max(label_size, key=label_size.get)


    for i, name in enumerate(distinct):
        indices = [j for j, inner_name in enumerate(aug_label_list) if inner_name == name]

        transparency = 1
        if name == major:
            transparency = 0.3
        ax.scatter(pca_scores[:, 0][indices], pca_scores[:, 1][indices], pca_scores[:, 2][indices],
                   c=colors[i], s=50, label=aug_label_list[i], alpha=transparency)

    # ax.scatter(pca_scores[:, 0], pca_scores[:, 1], pca_scores[:, 2], c=target,
    #          cmap=plt.cm.Set1, edgecolor='k', s=40, label=labels)
    ax.set_xlabel("1st eigenvector")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd eigenvector")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd eigenvector")
    ax.w_zaxis.set_ticklabels([])

    ax.set_title("Data from the class {:s} - {:s}".format(artifact, method))
    plt.legend()
    plt.show()

    #fig = px.scatter(pca_scores, x=0, y=1, color=aug_label_list)
    #fig.show()

def plotPCAScatter(pca_scores, pca, aug_label_list, colors, artifact, method):
    y_text = []
    for number in aug_label_list:
        if number == 1:
            y_text.append("Present")
        elif number == 0:
            y_text.append("Absent")

    if y_text != []:
        aug_label_list = y_text

    distinct = np.unique(aug_label_list)

    label_size = Counter(aug_label_list)
    major = max(label_size, key=label_size.get)


    for i, name in enumerate(distinct):
        indices = [j for j, inner_name in enumerate(aug_label_list) if inner_name == name]
        transparency = 1

        if name == major:
            transparency = 0.3
        plt.scatter(pca_scores[:, 0][indices], pca_scores[:, 1][indices], s=50, c=colors[i], label=distinct[i], alpha=transparency)


    plt.xlabel('1st eigenvector')
    plt.ylabel('2nd eigenvector')
    plt.title("Data from the class {:s} - {:s}".format(artifact, method))
    plt.legend(loc=1)

    #plt.show()

def investigateSMOTE(X, y, ID_frame, ratio, random_state_val):
    n_components = 3
    pca, scaler = runPCA(X, y, n_components=n_components)
    Xstd = scaler.transform(X)
    y_std = y

    X, y = smote(X, y, multi = False, state = random_state_val)
    X = scaler.transform(X)

    return X, y, Xstd, y_std, pca


def investigateAugmentation(X, y, X_aug, y_aug, colors, artifact, method, view_classes=False):
    n_components = 3
    pca, scaler = runPCA(X, y, n_components=n_components)
    Xstd = scaler.transform(X)
    y_std = y

    X_with_aug = np.concatenate((X_under, X_aug))

    if view_classes:
        y_labels_class = np.concatenate((y, y_aug))

    y_labels = np.concatenate((["Original"] * len(y), ["Augmented"] * len(y_aug)))

    X_aug_pca = scaler.transform(X_with_aug)
    pca_scores_aug = pca.fit_transform(X_aug_pca)


    fig = plt.figure(figsize=(10, 12))
    ax1 = plt.subplot(2, 1, 1)
    plotPCAScatter(pca.fit_transform(Xstd), pca, ["Original"]*len(y_std), colors, artifact, method)
    ax2 = plt.subplot(2, 1, 2, sharex=ax1, sharey=ax1)
    plotPCAScatter(pca_scores_aug, pca, y_labels, colors, artifact, method)
    fig.show()

    if view_classes:
        fig = plt.figure(figsize=(10, 12))
        ax1 = plt.subplot(2, 1, 1)
        plotPCAScatter(pca_scores_aug, pca, y_labels, colors, artifact, method)
        ax2 = plt.subplot(2, 1, 2, sharex=ax1, sharey=ax1)
        plotPCAScatter(pca_scores_aug, pca, y_labels_class, colors, artifact, method)
        fig.show()


def investigateGAN(X, y, ID_frame):

    pass

def investigateMixUp(X, y, X_mixup, y_mixup, colors, artifact, method):
    n_components = 3
    pca, scaler = runPCA(X, y, n_components=n_components)
    Xstd = scaler.transform(X)
    y_std = y

    X_with_noise = np.concatenate((X_under, noise_X))
    y_labels = np.concatenate((["Original"] * len(y), ["Augmented"] * len(y_noise)))

    X_noise_pca = scaler.transform(X_with_noise)
    pca_scores_Noise = pca.fit_transform(X_noise_pca)

    fig = plt.figure(figsize=(10, 12))
    ax1 = plt.subplot(2, 1, 1)
    plotPCAScatter(pca.fit_transform(Xstd), pca, ["Original"] * len(y_std), colors, artifact, method)
    ax2 = plt.subplot(2, 1, 2, sharex=ax1, sharey=ax1)
    plotPCAScatter(pca_scores_Noise, pca, y_labels, colors[::-1], artifact, method)
    fig.show()



def savePickleGAN():
    pass

def savePickleMixUp():
    pass



if __name__ == '__main__':

    random_state_val = 0
    windowsOS = True
    pickle_path = r"C:\Users\Albert Kjøller\Documents\GitHub\EEG_epilepsia"

    #Pick artifact for investigation
    artifact = 'eyem'
    colors = ["lightsteelblue", "lightcoral"]


    # loading data - define which pickles to load (with NaNs or without)
    X_file = r"\X_clean.npy"  # X_file = r"\X.npy"
    y_file = r"\y_clean.npy"  # y_file = r"\y_clean.npy"
    ID_file = r"\ID_frame_clean.npy"  # ID_file = r"\ID_frame.npy"

    X = LoadNumpyPickles(pickle_path=pickle_path, file_name=X_file, windowsOS=windowsOS)
    y = LoadNumpyPickles(pickle_path=pickle_path, file_name=y_file, windowsOS=windowsOS)
    ID_frame = LoadNumpyPickles(pickle_path=pickle_path, file_name=ID_file, windowsOS=windowsOS)


    X, y, ID_frame = binary(X, y, ID_frame)

    artifact_names = ['eyem', 'chew', 'shiv', 'elpp', 'musc', 'null']
    artifact_pos = [i for i, name in enumerate(artifact_names) if name==artifact][0]

    y = y[:, artifact_pos]

    np.random.seed(20)
    ratio = 0  # Currently not doing anything as we do not want to visualize the downsampling, but only the SMOTING.
    subsample_n = 5000
    indices = np.random.choice(len(X), subsample_n)

    X_under, y_under = rand_undersample(X[indices, :], y[indices], arg='majority', state=random_state_val, multi=False)


    # Create PCA basis (from undersampled data) and plot this
    n_components = 4

    pca, scaler = runPCA(X_under, y_under, n_components=n_components)
    Xstd = scaler.transform(X_under)
    pca_scores = pca.fit_transform(Xstd)
    aug_label_list = ["Original"] * Xstd.shape[0]

    method = "Undersampling"
    #plotPCABig(pca_scores, pca, aug_label_list)
    plotPCAScatter(pca_scores, pca, aug_label_list, colors, artifact, method)
    plt.show()
    plotPCAScatter3D(pca_scores, pca, aug_label_list, colors, artifact, method)



    # SMOTE - different function than the rest because it concatenates dataframe in a differetn way
    X_SMOTE, y_SMOTE, Xstd, y_std, pca_SMOTE = investigateSMOTE(X[indices[:1000],:], y[indices[:1000]], ID_frame, ratio, random_state_val)

    method1 = "Without SMOTE"
    pca_scores_unbalanced = pca_SMOTE.fit_transform(Xstd)
    plotPCAScatter(pca_scores_unbalanced, pca_SMOTE, y_std, colors, artifact, method1)
    plt.show()
    plotPCAScatter3D(pca_scores_unbalanced, pca_SMOTE, y_std, colors, artifact, method1)

    method2 = "With SMOTE"
    pca_scores_SMOTE = pca_SMOTE.fit_transform(X_SMOTE)
    plotPCAScatter(pca_scores_SMOTE, pca_SMOTE, y_SMOTE, colors, artifact, method2)
    plt.show()
    plotPCAScatter3D(pca_scores_SMOTE, pca_SMOTE, y_SMOTE, colors, artifact, method2)

    fig = plt.figure(figsize=(10,12))
    ax1 = plt.subplot(2, 1, 1)
    plotPCAScatter(pca_scores_unbalanced, pca_SMOTE, y_std, colors, artifact, method1)
    ax2 = plt.subplot(2, 1, 2, sharex=ax1, sharey=ax1)
    plotPCAScatter(pca_scores_SMOTE, pca_SMOTE, y_SMOTE, colors, artifact, method2)
    fig.show()






    # Noise addition - most of this code is just taken from the pipeline...
    pickle_path_aug = pickle_path + r"\augmentation_pickles"

    experiment = 'DataAug_white_noiseAdd_LR'  # 'DataAug_color_noiseAdd_LR'
    experiment_name = "_DataAug_white_Noise"  # "_DataAug_color_Noise" added to saving files
    noise_experiment = r"\colornoise30Hz_covarOne"

    X_noise = LoadNumpyPickles(pickle_path_aug + noise_experiment, file_name=X_file, windowsOS=windowsOS)
    y_noise = LoadNumpyPickles(pickle_path_aug + noise_experiment, file_name=y_file, windowsOS=windowsOS)

    ratio = 1

    X_noise_new = X_noise[indices, :]
    y_noise_new = y_noise[indices, :]

    # Balance noisy data
    label_size = Counter(y_noise_new[:, artifact_pos])
    major = max(label_size, key=label_size.get)
    decrease = label_size[1 - major]
    label_size[major] = int(np.round(decrease, decimals=0))
    X_noise_new, y_noise_new = rand_undersample(X_noise_new, y_noise_new[:, artifact_pos], arg=label_size,
                                                state=random_state_val, multi=False)

    # Find new points
    N_noise = X_noise_new.shape[0]
    N_clean = X_under.shape[0]
    n_new_points = int(ratio * N_clean)
    noise_idxs = np.random.choice(N_noise, n_new_points)

    # Select noisy data
    noise_X = X_noise_new[noise_idxs, :]
    noise_y = y_noise_new[noise_idxs]

    # Concatenate
    X_with_noise = np.concatenate((X_under, noise_X))
    y_with_noise = np.concatenate((y_under, noise_y))

    # Function should take the original data and the concatenated dataframe with noisy files.
    # The function can therefore be called quite easily
    method = "Noise addition - " + noise_experiment
    investigateAugmentation(X_under, y_under, noise_X, noise_y, colors, artifact, method, view_classes=True)
    investigateAugmentation(X_under, y_under, noise_X, noise_y, colors, artifact, method)





    # MixUP
    # Onehot-encoding for mixup to work
    y_onehot_encoded = OneHotEncoder(sparse=False).fit_transform(y_under.reshape(len(y_under), 1))

    # Running mixup
    mix_X, mix_y, _ = mixup(X_under, y_onehot_encoded, ratio)

    # Undoing the onehot-encoding
    mix_y = np.argmax(mix_y, axis=1)

    X_with_mix = np.concatenate((X_under, mix_X))
    y_with_mix = np.concatenate((y_under, mix_y))

    method = "MixUp"
    investigateAugmentation(X_under, y_under, mix_X, mix_y, colors, artifact, method, view_classes=True)
    investigateAugmentation(X_under, y_under, mix_X, mix_y, colors, artifact, method)






    # GAN
    GAN_epochs = 100
    class_size = int(sum(y_under))  # Sum of all the ones. Since data is balanced, the other class is same size

    # Existing data for class 0 and 1 (Since not yet shuffled)
    class0 = X_under[:class_size]
    class1 = X_under[class_size:]

    # GAN-augmented data, generated from existing data of each class.
    GAN_class0 = GAN(class0, NtoGenerate=int(ratio * class_size), epochs=GAN_epochs)
    print("GAN class 0 complete")
    GAN_class1 = GAN(class1, NtoGenerate=int(ratio * class_size), epochs=GAN_epochs)
    print("GAN class 1 complete")

    X_GAN = np.concatenate((GAN_class0, GAN_class1))
    y_GAN = np.concatenate((np.zeros(int(ratio * class_size)), np.ones(int(ratio * class_size))))

    X_with_GAN = np.concatenate((X_under, GAN_class0, GAN_class1))
    y_with_GAN = np.concatenate((y_under, np.zeros(int(ratio * class_size)), np.ones(int(ratio * class_size))))

    method = "GAN"
    investigateAugmentation(X_under, y_under, X_GAN, y_GAN, colors, artifact, method, view_classes=True)
    investigateAugmentation(X_under, y_under, X_GAN, y_GAN, colors, artifact, method)

    print("")


