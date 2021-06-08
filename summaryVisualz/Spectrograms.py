

import seaborn as sns; sns.set_theme()

from sklearn.model_selection import cross_val_score, KFold
from prepData.dataLoader import *
from models.balanceData import *

import matplotlib.pyplot as plt
from matplotlib import rcParams
from collections import defaultdict

from summaryVisualz.GAN import useGAN
from models.mixup import useMixUp
from models.noiseAddition import useNoiseAddition, prepareNoiseAddition
from sklearn import preprocessing

def plotSpectro(dict):
    # Reshaping Data
    # plotting as heatmap:
    #plt.figure(dpi=1000)

    fig, axs = plt.subplots(5,6, sharex=True, sharey=True, constrained_layout = True)
    fig.text(0.45, 0.1, 'Freq.', ha='center', va='center')
    fig.text(0.87, 0.25, 'Channel', ha='center', va='center', rotation='vertical')
    fontsz = 13

    plt.subplots_adjust(left = 0.1, right = 0.8)

    cbar_ax = fig.add_axes([.89, .3, .03, .4])
    for AugNumber, AugMethod in enumerate(['NoAug', 'GAN', 'MixUp', 'white_noise', 'color_noise']):
        for artifactNumber, artifact in enumerate(['eyem', 'chew', 'shiv', 'elpp', 'musc', 'null']):
            spectro = sns.heatmap(dict[AugMethod][artifact], ax=axs[AugNumber, artifactNumber], vmin=-1, vmax=1,
                               cmap="viridis", cbar_ax=cbar_ax)#, xticklabels=['0', '5', '10', '15', '20', '24'])
            spectro.set_aspect('equal', 'box')
            axs[AugNumber, artifactNumber].set_xticks(np.array([3, 10, 17, 24]))
            axs[AugNumber, artifactNumber].set_xticklabels(np.array([3, 10, 17, 24]))#, rotation = -45)
            axs[AugNumber, artifactNumber].set_yticks(np.array([3, 8, 13, 18]))
            axs[AugNumber, artifactNumber].set_yticklabels(np.array([4, 9, 14, 19]), rotation = 0)  # , rotation = -45)
            axs[AugNumber, 0].tick_params(left=False, labelleft=False)
            axs[AugNumber, -1].tick_params(right=True, labelright=True)
            axs[0, artifactNumber].tick_params(top=False, labeltop=False)
            axs[-1, artifactNumber].tick_params(bottom=True, labelbottom=True)

            if artifactNumber == 0 and AugNumber == 0:
                spectro.set_ylabel(AugMethod, fontsize=fontsz-1)
                spectro.set_title(artifact, fontsize=fontsz)

            elif artifactNumber == 0:
                spectro.set_ylabel(AugMethod.split("_")[0], fontsize=fontsz-1)

            elif AugNumber == 0:
                spectro.set_title(artifact, fontsize=fontsz)

    plt.subplots_adjust(hspace = -0.5, wspace = 0.10)

    plt.savefig('spectro.png',bbox_inches='tight')
    plt.show()

#plotSpectro(newData)


def MeanStandardize(dictionary):

    Standardized = defaultdict(lambda: defaultdict(dict))

    for artifact in ['eyem', 'chew', 'shiv', 'elpp', 'musc', 'null']:
        for AugMethod in ['NoAug', 'GAN', 'MixUp', 'white_noise', 'color_noise']:
            # mean spectrograms: Calculated as mean spectrogram in fold, then mean over folds
            Standardized[AugMethod][artifact] = np.mean([np.reshape(np.mean(dictionary[AugMethod][artifact][fold], axis = 0).tolist(), (19,25)) for fold in range(1,6)], axis = 0)

    minval = [np.min([np.min(Standardized[AugMethod][artifact]) for AugMethod in ['NoAug', 'GAN', 'MixUp', 'white_noise', 'color_noise']]) for artifact in ['eyem', 'chew', 'shiv', 'elpp', 'musc', 'null']]
    maxval = [np.max([np.max(Standardized[AugMethod][artifact]) for AugMethod in ['NoAug', 'GAN', 'MixUp', 'white_noise', 'color_noise']]) for artifact in ['eyem', 'chew', 'shiv', 'elpp', 'musc', 'null']]

    for artifactNumber, artifact in enumerate(['eyem', 'chew', 'shiv', 'elpp', 'musc', 'null']):
        for AugMethod in ['NoAug', 'GAN', 'MixUp', 'white_noise', 'color_noise']:
            # mean spectrograms: Calculated as mean spectrogram in fold, then mean over folds
            Standardized[AugMethod][artifact] = -1 + ((Standardized[AugMethod][artifact] - minval[artifactNumber])*(1 - (-1))) / (maxval[artifactNumber] - minval[artifactNumber])

    return Standardized






if __name__ == '__main__':

    GAN_epochs = 5
    Smote = True

    # Dataload data
    pickle_path = r'/Users/philliphoejbjerg/Documents/GitHub/EEG_epilepsia/'
    X = LoadNumpyPickles(pickle_path, r'X_clean.npy', True)
    y = LoadNumpyPickles(pickle_path, r'y_clean.npy', True)
    ID_frame = LoadNumpyPickles(pickle_path, r'ID_frame_clean.npy', True)

    # DataLoad noisy data
    pickle_path_aug = pickle_path + r"augmentation_pickles"
    X_color, y_color, ID_frame_color = prepareNoiseAddition(pickle_path_aug, '/colornoise30Hz_covarOne/', r'X_clean.npy', r'y_clean.npy',
                                                            r'ID_frame_clean.npy', windowsOS=True)
    X_white, y_white, ID_frame_white = prepareNoiseAddition(pickle_path_aug, '/whitenoise_covarOne/', r'X_clean.npy', r'y_clean.npy',
                                                            r'ID_frame_clean.npy', windowsOS=True)



    # apply the inclusion principle
    X, y, ID_frame = binary(X, y, ID_frame)
    X_color, y_color, ID_frame_color = binary(X_color, y_color, ID_frame_color)
    X_white, y_white, ID_frame_white = binary(X_white, y_white, ID_frame_white)


    # The KFold will be splitted by #TODO: forskel mellem ID_frame noise/alm?
    individuals = np.unique(ID_frame)

    random_state_val = 0
    K = 5
    ratio = 1
    aug_ratio = 1

    #### define classes ####

    artifact_names = ['eyem', 'chew', 'shiv', 'elpp', 'musc', 'null']
    classes = len(artifact_names)

    # setting fold details
    kf = KFold(n_splits=K, random_state=random_state_val,
               shuffle=True)  # random state + shuffle ensured same repeated experiments

    MeanData = defaultdict(lambda: defaultdict(dict))

    fold = 0
    for train_idx, test_idx in kf.split(individuals):
        fold += 1
        # single loop
        # while i < 1:
        #    trainindv, testindv = train_test_split(individuals, test_size=0.20, random_state=random_state_val, shuffle = True)
        #   REMEMBER to # the other below


        #### initializing data ####
        # their IDs
        trainindv = individuals[train_idx]
        #testindv = individuals[test_idx]
        # their indexes in train and test
        train_indices = [i for i, ID in enumerate(ID_frame) if ID in trainindv]
        #test_indices = [i for i, ID in enumerate(ID_frame) if ID in testindv]
        # testID_list = [ID for i, ID in enumerate(ID_frame) if ID in testindv]
        X_train, y_train = X[train_indices, :], y[
            train_indices]  # we keep the original and balance new later
        #X_test, y_test = X[test_indices, :], y[test_indices]

        # Splitting for trainset
        X_color_train = X_color[train_indices, :]
        y_color_train = y_color[train_indices, :]
        X_white_train = X_white[train_indices, :]
        y_white_train = y_white[train_indices, :]

        #### for each artifact ####
        for artifact, artifact_name in enumerate(artifact_names):
            #### initializing data ####
            # only include the artifact of interest
            # new name for the ones with current artifact
            Xtrain = X_train  # only to keep things similar
            #Xtest = X_test  # only to keep things similar
            ytrain = y_train[:, artifact]
            #ytest = y_test[:, artifact]

            ##################################
            # on small runs
            # ytrain[5:8] = 1
            # ytest[5:8] = 1
            ##################################

            #### balancing data ####
            # now resample majority down to minority to achieve equal
            # - new name in order to not interfere with hyperopt

            if Smote == False:  # TODO: Hvis kun undersampling
                Xtrain_new, ytrain_new = rand_undersample(Xtrain, ytrain, arg='majority',
                                                          state=random_state_val, multi=False)
            else: # TODO: Hvis SMOTE + undersampling
                # Using mix of undersampling and smote
                Xtrain_new, ytrain_new = balanceData(Xtrain, ytrain, ratio = 2, random_state_val=random_state_val)


            #plot Normal
            NoAug = Xtrain_new[np.where(ytrain_new == 1)]
            MeanData['NoAug'][artifact_name][fold] = NoAug

            #plotSpectro(NoAug, min_val, max_val, artifact_names[artifact], AugMethod = "None")


            #plot GAN
            GAN_X, GAN_y = useGAN(Xtrain_new, ytrain_new, aug_ratio, GAN_epochs = GAN_epochs, experiment_name = "Spectrograms")
            GAN_y = GAN_y[len(Xtrain_new):]
            GAN_X = GAN_X[len(Xtrain_new):][np.where(GAN_y == 1)]
            MeanData['GAN'][artifact_name][fold] = GAN_X
            #plotSpectro(GAN_X, min_val, max_val, artifact_names[artifact], AugMethod = "GAN")

            #plot MixUp
            Mix_X, Mix_y = useMixUp(Xtrain_new, ytrain_new, aug_ratio)
            Mix_y = Mix_y[len(Xtrain_new):]
            Mix_X = Mix_X[len(Xtrain_new):][np.where(Mix_y == 1)]
            MeanData['MixUp'][artifact_name][fold] = Mix_X
            #plotSpectro(Mix_X, min_val, max_val, artifact_names[artifact], AugMethod = "MixUp")

            # plot Colored Noise
            y_color_artifact = y_color_train[:, artifact]
            X_color_artifact, y_color_artifact = useNoiseAddition(X_color_train, y_color_artifact, Xtrain_new,
                                                                  ytrain_new, aug_ratio, random_state_val)

            y_color_artifact = y_color_artifact[len(Xtrain_new):]
            X_color_artifact = X_color_artifact[len(Xtrain_new):][np.where(y_color_artifact == 1)]
            MeanData['color_noise'][artifact_name][fold] = X_color_artifact
            #plotSpectro(X_color_artifact, min_val, max_val, artifact_names[artifact], AugMethod = "colored noise")

            # plot White Noise
            y_white_artifact = y_white_train[:, artifact]
            X_white_artifact, y_white_artifact = useNoiseAddition(X_white_train, y_white_artifact, Xtrain_new,
                                                                      ytrain_new, aug_ratio, random_state_val)

            y_white_artifact = y_white_artifact[len(Xtrain_new):]
            X_white_artifact = X_white_artifact[len(Xtrain_new):][np.where(y_white_artifact == 1)]

            MeanData['white_noise'][artifact_name][fold] = X_white_artifact
            #plotSpectro(X_white_artifact, min_val, max_val, artifact = artifact_names[artifact], AugMethod = "white noise")

    newData = MeanStandardize(MeanData)
    plotSpectro(newData)
