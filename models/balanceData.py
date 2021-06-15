

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn import preprocessing

from imblearn.pipeline import Pipeline
from imblearn.under_sampling import NearMiss
from sklearn.preprocessing import LabelEncoder
from collections import Counter




def subset(X, y, ID_frame, no_indiv):

    individuals = np.unique(ID_frame)

    new_indiv = set(np.random.choice(individuals, no_indiv, replace=False))
    #Splitting train and test set.
    indices = [i for i, ID in enumerate(ID_frame) if ID in new_indiv]

    new_X, new_y, new_ID_frame = X[indices,:], y[indices], ID_frame[indices]

    return new_X, new_y, new_ID_frame


def binary(X, y, ID_frame):

    classes = len(y[0,:])

    # where only one of the classes are present (except null)
    transform_indices = np.where(np.sum(y[:,:classes-1],axis=1) == 1)[0]

    # we one is present, we set is to 0 in the null class
    y[transform_indices, classes-1] = np.zeros(len(transform_indices))

    # we now only include where one class is present
    include = np.where(np.sum(y[:,:classes],axis=1) == 1)[0]
    y = y[include, :]
    X = X[include, :]
    ID_frame = ID_frame[include]

    '''
    #indices with more than 1 class
    del_indices = np.where(np.sum(y[:,:classes],axis=1) > 1)[0]

    X = np.delete(X, del_indices, axis=0)
    y = np.delete(y, del_indices, axis=0)
    ID_frame = np.delete(ID_frame,del_indices, axis=0)
    '''

    return X, y, ID_frame



def smote(X, y, multi, state, k_neighbors=None):

    #undersample majority
    #balance_b = Counter(y)
    if multi:
        lb = preprocessing.LabelBinarizer()
        y = np.argmax(y, axis=1)


    #oversample minority
    if k_neighbors != None:
        over = SMOTE(random_state = state, k_neighbors=k_neighbors) # increase minority to have % of majority
    else:
        over = SMOTE(random_state = state) # increase minority to have % of majority

    X_over, y_over = over.fit_resample(X, y)
    # how is the balance now?
    #balance_a = Counter(y)

    #print('Before:', balance_b)
    #print('After: ', balance_a)

    if multi:
        y_over = lb.fit_transform(y_over)



    return X_over, y_over



def rand_undersample(X, y, arg, state, multi):

    if multi:
        lb = preprocessing.LabelBinarizer()
        y = np.argmax(y, axis=1)
        under = RandomUnderSampler(sampling_strategy=arg, random_state = state)
        X_under, y_under = under.fit_resample(X, y)
        y_under = lb.fit_transform(y_under)
    else:
        #undersample majority
        #balance_b = Counter(y) # for binary
        # https://machinelearningmastery.com/random-oversampling-and-undersampling-for-imbalanced-classification/
        under = RandomUnderSampler(sampling_strategy=arg, random_state = state)
        X_under, y_under = under.fit_resample(X, y)
        # how is the balance now?

        #balance_a = Counter(y)
        #print('Before:', balance_b)
        #print('After: ', balance_a)


    return X_under, y_under


def nearmiss(X, y, version, n_neighbors):

    #undersample majority
    #balance_b = Counter(y)
    under = RandomUnderSampler(sampling_strategy=reduce) # reduce majority to have % more than minority
    X_under, y_under = under.fit_resample(X, y)
    # how is the balance now?
    #balance_a = Counter(y)

    #print('Before:', balance_b)
    #print('After: ', balance_a)


    return X_under, y_under


def balanceData(Xtrain, ytrain, ratio, random_state_val):
    # for class 0 (artifact) we downsample to no. obs. for artifact * ratio(to smote up)
    label_size = Counter(ytrain)
    major = max(label_size, key=label_size.get)
    decrease = label_size[1 - major] * ratio
    label_size[major] = int(np.round(decrease, decimals=0))
    Xtrain_new, ytrain_new = rand_undersample(Xtrain, ytrain, arg=label_size,
                                              state=random_state_val, multi=False)
    Xtrain_new, ytrain_new = smote(Xtrain_new, ytrain_new, multi=False, state=random_state_val)

    return Xtrain_new, ytrain_new