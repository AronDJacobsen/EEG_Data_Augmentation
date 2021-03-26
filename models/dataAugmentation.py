

import numpy as np
import torch
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import NearMiss
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing



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
    # remove null from there
    y[transform_indices, classes-1] = np.zeros(len(transform_indices))

    #indices with more than 1 class
    del_indices = np.where(np.sum(y[:,:classes],axis=1) > 1)[0]

    X = np.delete(X, del_indices, axis=0)
    y = np.delete(y, del_indices, axis=0)
    ID_frame = np.delete(ID_frame,del_indices, axis=0)

    return X, y, ID_frame



def smote(X, y, multi):

    #undersample majority
    #balance_b = Counter(y)
    if multi:
        lb = preprocessing.LabelBinarizer()
        y = np.argmax(y, axis=1)


    #oversample minority
    over = SMOTE() # increase minority to have % of majority

    X_over, y_over = over.fit_resample(X, y)
    # how is the balance now?
    #balance_a = Counter(y)

    #print('Before:', balance_b)
    #print('After: ', balance_a)

    if multi:
        y_over = lb.fit_transform(y_over)



    return X_over, y_over



def rand_undersample(X, y, arg):

    lb = preprocessing.LabelBinarizer()
    y = np.argmax(y, axis=1)

    #undersample majority
    #balance_b = Counter(y)
    under = RandomUnderSampler(sampling_strategy=arg) # reduce majority to have % more than minority
    X_under, y_under = under.fit_resample(X, y)
    # how is the balance now?
    #balance_a = Counter(y)

    #print('Before:', balance_b)
    #print('After: ', balance_a)

    y_under = lb.fit_transform(y_under)

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





def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# loss based on mixup
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)







