import numpy as np
import random
import torch
from sklearn.preprocessing import OneHotEncoder

#TODO: Link til hvor vi har fundet koden!


def mixup(X, y, ratio, alpha=0.4):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    size = X.shape[0]
    mixup_size = round(size * ratio)

    # List of lambda-values
    lam = np.random.beta(alpha, alpha, mixup_size)

    idxs = list(range(0, size))

    list1_idx = random.choices(idxs, k = mixup_size)
    list2_idx = random.choices(idxs, k = mixup_size)

    mixed_X = np.multiply(lam,X[list1_idx, :].T).T + np.multiply( (1-lam) ,X[list2_idx, :].T).T
    mixed_y = np.multiply(lam,y[list1_idx, :].T).T + np.multiply( (1-lam) ,y[list2_idx, :].T).T

    # Create matrix of zeros:
    labels = np.zeros(mixed_y.shape)

    for i in range(mixup_size):
        # Two mixup-ed idxs
        ids = np.argsort(mixed_y[i, :])[-2:]
        # Pick one at random by probability, and set = 1
        labels[i, np.random.choice(ids, p=[mixed_y[i, ids[0]], mixed_y[i, ids[1]]])] = 1



    # Remember to concatenate outside of function
    return mixed_X, labels, lam



# loss based on mixup
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

