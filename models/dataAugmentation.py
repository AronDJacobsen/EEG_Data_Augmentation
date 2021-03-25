

import numpy as np
import torch

# maybe seperate smote and downsampling

def smote(X, y, increase):
    pass


def balance(X, y, increase, reduce):
    #### dealing with class implanace ####

    #used:
    #   - https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/

    from collections import Counter
    # what is the class balance?
    balance_b = np.unique(y, return_counts=True)
    #balance_b = Counter(y)

    #if necessary: pip install imbalanced-learn
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline

    #undersample majority
    under = RandomUnderSampler(sampling_strategy=reduce) # reduce majority to have % more than minority
    #oversample minority
    over = SMOTE(sampling_strategy=increase) # increase minority to have % of majority
    #initializing pipeline
    steps = [('u', under), ('o', over)]
    pipeline = Pipeline(steps=steps)

    # transform the dataset
    X, y = pipeline.fit_resample(X, y)

    # how is the balance now?
    #balance_a = Counter(y)
    balance_a = np.unique(y, return_counts=True)


    print('Before:', balance_b)
    print('After: ', balance_a)

    return X, y




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







