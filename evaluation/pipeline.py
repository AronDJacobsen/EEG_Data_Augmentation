


#TODO: Cross-validation

from prepData.dataLoader import LoadPickles
from models.modelFitting_vol2 import *
from sklearn.model_selection import cross_val_score
from models.models import *

#loading data
X, y, ID_frame = LoadPickles(DelNan = True)
#finding individuals
individuals = np.unique(ID_frame)




spaceknn = {'n_neighbors': hp.choice('n_neighbors', range(1,100))}

modeldict = {'KNN' : (knnf, spaceknn)}


#setting fold
KFold(n_splits=5, random_state=None, shuffle=True)
#looping CV folds
for train_index, test_index in kf.split(individuals):
    #their IDs
    trainindv = individuals[train_index]
    testindv = individuals[test_index]

    #their indexes in train and test
    train_indices = [i for i, ID in enumerate(ID_frame) if ID in trainindv]
    test_indices = [i for i, ID in enumerate(ID_frame) if ID in testindv]

    X_train, y_train = X[train_indices,:], y[train_indices]
    X_test, y_test = X[test_indices,:], y[test_indices]

    #loop through models
    for key in modeldict:
        #https://medium.com/district-data-labs/parameter-tuning-with-hyperopt-faa86acdfdce
        #https://towardsdatascience.com/hyperparameter-optimization-in-python-part-2-hyperopt-5f661db91324

        name, space = modeldict[key]
        f = models(X_train, y_train, X_test, y_test).name

        trials = Trials()
        best = fmin(f, space, algo=tpe.suggest, max_evals=100, trials=trials)

        xs = [t['misc']['vals']['n'] for t in trials.trials]
        ys = [-t['result']['loss'] for t in trials.trials]
        stop = 0


#TODO: Error bars and plots




