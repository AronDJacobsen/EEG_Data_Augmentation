import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from prepData.dataLoader import LoadPickles

from models.models import lr_mixup, nn_mixup, baseline, lr, gnb, knn, rf


def balance(X, y, increase, reduce):
    #### dealing with class implanace ####

    #used:
    #   - https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/

    from collections import Counter
    # what is the class balance?
    balance_b = Counter(y)

    #if necessary: pip install imbalanced-learn
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline

    #oversample minority
    over = SMOTE(sampling_strategy=increase) # increase minority to have % of majority
    #undersample majority
    under = RandomUnderSampler(sampling_strategy=reduce) # reduce majority to have % more than minority

    #initializing pipeline for transformation of dataset
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)
    # transform the dataset
    X, y = pipeline.fit_resample(X, y)

    # how is the balance now?
    balance_a = Counter(y)

    print('Before:', balance_b)
    print('After: ', balance_a)

    return X, y


X, y, ID_frame = LoadPickles(DelNan = True)

individuals = np.unique(ID_frame)

# Defining subjects to use for the train and test set. For now 10 random train_individuals.
# TODO: Vi bør have et valideringssæt til træningen, der også er delt ind efter subjects.
np.random.seed(0)
train_indiv = set(np.random.choice(individuals, 15, replace=False))
test_indiv = set(np.random.choice(np.setdiff1d(individuals, list(train_indiv)), 10, replace=False))

#Splitting train and test set.
train_indices = [i for i, ID in enumerate(ID_frame) if ID in train_indiv]
test_indices = [i for i, ID in enumerate(ID_frame) if ID in test_indiv]

X_train, y_train = X[train_indices,:], y[train_indices]
X_test, y_test = X[test_indices,:], y[test_indices]

# Standardizing data
#TODO: Vil vi standardisere? Er det gjort korrekt nu?
Xmean, Xdev = np.mean(X_train, axis=0), np.std(X_train, axis=0)
X_train = (X_train - Xmean) / Xdev
X_test = (X_test - Xmean) / Xdev


# balancing dataset
# we only want to evaluate eye movement
# only on training data
X_train, y_train = balance(X_train, y_train[:,0], increase = 0.15, reduce = 0.4)
y_test = y_test[:,0]

acc_lr_mixup, f1_lr_mixup = lr_mixup(X_train, y_train, X_test, y_test, augr=0.5, mixup = False, alpha= None,
                                batch_size = 100, lr_rate = 0.05, epochs = 2)


acc_nn_mixup, f1_nn_mixup = nn_mixup(X_train, y_train, X_test, y_test, augr=0.5, mixup = False, alpha= None,
                                batch_size = 256, lr_rate = 0.01, epochs = 5)


models = {'Baseline' : baseline, 'LR' : lr, 'GNB' : gnb,
          'KNN' : knn,  'RF' : rf
          }
data = [X_train, y_train, X_test, y_test]
output = ['accuracy', 'f1_score']




def evaluate(models, data, output):
    initial_data = {}
    for key in models:
        accuracy, f1_score = models[key](data[0],data[1],data[2],data[3])
        initial_data[key] = [accuracy, f1_score]

    df = pd.DataFrame.from_dict(initial_data)
    df.index = output
    #print(df)
    return df


df = evaluate(models, data, output)

df['nn'] = [acc_nn_mixup, f1_nn_mixup]



# Fitting and evaluating a binary logistic regression. Solely predicting presence/absence of eyemovement in a window.
#TODO: Kan man specificere et valideringssæt i LogisticRegression?
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train[:,0]) # Starting with only testing eyemovement.
model.predict(X_test)
model.score(X_test, y_test[:,0])

# Multi-class problem - collapsing one-hot encoding to get 6 categorical labels for the multinomial log. reg.
# Encoding choice can be found in loadPrepData-function in dataLoader-script
#TODO: Der er nok en fejl i bestemmelsen af klassen, hvis der er flere labels på et vindue..
multi_class_labels = np.array([np.where(y_win==1)[0][0] for y_win in y]) #Don't know if this should be used..
y_train_multi, y_test_multi = multi_class_labels[train_indices], multi_class_labels[test_indices]

#Fitting and evaluating multinomial logistic regression
model = LogisticRegression(multi_class='multinomial', solver='lbfgs',max_iter=1000)
model.fit(X_train, y_train_multi) # Trying a multiclass
model.predict(X_test)
model.score(X_test, y_test_multi)


# Now we will ignore 'null'-class completely from training and test data.
# Excluding 'null'-observations from data.
train_indices, test_indices = np.where(y_train_multi!=5)[0], np.where(y_test_multi!=5)[0]
X_train_nonull, y_train_nonull = X_train[train_indices,:], y_train_multi[train_indices]
X_test_nonull, y_test_nonull = X_test[test_indices,:], y_test_multi[test_indices]

# Fitting a 5-class multinomial logistic regression without 'null' as a class.
model = LogisticRegression(multi_class='multinomial', solver='lbfgs',max_iter=1000)
model.fit(X_train_nonull, y_train_nonull) # Trying a multiclass
model.predict(X_test_nonull)
model.score(X_test_nonull, y_test_nonull)




"""
# Not done at all - might be used to divide by electrodes
a = []
for i in range(19):
    a.append(X[0,i*241:(i+1)*241])
"""



""" Kode til at tjekke fordelingen af klasser

#Eyemovement
print(np.unique(y_train[:,0], return_counts=True))

#Chew
print(np.unique(y_train[:,1], return_counts=True))

#Shiver
print(np.unique(y_train[:,2], return_counts=True))

#Elpp
print(np.unique(y_train[:,3], return_counts=True))

#Musc
print(np.unique(y_train[:,4], return_counts=True))

#Null
print(np.unique(y_train[:,5], return_counts=True))
"""
