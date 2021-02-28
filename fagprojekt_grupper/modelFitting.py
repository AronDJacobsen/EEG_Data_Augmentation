import os, glob, torch, time, random
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from fagprojekt_grupper.dataLoader import processRawData, loadPrepData, createSubjectDict

# Ensuring correct path
os.chdir(os.getcwd())

# What is your execute path? #
save_dir = r"/Users/philliphoejbjerg/NovelEEG" + "/"  # /Users/philliphoejbjerg/NovelEEG # "/Users/AlbertoK/Desktop/EEG/subset" + "/"  # ~~~ What is your execute path? # /Users/philliphoejbjerg/NovelEEG

#Directing to correct directories
TUAR_dir = r"data_TUH_EEG/TUH_EEG_CORPUS/artifact_dataset" + "/"  # \**\01_tcp_ar #\100\00010023\s002_2013_02_21
jsonDir = r"tmp.json"
prep_dir = r"tempData" + "/"

jsonDataDir = save_dir + jsonDir
TUAR_dirDir = save_dir + TUAR_dir
prep_dirDir = save_dir + prep_dir



# Creating directory for subjects, sessions, windows for easy extraction of tests in loadPrepData
subjects = createSubjectDict(prep_dirDir)

# X = Number of windows, 19*241    y = Number of windows, 6 categories     ID_frame = subjectID for each window
X, y, ID_frame = loadPrepData(subjects, prep_dirDir)
individuals = np.unique(ID_frame)

# Defining subjects to use for the train and test set. For now 10 random train_individuals.
# TODO: Vi bør have et valideringssæt til træningen, der også er delt ind efter subjects.
np.random.seed(0)
train_indiv = set(np.random.choice(individuals, 10, replace=False))
test_indiv = set(np.setdiff1d(individuals, list(train_indiv)))

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
