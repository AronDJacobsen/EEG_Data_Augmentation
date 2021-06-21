# 02466 (DTU - AI and Data): Project work on TUH EEG Artifact classification 
Created by Albert Kjøller Jacobsen, Aron Djurhuus Jacobsen & Phillip Chavarria Højbjerg

This project investigates the role of data augmentation on the Temple University Hospital EEG Artifact corpus.

The outcome of the project is a student report that in detail describes our approach
and discusses the outcome of our experiments.


### Data
The data considered is the open-sourced TUH EEG Artifact data set published by
the Neural Engineering Data Consortium (NEDC).

## Repository
This Git-repo contains python-files for pre-processing as well as fully implemented pipelines coded with an object-oriented 
approach for easily running Machine Learning fitting (using various data augmentation techniques). Furthermore, python scripts
for obtaining results from pickled data files are to be found.


### Preprocessing
Code taken from David Enslev Nyrnbergs Git-repo, shared with us for this project.
The preprocessing is done in the "prepData"-directory.

### Model fitting
Original inspiration found in the benchmark paper on the TUH EEG Artifact data set published
by Roy, S. (2019)

The models considered as well as the code for the augmentation methods can be seen in th
"models"-directory. The hyperparameter-spaces considered for optimization of the models
(using Bayesian Optimization) can be found in the "evaluation.pipeline.py"-file.

### Evaluation
SMOTE / balancing pipeline as well as the augmentation-pipeline can be found in the "evaluation"-directory.
Some of the files, namely the activeLearning_pipeline is currenly broken as it does not store the correct results and is neither
optimized for the EEG approach.
Active Learning pipeline includes coding parts taken from exercises in the course 02463 on Active Learning and agency.

### Results
Our results are saved as numpy-pickles using a function from the "prepData.dataLoader"-script.
All results are found using a random seed set to 0 and can be reconstructed by reading the "paper/article"
written for this project.

