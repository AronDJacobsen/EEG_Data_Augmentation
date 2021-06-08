# 02466 (DTU - AI and Data): Project work on TUH EEG Artifact classification 


This project investigates...

Data modelling and Machine Learning approaches to fitting ...


The outcome of the project is a student report that in detail describes our approach
and discusses the outcome of our experiments.



### Data
The data considered is the open-sourced TUH EEG Artifact data set published by
the Neural ... Consortium (NEDC). It consists of ...


## Repository
This Git-repo...



### Preprocessing
Code taken from ...

The preprocessing is done in the "prepData"-directory.


### Model fitting
Inspiration found in the benchmark paper on the TUH EEG Artifact data set published
by Roy, S. (20XX) LINK!

The models considered as well as the code for the augmentation methods can be seen in th
"models"-directory. The hyperparameter-spaces considered for optimization of the models
(using Bayesian Optimization) can be found in the "evaluation.pipeline.py"-file.



### Evaluation
SMOTE / balancing pipeline found in "evaluation.pipeline.py"-file and ...


Augmentation pipeline and Ensemble pipeline are found in evaluation as well
and are created similarly to the SMOTE / balancing pipeline.

Active Learning pipeline includes coding parts taken from exercises in the course 02463 on Active Learning and agency.

### Results
Her
#TODO: Create the correct npy results-files (merged)
Our results are saved as numpy-pickles using a function from the "prepData.dataLoader"-script.
All results are found using a random seed set to 0 and can be reconstructed by reading the "paper/article"
written for this project.

