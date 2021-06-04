from evaluation.pipeline import *

if __name__ == '__main__':
    """ Select path to the data-pickles ! """
    # pickle_path = r"/zhome/2d/7/138174/Desktop/EEG_epilepsia"
    # pickle_path = r"/Users/Jacobsen/Documents/GitHub/EEG_epilepsia" + "/"
    pickle_path = r"/Users/philliphoejbjerg/Documents/GitHub/EEG_epilepsia"

    windowsOS = False

    """ Loading data - define which pickles to load (with NaNs or without) """
    X_file = r"\X_clean.npy"  # X_file = r"\X.npy"
    y_file = r"\y_clean.npy"  # y_file = r"\y.npy"
    ID_file = r"\ID_frame_clean.npy"  # ID_file = r"\ID_frame

    """ Define pipeline-class to use."""
    aug_pipeline = pipeline(pickle_path, X_file, y_file, ID_file, windowsOS=windowsOS)

    """ Define model to be evaluated and filename:    
    >>> 'model'             should be the name of the function that should be called form the models.models.py script.
                            I.e. 'LR' for calling the Logistic Regression.

    >>> 'experiment'        is the folder in which the files will be created. 

    >>> 'experiment_name'   is the name ending of the name of the pickles created.
                            When doing Augmentation it should end with either _GAN / _Noise / _MixUp,
                            such that the following command can work properly:  

                            if experiment_name.split("_")[-1] == 'GAN':

                            When running the pipeline on a single artifact, one should manually write this in the
                            'experiment_name', i.e. experiment + model + "Null" + aug_method

    >>> 'noise_experiment'  is the directory of the folder containing the noise files to be used. Should be None when
                            not experimenting with Noise Addition augmentation technique. """

    model = "LR"
    aug_method = "_GAN"  # or '_Noise' or so.
    artifact = "shiv"

    experiment = "augmentation_GAN"  # 'DataAug_color_noiseAdd_LR'
    experiment_name = "_" + experiment + "_" + model + artifact + aug_method  # "_DataAug_color_Noise" added to saving files. For augmentation end with "_Noise" or so.
    noise_experiment = None #r"\colornoise30Hz_covarOne"  # None or r"\whitenoise_covarOne" or  r"\colornoise30Hz_covarOne" #

    """ Define ratios to use for SMOTE and data augmentation techniques !"""
    smote_ratios = np.array([0])#np.array([0, 0.5, 1, 1.5, 2])
    aug_ratios = np.array([0.5])

    """ Specify other parameters"""
    HO_evals = 1
    K = 2
    random_state_val = 0

    # Example of normal run - with no smote and no augmentation. For illustration, 1-Fold CV.
    aug_pipeline.runPipeline(model=model,
                             HO_evals=HO_evals,
                             smote_ratios=smote_ratios, aug_ratios=aug_ratios,
                             experiment=experiment,
                             experiment_name=experiment_name,
                             random_state=random_state_val,
                             noise_experiment=noise_experiment,
                             #artifact_names=[artifact],
                             K=K)
