from results import *




#imported from toolbox from data mining course
def mcnemar(y_true, yhatA, yhatB, alpha=0.05):
    # perform McNemars test
    nn = np.zeros((2,2))
    c1 = yhatA - y_true == 0
    c2 = yhatB - y_true == 0

    nn[0,0] = sum(c1 & c2)
    nn[0,1] = sum(c1 & ~c2)
    nn[1,0] = sum(~c1 & c2)
    nn[1,1] = sum(~c1 & ~c2)

    n = sum(nn.flat);
    n12 = nn[0,1]
    n21 = nn[1,0]

    thetahat = (n12-n21)/n
    Etheta = thetahat

    Q = n**2 * (n+1) * (Etheta+1) * (1-Etheta) / ( (n*(n12+n21) - (n12-n21)**2) )

    p = (Etheta + 1)*0.5 * (Q-1)
    q = (1-Etheta)*0.5 * (Q-1)

    CI = tuple(lm * 2 - 1 for lm in scipy.stats.beta.interval(1-alpha, a=p, b=q) )

    p = 2*scipy.stats.binom.cdf(min([n12,n21]), n=n12+n21, p=0.5)
    print("Result of McNemars test using alpha=", alpha)
    print("Comparison matrix n")
    print(nn)
    if n12+n21 <= 10:
        print("Warning, n12+n21 is low: n12+n21=",(n12+n21))

    print("Approximate 1-alpha confidence interval of theta: [thetaL,thetaU] = ", CI)
    print("p-value for two-sided test A and B have same accuracy (exact binomial test): p=", p)

    return thetahat, CI, p







if __name__ == '__main__':

    pd.set_option("display.max_rows", None, "display.max_columns", None)

    #Albert
    dir = r"C:\Users\Albert Kjøller\Documents\GitHub\EEG_epilepsia"
    y_true_path = r"C:\Users\Albert Kjøller\Documents\GitHub\EEG_epilepsia\results\y_true"

    #Aron
    #dir = r"/Users/Jacobsen/Documents/GitHub/EEG_epilepsia" + "/"
    #y_true_path = r"/Users/Jacobsen/Documents/GitHub/EEG_epilepsia/results/y_true" + "/"


    windowsOS = True


    #smote(T/F), best_model, best_ratio
    #hardcoding
    best = {
        'MixUp': {
            'eyem': (False, 'RF', 1),
            'chew': (False, 'LR', 2),
            'shiv': (False, 'LR', 2),
            'elpp': (False, 'LR', 0.5),
            'musc': (False, 'GNB', 0.5),
            'null': (False, 'SGD', 1)
        },

        'colorNoise': {
            'eyem': (False, 'RF', 1.5),
            'chew': (True, 'LR', 0),
            'shiv': (True, 'LR', 0),
            'elpp': (False, 'LR', 0),
            'musc': (False, 'GNB', 2),
            'null': (False, 'SGD', 0.5)
        },

        'whiteNoise': {
            'eyem': (True, 'RF', 1.5),
            'chew': (False, 'LR', 0),
            'shiv': (True, 'LR', 0),
            'elpp': (True, 'LR', 0.5),
            'musc': (False, 'GNB', 2),
            'null': (False, 'SGD', 0.5)
        },

        'GAN': {
            'eyem': (False, 'RF', 0),
            'chew': (False, 'LR', 1),
            'shiv': (False, 'LR', 0.5),
            'elpp': (False, 'LR', 0),
            'musc': (False, 'GNB', 0),
            'null': (False, 'SGD', 0)
        }
    }

    artifact_names = ['eyem', 'chew', 'shiv', 'elpp', 'musc', 'null']

    #experiments = ["colorNoiseimprovement", "whiteNoiseimprovement", "GANimprovement", "MixUpimprovement"]
    #experiments = ["augmentation_colorNoise", "augmentation_whiteNoise", "augmentation_GAN", "augmentation_MixUp"]

    methods = ["colorNoise", "whiteNoise", "GAN", "MixUp"]
    #methods = ["MixUp"]

    df = pd.DataFrame(columns=artifact_names, index=methods)

    # y true
    folds = [0, 1, 2, 3, 4]

    # getting y_true predicitons
    results_y_true = LoadNumpyPickles(pickle_path=y_true_path, file_name=r"\y_true_5fold_randomstate_0.npy", windowsOS=windowsOS)
    results_y_true = results_y_true[()]
    y_true_dict = {}
    for artifact in artifact_names:
        y_true_art = []
        for i in folds:
            y_true_art.append(results_y_true[i][artifact]['y_true'])
        y_true_dict[artifact] = y_true_art



    for i, method in enumerate(methods):
        print(method)

        #for each method
        #activating augmentation
        augmentation = "augmentation_" + method
        augmentation_name = "_" + augmentation + "_merged_allModels"
        obj_aug = getResults(dir, augmentation, augmentation_name, merged_file=True, windowsOS=windowsOS)

        #activating improvement
        improvement = method + "improvement"
        improvement_name = "_" + improvement + "_merged_allModels"
        obj_improv = getResults(dir, improvement, improvement_name, merged_file=True, windowsOS=windowsOS)


        # temporary for saving
        artifact_list = []

        for j, artifact in enumerate(artifact_names):
            print(artifact)
            # y true predictions
            y_true = np.concatenate(y_true_dict[artifact])


            #finding best for this artifact and method
            smote, best_model, best_ratio = best[method][artifact]

            #getting predictions

            if not smote: # if augmentation was best model
                # getting predictions
                y = obj_aug.getPredictions(models=[best_model], aug_ratios=[best_ratio], smote_ratios = [1],  withFolds=False)
                y = obj_aug.compressDict(y, smote_ratio=1, aug_ratio=best_ratio)
                # best models predictions
                B = y[best_model][artifact]

                y_c = obj_aug.getPredictions(models=[best_model], aug_ratios=[0], smote_ratios = [1], withFolds=False)
                y_c = obj_aug.compressDict(y_c, smote_ratio=1, aug_ratio=0)
                A = y_c[best_model][artifact]

            else: # if improvement was best model
                # getting predictions
                y = obj_improv.getPredictions(models=[best_model], aug_ratios=[best_ratio], smote_ratios = [1],  withFolds=False)
                y = obj_improv.compressDict(y, smote_ratio=1, aug_ratio=best_ratio)
                # best models predictions
                B = y[best_model][artifact]

                #getting control predictions
                y_c = obj_improv.getPredictions(models=[best_model], aug_ratios=[0], smote_ratios = [1], withFolds=False)
                y_c = obj_improv.compressDict(y_c, smote_ratio=1, aug_ratio=0)
                A = y_c[best_model][artifact]

            # McNemar, from toolbox in data mining course
            [thetahat, CI, p] = mcnemar(y_true, A, B, alpha = 0.05)
            #investigate if A is better than B
            #H0: have same performance
            #thetahat is difference in accuracy, A-B
            #CI is condfidence interval of this difference
            #p is pvalue, low means reject H0 of same performance
            artifact_list.append(float(f"{p:.3e}")) # modifying format

        df.loc[method] = artifact_list
        stop=0

    #getting numbers
    df = df.astype(float)

    print('Dataframe:')
    print(df)

    print('Latex format:')
    df_latex = df.to_latex()
    print(df_latex)


