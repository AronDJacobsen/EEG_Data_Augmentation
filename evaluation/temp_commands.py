

#TODO: Hørte til mergeResultFiles





#TODO: Originally from the beginning of the class

self.aug_ratios = list(results.keys())
    self.smote_ratios = list(results[self.aug_ratios[0]].keys())
    self.folds = [key for key in results[self.aug_ratios[0]][self.smote_ratios[0]].keys() if type(key) == int]
    self.artifacts = list(results[self.aug_ratios[0]][self.smote_ratios[0]][self.folds[0]].keys())
    if i == 0:
        self.models = list(results[self.aug_ratios[0]][self.smote_ratios[0]][self.folds[0]][self.artifacts[0]].keys())
    else:
        self.models.append(
            list(results[self.aug_ratios[0]][self.smote_ratios[0]][self.folds[0]][self.artifacts[0]].keys()))


#TODO: Broken functions. Does not work with the data structure.


def plotPerformanceModels(self, performance_dict, error_dict, experiment, ratio, save_img=False):
    save_path = dir + self.slash + 'Plots' + self.slash + experiment

    # Plotting results
    art = len(self.artifacts)
    performance_vals = np.array(list(performance_dict.values())[:art]).T
    error_vals = np.array(list(error_dict.values())[:art]).T

    for indv_model, name in enumerate(self.models):
        plt.bar(x=self.artifacts, height=performance_vals[indv_model, :], width=0.5, color="lightsteelblue")
        plt.errorbar(x=self.artifacts, y=performance_vals[indv_model, :], yerr=error_vals[indv_model, :], fmt='.',
                     color='k')
        plt.title(name + " - SMOTE RATIO:" + str(ratio - 1))
        plt.ylim(0, 1)
        if save_img:
            plt.savefig(("{}{:s}{}_SMOTE_{}.png").format(save_path, self.slash, name, ratio - 1))
        plt.show()


def plotPerformanceClasses(self, performance_dict, error_dict, experiment, ratio,
                           save_img=False):
    save_path = dir + self.slash + 'Plots' + self.slash + experiment

    # Plotting results
    art = len(self.artifacts)
    performance_vals = np.array(list(performance_dict.values())[:art])
    error_vals = np.array(list(error_dict.values())[:art])

    for indv_art, name in enumerate(self.artifacts):
        plt.bar(x=self.models, height=performance_vals[indv_art, :], width=0.5, color="lightsteelblue")
        plt.errorbar(x=self.models, y=performance_vals[indv_art, :], yerr=error_vals[indv_art, :], fmt='.',
                     color='k')
        plt.title(name + " - SMOTE RATIO:" + str(ratio - 1))
        plt.xticks(rotation=25)
        plt.ylim(0, 1)
        if save_img:
            plt.savefig(("{}{:s}{}_SMOTE_{}.png").format(save_path, self.slash, name, ratio - 1))
        plt.show()


def plotHyperopt(self, file_name):
    try:
        results_basepath = self.slash.join(self.pickle_path.split(self.slash)[:-1])

        # fold, artifact, model, scores
        results = LoadNumpyPickles(pickle_path=results_basepath + self.slash + "performance",
                                   file_name=self.slash + "results" + experiment_name + '.npy',
                                   windowsOS=self.windowsOS)
        results = results[()]

        # fold, artifact, model, hyperopt iterations
        HO_trials = LoadNumpyPickles(pickle_path=results_basepath + self.slash + "hyperopt",
                                     file_name=self.slash + "ho_trials" + experiment_name + '.npy',
                                     windowsOS=self.windowsOS)
        HO_trials = HO_trials[()]

        # only choose one fold
        # construct keys
        folds = list(results.keys())
        artifacts = list(results[folds[0]].keys())
        models = list(results[folds[0]][artifacts[0]].keys())
        scores = list(results[folds[0]][artifacts[0]][models[0]].keys())

        single = HO_trials[folds[0]][artifacts[0]][models[0]]

        # TODO: Not completely functioning! It does not show the plots

        # hyperopt
        cols = list(single.columns)
        n = len(cols)
        for i in range(n - 1):  # for every parameter
            plt.scatter(single[cols[i]], single[cols[n - 1]])
            plt.title('HyperOpt: model: {}, artifact: {}'.format(models[0], cols[i]))
            plt.xlabel(cols[i])
            plt.ylabel('accuracy')
            plt.show()

    except KeyError:
        print("\n\nERROR: No Hyperopt queries used for this model!")
