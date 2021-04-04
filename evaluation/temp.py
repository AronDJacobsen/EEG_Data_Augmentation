

for model in CV_scores:
    model_names.append(model)

    #initialize for appending
    accuracies = []
    weighted_f1s = []
    for artifact in CV_scores[model]: # each artifact
        #add sensitivity to the data
        sensitivities = CV_scores[model][artifact]['sensitivity']

        if artifact_names[artifact] in initial_data:
            initial_data[artifact_names[artifact]].append(sensitivities.mean())
        else:
            initial_data[artifact_names[artifact]] = [sensitivities.mean()]

        #confidence
        minus, mean, plus = mean_confidence_interval(sensitivities)
        confidence[model][artifact_names[artifact]] = [minus, mean, plus]


        accuracies.append(CV_scores[model][artifact]['accuracy'].mean())
        weighted_f1s.append(CV_scores[model][artifact]['F1'].mean())

    #confidence
    minus, mean, plus = mean_confidence_interval(accuracies)
    confidence[model]['accuracy'] = [minus, mean, plus]
    minus, mean, plus = mean_confidence_interval(weighted_f1s)
    confidence[model]['weighted_f1'] = [minus, mean, plus]


    # overall performance for all artifacts
    if 'accuracy' in initial_data:
        initial_data['accuracy'].append(np.mean(accuracies))
    else:
        initial_data['accuracy'] = [np.mean(accuracies)]

    if 'weighted_f1' in initial_data:
        initial_data['weighted_f1'].append(np.mean(weighted_f1s))
    else:
        initial_data['weighted_f1'] = [np.mean(weighted_f1s)]







