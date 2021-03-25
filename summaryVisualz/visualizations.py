import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from bioinfokit.visuz import cluster
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from prepData.dataLoader import LoadNumpyPickles
from collections import defaultdict
from sklearn.model_selection import train_test_split


pickle_path = r"C:\Users\Albert Kjøller\Documents\GitHub\EEG_epilepsia"

#loading data - define which pickles to load (with NaNs or without)
X_file = r"\X_clean.npy"    #X_file = r"\X.npy"
y_file = r"\y_clean.npy"    #y_file = r"\y.npy"
ID_file = r"\ID_frame_clean.npy"   #ID_file = r"\ID_frame.npy"

X, y, ID_frame = LoadNumpyPickles(pickle_path=pickle_path, X_file=X_file, y_file=y_file, ID_file=ID_file, DelNan = False)

labels = ['eyem','chew','shiv','elpp','musc','null']
count_labels = [np.unique(y[:,i],return_counts=True)[1][1] for i in range(len(labels))]
ratio_labels = np.array(count_labels) / len(y)


# Fining out how many of the files have multiple labels and how many labels these files have.
multiple_labels = [np.sum(y[i,:]) for i in range(len(y))]
multiple_labels_count = np.unique(multiple_labels, return_counts=True)

# Initializing dictionaries for finding where we have multiple labels
indices_2 = []
mult_dict_2 = {'eyem':0, 'chew':0 ,'shiv':0, 'elpp':0, 'musc':0,'null':0}
indices_3 = []
mult_dict_3 = {'eyem':0, 'chew':0 ,'shiv':0, 'elpp':0, 'musc':0,'null':0}

for i, y_single in enumerate(y):
    pos = np.where(y_single == 1)[0]
    if len(pos) == 2:
        indices_2.append(i)
        for pos_idx in pos:
            mult_dict_2[labels[pos_idx]] += 1
    if len(pos) > 2:
        indices_3.append(i)
        for pos_idx in pos:
            mult_dict_3[labels[pos_idx]] += 1

print("\n{:d} windows contain two different labels. These are distributed as follows: ".format(len(indices_2)) + str(mult_dict_2))

print("\n{:d} windows contain three different labels. These are distributed as follows: ".format(len(indices_3)) + str(mult_dict_3))
print("The labelling of the files holding three labels are seen below: \n" + str(y[indices_3]))


# colors = "lightcoral" or "lightsteelblue" or lightslategrey"
plt.bar(np.arange(len(labels)), ratio_labels, color="lightsteelblue", tick_label=labels)
plt.show()
print("Ratio of labels, following " + str(labels) + ": \t" + str(np.round(ratio_labels*100, 3)))




# Choosing a subset of data (still stratified) to better visualize data points.
n = 750 # But we will get a bit more since we use the ratio_labels that does not account for multilabelled points
np.random.seed(20)
sample_ratio = np.ceil(ratio_labels * n)
X_for_viz = np.empty(( int(sum(sample_ratio)) , X.shape[1]))
y_for_viz = np.empty(( int(sum(sample_ratio)) , y.shape[1]))

sample_ratio_cum = 0
for i in range(len(labels)):
    singles = np.setdiff1d(np.arange(len(y)), indices_2)
    pos = np.where(y[singles,i] == 1)[0]
    np.random.shuffle(pos)
    pos = pos[:int(sample_ratio[i])]
    # pos_valid = [True if p not in indices_2 and i not in indices_3 else False for p in pos]

    X_for_viz[sample_ratio_cum : (sample_ratio_cum + int(sample_ratio[i])), :] = X[pos, :]
    y_for_viz[sample_ratio_cum : (sample_ratio_cum + int(sample_ratio[i])), :] = y[pos, :]

    sample_ratio_cum += int(sample_ratio[i])

randomize = np.arange(len(X_for_viz))
np.random.shuffle(randomize)
X_for_viz = X_for_viz[randomize]
y_for_viz = y_for_viz[randomize]


# Standardizing data and reformating target variable
Xmean, Xerr = np.mean(X_for_viz, axis=0), np.std(X_for_viz, axis=0)
Xstd = (X_for_viz - Xmean) / Xerr
target = np.array([np.where(y_single==1)[0][0] for y_single in y_for_viz]).T  # Not accounting for multiple labels here (due to plotting)

pca = PCA(n_components = 20)

print("Running PCA...")
pca_out = pca.fit(Xstd)
loadings = pca_out.components_

# Projecting data to PCA basis
pca_scores = pca.fit_transform(Xstd)

upsample_plot = False
if upsample_plot == True:
    label_to_up = 0
    #Upsampling for visualization (only eyemovement)
    from models.dataAugmentation import balance
    Xnew, ynew = balance(X_for_viz, y_for_viz[:,label_to_up], 0.3, 0.05)
    new_pos = np.where(ynew == 1)[0]
    X_for_viz = np.concatenate((X_for_viz, Xnew[new_pos]))
    Xup = (X_for_viz - Xmean) / Xerr

    one = np.zeros(6)
    one[label_to_up] = 1
    y_up = np.array([list(one)] * len(new_pos))
    y_for_viz = np.concatenate((y_for_viz, y_up))
    target = np.array([np.where(y_single == 1)[0][0] for y_single in
                       y_for_viz]).T  # Not accounting for multiple labels here (due to plotting)

    pca_scores = pca.fit_transform(Xup)


colors = ["tab:red", "tab:blue", "tab:green", "darkslategrey", "m", "lightsteelblue"]
transparency = [1] * len(labels)
transparency[-1] = 0.3 # Blurrying the null-class

fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
for i in range(len(labels)-1, -1, -1):
    ax.scatter(pca_scores[:,0][target==i], pca_scores[:,1][target==i], pca_scores[:,2][target==i], c = colors[i] ,s=50, label= labels[i], alpha=transparency[i])

# ax.scatter(pca_scores[:, 0], pca_scores[:, 1], pca_scores[:, 2], c=target,
 #          cmap=plt.cm.Set1, edgecolor='k', s=40, label=labels)
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])

plt.legend()
plt.show()

#Adding null as last class
for i in range(len(labels)-1, -1, -1):
    plt.scatter(pca_scores[:,0][target==i], pca_scores[:,1][target==i], s=50, c=colors[i], label = labels[i], alpha=transparency[i])
plt.xlabel('1st eigenvector')
plt.ylabel('2nd eigenvector')
plt.legend(loc=1)
plt.show()

print(np.unique(target, return_counts=True))

print("Breakpoint for evaluation of PCA plot...")

# Might be useful
"""
np.cumsum(pca_out.explained_variance_ratio_)
num_pc = pca_out.n_features_
pc_list = ["PC"+str(i) for i in list(range(1, num_pc+1))]

cluster.screeplot(obj=[pc_list, pca_out.explained_variance_ratio_])



model = pca(n_components=3)
results = model.fit_transform(X)

label = ['eyem','chew','shiv','elpp','musc','null']

fig, ax = model.scatter3d()
fig, ax = model.biplot3d(n_feat=6, legend=True)

print(X)
"""