import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from bioinfokit.visuz import cluster
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from prepData.dataLoader import LoadPickles

pickle_dir = r"C:\Users\Albert Kjøller\Documents\GitHub\EEG_epilepsia"


X, y, ID_frame = LoadPickles(pickle_path=pickle_dir, DelNan=True)
labels = ['eyem','chew','shiv','elpp','musc','null']

# Standardizing data and reformating target variable
Xstd = StandardScaler().fit_transform(X)
target = np.array([np.where(y[0]==1)[0][0] for i in y]).T
target[:1000] = 1 # Just an example!

pca = PCA(n_components = 3)

print("Running PCA...")
pca_out = pca.fit(Xstd)
loadings = pca_out.components_

# Projecting data to PCA basis
pca_scores = pca.fit_transform(Xstd)



fig = plt.figure(1, figsize=(4, 3))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
# Reorder the labels to have colors matching the cluster results
ax.scatter(pca_scores[:, 0], pca_scores[:, 1], pca_scores[:, 2], c=target, cmap=plt.cm.rainbow,
           edgecolor='k')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

plt.show()

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