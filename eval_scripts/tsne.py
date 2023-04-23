from scipy.spatial import distance
from sklearn.manifold import TSNE
from collections import defaultdict
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os

anomaly = None
normal = None

SEM_DIR = "resnet18_semcodes/"

#Load labels
labels = []
with open("/home/eprakash/shanghaitech/testing/small_obj_test_labels_list.txt", "r") as fp:
    for line in fp:
        label = int(line.strip())
        labels.append(label)

#Load centerframe idxs
idx_to_centerframe = defaultdict(int)
with open("/home/eprakash/diffae/idx_to_centerframe_small_obj.txt", "r") as fp:
    for line in fp:
        i, idx = line.strip().split(",")
        i = int(i)
        idx = int(idx)
        idx_to_centerframe[i] = idx

#Separate normal and anomaly codes
n = 0
for n in range(len(os.listdir(SEM_DIR))):
    label = labels[idx_to_centerframe[n]]
    cond = np.load(SEM_DIR + str(n) + ".npy")[None, :]
    if (label == 1):
        if (anomaly is None):
            anomaly = cond
        else:
            anomaly = np.concatenate((anomaly, cond), axis=0)
    else:
        if (normal is None):
            normal = cond
        else:
            normal = np.concatenate((normal, cond), axis=0)

#print(normal.shape)
#print(anomaly.shape)

#Find dists as sanity check
np.random.shuffle(anomaly)
np.random.shuffle(normal)
print("L2 Dist: ", np.linalg.norm(np.mean(normal, axis=0) - np.mean(anomaly, axis=0)))
print("JS Dist: ", np.mean(distance.jensenshannon(normal[:600], anomaly[:600])))

#Consolidate data and labels
y = ['Anomaly' for i in range(len(anomaly))]
y = y + ['Normal' for i in range(len(normal))]
semantic = np.concatenate((anomaly, normal), axis=0)

#print(semantic.shape, len(y))

#Find TSNE
n_components = 2
tsne = TSNE(n_components, perplexity=10)
tsne_result = tsne.fit_transform(semantic)

#Plot TSNE result
tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'label': y})
fig, ax = plt.subplots(1)
sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax,s=120)
lim = (tsne_result.min()-5, tsne_result.max()+5)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_aspect('equal')
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

#Save plot
plt.savefig("st_resnet18.png")
