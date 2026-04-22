import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# load data from the result of semantic_space_pca

data=np.load("/home/trojan/workspace/sample/HDSAD/semantic_space_pca.npz")
Z_all=data["Z_all"]
n_weibo=int(data["n_weibo"])
n_renmin=int(data["n_renmin"])

# check the accuracy of the data

assert Z_all.shape[0]==n_weibo+n_renmin

## elbow method

# select 50000 random samples for elbow method

rng=np.random.default_rng(42)
idx=rng.choice(Z_all.shape[0],size=50000,replace=False)
Z_elbow=Z_all[idx]

# calculate the elbow method

k_min,k_max=10,30
inertias=[]
k_list=list(range(k_min,k_max+1))

for k in k_list:
    kmeans=KMeans(
        n_clusters=k,
        n_init=10,
        max_iter=300,
        random_state=42,
        )
    kmeans.fit(Z_elbow)
    inertias.append(kmeans.inertia_)

# plot the elbow method

plt.figure(figsize=(10,5))
plt.plot(k_list,inertias,marker="o")
plt.xlabel("K")
plt.ylabel("Inertia(within-cluster sum of squares)")
plt.title("Elbow Method For Optimal K")
plt.grid(True,alpha=0.5)
plt.tight_layout()
plt.show()

## k-means clustering

k_final=50

kmeans_final=KMeans(
    n_clusters=k_final,
    n_init=10,
    max_iter=300,
    random_state=42,
    )
labels=kmeans_final.fit_predict(Z_all)

# save the result of k-means clustering

np.savez(
    "/home/trojan/workspace/sample/HDSAD/semantic_embedding_cluster.npz",
    labels=labels,
    n_clusters=k_final,
    Z_all=Z_all,
    n_weibo=n_weibo,
    n_renmin=n_renmin,
    inertia=float(kmeans_final.inertia_)
    )