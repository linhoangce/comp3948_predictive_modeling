import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, SpectralClustering

path = r"C:\Users\linho\Desktop\CST\term3\pred_analytics\data\old_faithful.csv"

df = pd.read_csv(path)

plt.figure(figsize=(6, 6))
plt.scatter(df.iloc[:, 0], df.iloc[:, 1])
plt.xlabel('Eruption time in mins')
plt.ylabel('Watting time to next eruption')
plt.title('Visualization of raw data')
plt.show()

X_std = StandardScaler().fit_transform(df)

# run local implementation of kmeans
km = KMeans(n_clusters=2, max_iter=100)
km.fit(X_std)
centroids = km.cluster_centers_

# plot clusters
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].scatter(X_std[km.labels_ == 0, 0], X_std[km.labels_ == 0, 1],
            c='green', label='cluster 1')
ax[0].scatter(X_std[km.labels_ == 1, 0], X_std[km.labels_ == 1, 1],
            c='blue', label='cluster 2')
ax[0].scatter(centroids[:, 0], centroids[:, 1],
            s=300, c='r', label='centroid')
ax[0].legend(loc='best')
ax[0].set_xlim(-2, 2)
ax[0].set_ylim(-2, 2)
ax[0].set_xlabel('Eruption time in mins')
ax[0].set_ylabel('Waiting time to next eruption')
ax[0].set_title('Visualization of clustered data', fontweight='bold')
ax[0].set_aspect('equal')

# run kmeans algorithm and get index of data points clusters
sse = []
list_k = list(range(1, 10))

for k in list_k:
    km = KMeans(n_clusters=k)
    km.fit(X_std)
    sse.append(km.inertia_)

# plot sse against k
ax[1].plot(list_k, sse, '-o')
ax[1].set_xlabel('Number of clusters *k*')
ax[1].set_ylabel('Sum of squared distance')

plt.tight_layout()
plt.show()