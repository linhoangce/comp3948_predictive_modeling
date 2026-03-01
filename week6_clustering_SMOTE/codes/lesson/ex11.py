import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import MinMaxScaler

iris = load_iris()
X = iris.data
y = iris.target

sc = MinMaxScaler()
X_scaled = sc.fit_transform(X)

sse = []
k_list = list(range(1, 10))

for k in k_list:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit_predict(X_scaled)
    sse.append(kmeans.inertia_)

# ============================================================
# cluster with k = 2

kmeans = KMeans(n_clusters=2, max_iter=100)
kmeans.fit(X_scaled)
centroids = kmeans.cluster_centers_

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 7))

ax[0].plot(k_list, sse, '-o')
ax[0].set_xlabel("Number of clusters")
ax[0].set_ylabel("Sum of squared distance")

ax[1].scatter(X_scaled[kmeans.labels_ == 0, 0], X_scaled[kmeans.labels_ == 0, 1],
              c='green', label='cluster 1')
ax[1].scatter(X_scaled[kmeans.labels_ == 1, 0], X_scaled[kmeans.labels_ == 1, 1],
              c='blue', label='cluter 2')
ax[1].scatter(centroids[:, 0], centroids[:, 1],
              c='r', label='Centroid', s=150)
plt.tight_layout()
plt.show()

nmi = normalized_mutual_info_score(y, kmeans.labels_)
ars = adjusted_rand_score(y, kmeans.labels_)

print(f"NMI: {nmi}")
print(f"ARS {ars}")



