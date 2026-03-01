import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def kmeans_demo():
    data = {
              # 2 2 2 2 2 2 2 2 2 2
        'x': [25, 34, 22, 27, 33, 33, 31, 22, 35, 34,
              # 0 0 0 0 0 0 0 0 0 0
              67, 54, 57, 43, 50, 57, 59, 52, 65, 47,
              # 1 1 1 1 1 1 1 1 1 1
              49, 48, 35, 33, 44, 45, 38, 43, 51, 46],

              # 2 2 2 2 2 2 2 2 2 2
        'y': [79, 51, 53, 78, 59, 74, 73, 57, 69, 75,
              # 0 0 0 0 0 0 0 0 0 0
              51, 32, 40, 47, 53, 36, 35, 58, 59, 50,
              # 1 1 1 1 1 1 1 1 1 1
              25, 20, 14, 12, 20, 5, 29, 27, 8, 7]}

    df = pd.DataFrame(data)
    kmeans = KMeans(n_clusters=4, random_state=142).fit(df)
    centroids = kmeans.cluster_centers_

    print("\n==== Centroid ======")
    print(centroids)

    print('\n==== Sample labels ====')
    print(kmeans.labels_)

    plt.scatter(df['x'], df['y'], c=kmeans.labels_, s=50, alpha=0.5)

    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, alpha=0.3)
    plt.show()

centroids = np.zeros((3, 2))
centroids[0][0] = 55.1
centroids[0][1] = 46.1
centroids[1][0] = 43.2
centroids[1][1] = 16.7
centroids[2][0] = 29.6
centroids[2][1] = 66.8

cluster_labels = [0, 1, 2]


def predict_cluster(centroids, X_test, cluster_labels):
    best_cluster_list = []

    for i in range(len(X_test)):
        smallest_dist = None
        best_cluster = None

        # compare each X value proximity with all centroids
        for row in range(centroids.shape[0]):
            distance = 0

            # get absolute distance between centroid and X
            for col in range(centroids.shape[1]):
                distance += math.sqrt((centroids[row][col] - X_test.iloc[i][col]) ** 2)

            # initialize best_cluster and smallest_dist during first iteration
            # or reassign best_cluster if smaller distance to centroid found
            if best_cluster is None or distance < smallest_dist:
                best_cluster = cluster_labels[row]
                smallest_dist = distance

        best_cluster_list.append(best_cluster)

    return best_cluster_list

X_test = pd.DataFrame({
    'x': [27, 40],
    'y': [68, 12]
})
predictions = predict_cluster(centroids, X_test, cluster_labels)
print(predictions)

X = pd.DataFrame({
    'x': [55],
    'y': [35]
})
print(predict_cluster(centroids, X, cluster_labels))