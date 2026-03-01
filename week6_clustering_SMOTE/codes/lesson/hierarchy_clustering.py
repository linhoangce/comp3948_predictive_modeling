import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

def demo():
    X = np.array(  [[5,3],
                    [10,15],
                    [15,12],
                    [24,10],
                    [30,30],
                    [85,70],
                    [71,80],
                    [60,78],
                    [70,55],
                    [80,91],])

    labels = range(1, 11)
    plt.scatter(X[:, 0], X[:, 1])

    # add labels to points
    for label, x, y in zip(labels, X[:, 0], X[:, 1]):
        plt.annotate(label, xy=(x,y),
                     xytext=(-1,2),
                     textcoords='offset points',
                     ha='right', va='bottom')
    plt.show()

    # draw dendrogram
    linked = linkage(X, 'single')
    label_list = range(1, 11)

    dendrogram(linked, orientation='top', labels=label_list,
               distance_sort='descending', show_leaf_counts=True)

    plt.show()

    cluster = AgglomerativeClustering(n_clusters=3, metric='euclidean',
                                      linkage='ward')
    cluster.fit_predict(X)
    print(cluster.labels_)

    plt.scatter(X[:, 0], X[:, 1], c=cluster.labels_, cmap='rainbow')
    plt.show()


# ======================================================================

