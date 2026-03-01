import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering

iris = load_iris()
X = iris.data
y = iris.target


def compare_different_distance_linkage_params(X, y, n_clusters, metric, linkage):
    X_scaled = MinMaxScaler().fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled,
                            columns=['sepal length (cm)', 'sepal width (cm)',
                                     'petal length (cm)', 'petal width (cm)'])

    cluster = AgglomerativeClustering(n_clusters=n_clusters,
                                      metric=metric, linkage=linkage)
    cluster.fit_predict(X_scaled)
    nmi = normalized_mutual_info_score(y, cluster.labels_)
    ars = adjusted_rand_score(y, cluster.labels_)

    print(f"\n============== {metric.upper()} and {linkage.upper()} =================")
    # print(cluster.labels_)
    print(f'*** Normalized mutual info: {nmi}')
    print(f'*** Adjusted rand score: {ars}')

    plt.scatter(X_scaled['petal length (cm)'], X_scaled['petal width (cm)'],
                c=cluster.labels_, alpha=0.5)
    plt.xlabel('Petal length')
    plt.ylabel('Petal width')
    plt.show()


metrics = ['euclidean', 'manhattan']
linkages = ['single', 'average', 'complete', 'ward']

for m in metrics:
    for l in linkages:
        if m == 'manhattan' and l == 'ward':
            break
        compare_different_distance_linkage_params(X, y, n_clusters=3,
                                                  metric=m, linkage=l)
        print('\n\n')
