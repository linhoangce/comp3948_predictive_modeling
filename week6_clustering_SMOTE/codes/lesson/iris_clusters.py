import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
from sklearn import datasets
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering


iris = datasets.load_iris()
X = iris.data
y = iris.target

# scale data
sc = MinMaxScaler()
X_scaled = sc.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled,
                        columns=['sepal length (cm)', 'sepal width (cm)',
                                 'petal length (cm)', 'petal width (cm)'])
print(X_scaled.head())

# draw dendrogram
plt.subplots(2, 1, figsize=(10, 7))
plt.subplot(2, 1, 1)
plt.title('Dendrogram')
dend = shc.dendrogram(shc.linkage(X_scaled, method='ward'))

# predict cluster
cluster = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
cluster.fit_predict(X_scaled)
print(cluster.labels_)

# draw scatter plot
plt.subplot(2, 1, 2)
# plt.scatter(X_scaled['petal length (cm)'], X_scaled['petal width (cm)'],
#             c=cluster.labels_, alpha=0.5)
plt.scatter(X_scaled['sepal length (cm)'], X_scaled['sepal width (cm)'],
            c=cluster.labels_, alpha=0.5)
plt.xlabel('Sepal length', fontsize=15)
plt.ylabel('Sepal width', fontsize=15)
plt.show()

# Calculate Normalized mutual information and adjusted rand score
nmi = normalized_mutual_info_score(y, cluster.labels_)
ars = adjusted_rand_score(y, cluster.labels_)

print(f'\n=== Normalized Mutual Information === {nmi}')
print(f'=== Adjusted Rand Score === {ars}')