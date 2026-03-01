import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

path = r"C:\Users\linho\Desktop\CST\term3\pred_analytics\data\individualTeammatePreferences.csv"

data = pd.read_csv(path)

df = data.copy()
del df['Full_Name']
del df['Unnamed: 0']
print(df)

kmeans = KMeans(n_clusters=25, random_state=0).fit(df)
centroids = kmeans.cluster_centers_

print('==== Centroids ====')
print(centroids)

print("\n==== Labels ====")
print(kmeans.labels_)

data['Label'] = 0.0
col_pos = len(data.keys()) - 1

for i in range(len(data)):
    data.iat[i, col_pos] = kmeans.labels_[i]

data = data.sort_values(['Label'])
pd.set_option("display.max_rows", data.shape[0]+1)
print(data.head(len(data)))

