import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from scipy import stats
import numpy as np

pd.set_option("display.max_column", None)
pd.set_option("display.width", 1000)

path = r"C:\Users\linho\Desktop\CST\term3\pred_analytics\data\USA_Housing.csv"

df_house = pd.read_csv(path)
df_sub = df_house[['Avg. Area Income','Avg. Area House Age',
            'Avg. Area Number of Rooms','Avg. Area Number of Bedrooms',
            "Area Population", 'Price']]

z = np.abs(stats.zscore(df_sub))
# print(z)

THRESHOLD = 3
outliers = np.where(z > THRESHOLD)
# print(outliers)


# ===========================================================
# Exercise 3

path = r"C:\Users\linho\Desktop\CST\term3\pred_analytics\data\babysamp-98.txt"

df = pd.read_csv(path, sep='\t')
df_sub = df[['MomAge', 'MomMarital', 'numlive', "dobmm", 'gestation']]

z = np.abs(stats.zscore(df_sub))
# print(z)

# print(np.where((z > 2.33) | (z < -2.33)))
#
# print(df_sub.iloc[48][2])
# print(df_sub.iloc[48][4])
# print(df_sub.iloc[100][0])

df_price = df_house[['Price']]

z = np.abs(stats.zscore(df_price))
outliers = np.where(z > 3)
print(outliers)
for i in range(len(outliers[0])):
    row = outliers[0][i]
    col = outliers[1][i]
    print(df_price.loc[row][col])