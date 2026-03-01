import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

path = r"C:\Users\linho\Desktop\CST\term3\pred_analytics\data\babysamp-98.txt"

df = pd.read_table(path, sep='\t')

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

print(df.head())

# Pat's version
def createBinaryDummyVar(df, colName, val0):
    imputedColName = "m_" + colName
    imputedCol = []

    for i in range(len(df)):
        if df.loc[i][colName] == val0:
            imputedCol.append(0)
        else:
            imputedCol.append(1)

    df[imputedColName] = imputedCol
    return df

# my version
def binary_dummy_var(df, col_name, val):
    df["m_"+col_name] = (df[col_name] == val).astype(int)
    return df

df1 = binary_dummy_var(df, 'preemie', True)
df1 = binary_dummy_var(df, "sex", 'F')
print(df1.head())

tempDf = df[['preemie', 'sex']]
dummyDf = pd.get_dummies(tempDf, columns=['preemie', 'sex'], dtype=int)
df = pd.concat([df, dummyDf], axis=1)
print(df.head())
