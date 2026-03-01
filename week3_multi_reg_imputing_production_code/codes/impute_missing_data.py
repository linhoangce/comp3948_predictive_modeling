import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn import metrics

path = r"C:\Users\linho\Desktop\CST\term3\pred_analytics\data\babysamp-98.txt"

df = pd.read_table(path, sep="\t")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

# print(df.head())
# print(df.describe())

### Pat's version
def convertNAcellsToNum(colName, df, measureType):
    # Create two new columns based on original column names
    indicatorColName = "m_" + colName
    imputedColName = "imp_" + colName

    # get imputing method
    if(measureType == "mean"):
        imputedVal = df[colName].mean()
    if(measureType == 'media'):
        imputedVal = df[colName].mean()
    if(measureType == "mode"):
        imputedVal = float(df[colName].mode())
    else:
        print("Invalid measure type.")

    # populate new columns
    imputedColumn = []
    indicatorColumn = []

    for i in range(len(df)):
        isImputed = False

        # mi_OriginalName column stores imputed & original data
        if(np.isnan(df.loc[i][colName])):
            isImputed = True
            imputedColumn.append(imputedVal)
        else:
            imputedColumn.append(df.loc[i][colName])

        # mi_OriginalName column tracks if is imputed (1) or not (0)
        if(isImputed):
            indicatorColumn.append(1)
        else:
            indicatorColumn.append(0)

    # Append new columns to dataframe but always keep original column
    df[indicatorColName] = indicatorColumn
    df[imputedColName] = imputedColumn
    return df

### My version
def imputeNaN(col_name, df, measure_type):
    # create mask for missing data column with pandas
    mask = df[col_name].isna()

    # add indicator column
    df['m_' + col_name] = mask.astype(int)
    # make a copy of original column for imputing and keep original column
    df["imp_"+col_name] = df[col_name]

    # check and retrieve imputed value based on argument
    if measure_type.lower() == "mean":
        imp_value = df[col_name].mean()
    elif measure_type.lower() == "median":
        imp_value = df[col_name].median()
    elif measure_type.lower() == "mode":
        imp_value = df[col_name].mode()[0]
    else:
        print("Invalid measure type")

    # apply the mask and impute missing values to column
    df.loc[mask, "imp_"+col_name] = imp_value

    return df

print(df.isna().sum())

# df1 = convertNAcellsToNum('DadAge', df, 'mode')
df = imputeNaN('DadAge', df, 'mode')
df = imputeNaN('MomEduc', df, "mean")
df = imputeNaN("prenatalstart", df, "median")

print(df.isna().sum())
print(df.head())

# construct feature matrix
X = df[['gestation', 'm_DadAge', 'imp_DadAge',
        'm_prenatalstart', 'imp_prenatalstart',
        'm_MomEduc', 'imp_MomEduc']].values

X = sm.add_constant(X)
y = df['weight'].values

# split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

model = sm.OLS(y_train, X_train).fit()
y_pred = model.predict(X_test)

print(model.summary())
print(f'RMSE: {np.sqrt(metrics.mean_squared_error(y_test, y_pred))}')


