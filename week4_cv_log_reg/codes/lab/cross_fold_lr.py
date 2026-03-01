import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

path = r"C:\Users\linho\Desktop\CST\term3\pred_analytics\data\winequality.csv"

dataset = pd.read_csv(path)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

X = dataset[['chlorides', 'total sulfur dioxide']]
y = dataset[['quality']]

# # split dataset
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2
# )

kfold = KFold(n_splits=3, shuffle=True)

rmses = []

model = LinearRegression()

for train_index, test_index in kfold.split(X):
    # use index list to isolate rows for train and test sets
    # get rows filtered by index and all columns
    # X.loc[row_number_array, all_columns]
    X_train = X.loc[train_index, :]
    X_test = X.loc[test_index, :]
    y_train = y.loc[train_index, :]
    y_test = y.loc[test_index, :]

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    rmses.append(rmse)

    print(f"RMSE: {rmse}")

avgRMSE = np.mean(rmses)
print(f"Average rmse: {avgRMSE}")


