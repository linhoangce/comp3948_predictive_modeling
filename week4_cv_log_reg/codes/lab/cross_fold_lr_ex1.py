import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LinearRegression

import statsmodels.api as sm

path = r"C:\Users\linho\Desktop\CST\term3\pred_analytics\data\USA_Housing.csv"

dataset = pd.read_csv(path)

pd.set_option('display.max_columns', None)
pd.set_option("display.width", 1000)

print(dataset.head())
print(dataset.describe())

# Extract only numeric columns
X = dataset[["Avg. Area Income", "Avg. Area House Age",
             "Avg. Area Number of Rooms", "Avg. Area Number of Bedrooms",
             "Area Population"]]
X = sm.add_constant(X)

y = dataset['Price']

# split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = sm.OLS(y_train, X_train).fit()
y_pred = model.predict(X_test)

print(model.summary())

# Based on p value, removing Avg. Area Number of Bedrooms from X
X = dataset[["Avg. Area Income", "Avg. Area House Age",
             "Avg. Area Number of Rooms", "Area Population"]]
X = sm.add_constant(X)

# resplit dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

model1 = sm.OLS(y_train, X_train).fit()
y_pred = model1.predict(X_test)

print("\n\n===========================================================================")
print(" ************************ OLS Model 1 *************************\n")
print(model1.summary())
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")

### We'll use this updated dataset for training a LinearRegression model
### since all p values are 0.

print("\n\n********************* LINEAR REGRESSION MODEL *************************\n")
X = dataset[["Avg. Area Income", "Avg. Area House Age",
             "Avg. Area Number of Rooms", "Area Population"]]
y = dataset[['Price']]
rmses = []

kfold = KFold(n_splits=5, shuffle=True)

for train_idx, test_idx in kfold.split(X):
    X_train = X.loc[train_idx, :]
    X_test = X.loc[test_idx, :]
    y_train = y.loc[train_idx, :]
    y_test = y.loc[test_idx, :]

    lr_model = LinearRegression()
    # train model
    lr_model.fit(X_train, y_train)

    # make predictions
    y_pred = lr_model.predict(X_test)
    # calculate loss
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    rmses.append(rmse)

    print(f"RMSE: {rmse}")

avgRMSE = np.average(rmses)
print(f"Average RMSE: {avgRMSE}")