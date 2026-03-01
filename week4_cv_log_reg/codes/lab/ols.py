import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import statsmodels.api as sm
import numpy as np

path = r"C:\Users\linho\Desktop\CST\term3\pred_analytics\data\winequality.csv"

dataset = pd.read_csv(path)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

print(dataset.head())
print(dataset.describe())

X = dataset[['volatile acidity', 'chlorides', 'total sulfur dioxide',
             'sulphates', 'alcohol']]
X = sm.add_constant(X)
y = dataset['quality']

# split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

model = sm.OLS(y_train, X_train).fit()
preds = model.predict(X_test)

print(model.summary())

print(f"RMSE: {np.sqrt(metrics.mean_squared_error(y_test, preds))}")