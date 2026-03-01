import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

import matplotlib.pyplot as plt

path = (r"C:\Users\linho\Desktop\CST\term3\pred_analytics"
        r"\data\winequality.csv")

dataset = pd.read_csv(path, sep=',')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print(dataset.head())
print(dataset.describe())

# extract feature columns
X = dataset[[ 'volatile acidity', 'chlorides',
             'sulphates', 'alcohol']]

# adding an intercept which is required.
# this is only need when using sm.OLS
# the intercept centers the error residuals around zero
# which helps to avoid overfitting
X = sm.add_constant(X)

y = dataset['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=0)

model = sm.OLS(y_train, X_train).fit()
preds = model.predict(X_test)

print(model.summary())

print(f'RMSE: {np.sqrt(metrics.mean_squared_error(y_test, preds))}')

def plot_prediction_vs_actual(plt, title, y_test, preds):
    plt.scatter(y_test, preds)
    plt.legend()
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Predicted (Y) vs. Actual (X): ' + title)
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             'k--')

def plot_residuals_vs_actual(plt, title, y_test, preds):
    residuals = y_test - preds
    plt.scatter(y_test, residuals, label='Residuals vs Actual')
    plt.xlabel('Actual')
    plt.ylabel('Residuals')
    plt.title('Error Residuals (Y) vs. Actual (X):'+title)
    plt.plot([y_test.min(), y_test.max()], [0, 0], 'k--')

def plot_residual_hist(plt, title, y_test, preds, bins):
    residuals = y_test - preds
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.hist(residuals, label='Residuals vs Actual', bins=bins)
    plt.title('Error Residual Frequency:'+ title)
    plt.plot()

def draw_validation_plots(title, bins, y_test, preds):
    plt.subplots(nrows=1, ncols=3, figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plot_prediction_vs_actual(plt, title, y_test, preds)

    plt.subplot(1, 3, 2)
    plot_residuals_vs_actual(plt, title, y_test, preds)

    plt.subplot(1, 3, 3)
    plot_residual_hist(plt, title, y_test, preds, bins)
    plt.show()

BINS = 8
TITLE = 'Wine Quality'
draw_validation_plots(TITLE, BINS, y_test, preds)