import pandas as pd
import numpy as np
import statsmodels.api as sm
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

from lab.plot import show_residual_rmse

path = r"C:\Users\linho\Desktop\CST\term3\pred_analytics\data\winequality.csv"

df = pd.read_csv(path)
feature_cols = ['volatile acidity', 'chlorides',
                'total sulfur dioxide','sulphates','alcohol']
X = df[feature_cols]
y = df['quality']

X = sm.add_constant(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

model = sm.OLS(y_train, X_train).fit()
y_pred = model.predict(X_test)

print(model.summary())
print(f'\nRMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}')

def show_residuals_vs_actual(y_test, y_pred):
    residuals = y_test - y_pred
    plt.scatter(y_test, residuals)
    plt.xlabel('Actual')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Actual')
    plt.plot([y_test.min(), y_test.max()], [0,0], 'k--')
    plt.show()

# show_residuals_vs_actual(y_test, y_pred)

def grid_search(X, y, transform_func, col_names):
    print(f'\nColumn name: {col_names}')

    results = {}

    for name, func in transform_func.items():
        df_X = X.copy()
        df_X = func(df_X[col_names])
        df_X.drop(col_names, axis=1)

        X_train, X_test, y_train, y_test = train_test_split(
            df_X, y, test_size=0.2, random_state=0
        )

        model = sm.OLS(y_train, X_train).fit()
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        results[name] = (r2, rmse)
        print(f'{name}: R-squared={r2:.4f} | RMSE={rmse:.4f}')
    return results

trans_func = {
    'sqrt': lambda x: np.sqrt(x),
    'inv': lambda x: 1 / x,
    'neg_inv': lambda x: -1 / x,
    'sqr': lambda x: x * x,
    'log': lambda x: np.log(x),
    'neg_log': lambda x: -np.log(x),
    'exp': lambda x: np.exp(x),
    'neg_exp': lambda x: np.exp(-x),
}

grid_search(X, y, trans_func, feature_cols)