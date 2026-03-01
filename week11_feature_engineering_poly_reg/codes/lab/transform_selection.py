import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from lab.plot import show_residual_rmse

x = [0.1, 0.8, .99, 1.4, 2, 2.1, 2.4, 3.8]
y = [-2.303, -0.2231, -0.010050, 0.3364, 0.6931,
      0.74194, 0.87547, 1.3350]

df_X = pd.DataFrame({'X': x})
df_y = pd.DataFrame({'y': y})
df_X = sm.add_constant(df_X)

model = sm.OLS(y, df_X).fit()
y_pred = model.predict(df_X)
print(model.summary())
show_residual_rmse(x, y, y_pred)

X_train, X_test, y_train, y_test = train_test_split(
    df_X, df_y, test_size=0.3, random_state=42
)

transform_func = {
    'sqrt': lambda x: np.sqrt(x),
    'inv': lambda x: 1 / x,
    'neg_inv': lambda x: -1 / x,
    'sqr': lambda x: x*x,
    'log': lambda x: np.log(x),
    'neg_log': lambda x: -np.log(x),
    'exp': lambda x : np.exp(x),
    'neg_exp': lambda x: np.exp(-x)
}

def grid_search(df_X, y, transform_func):
    df_transform = pd.DataFrame()
    for name, func in transform_func.items():
        df_X['xt'] = func(df_X['X'])
        model = sm.OLS(y, df_X[['const', 'xt']]).fit()
        pred = model.predict(df_X[['const', 'xt']])
        rmse = np.sqrt(mean_squared_error(y, pred))
        df_transform = df_transform._append({
            'transform': name,
            'rmse': rmse
        }, ignore_index=True)

    df_transform.sort_values(by=['rmse'], inplace=True)
    print(df_transform)
    best_transform = df_transform.iloc[0]['transform']
    return best_transform

def evaluate(X_train, X_test, y_train, y_test, tran):
    X_train['xt'] = transform_func[tran](X_train['X'])
    model = sm.OLS(y_train, X_train[['const', 'xt']]).fit()
    X_test['xt'] = transform_func[tran](X_test['X'])
    y_pred = model.predict(X_test[['const', 'xt']])
    print(model.summary())
    show_residual_rmse(X_test['xt'], y_test.squeeze(), y_pred)
    plt.show()

best_transform = grid_search(X_train, y_train, transform_func)
evaluate(X_train, X_test, y_train, y_test, best_transform)
