import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from lab.plot import show_X_y, show_residual_rmse

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

path = r"C:\Users\linho\Desktop\CST\term3\pred_analytics\data\abs.csv"

df = pd.read_csv(path)

X = df[['abs(450nm)']]
y = df['ug/L']

show_X_y(X, y, 'absorbance x', 'Protein Concentration(y) and Absorbance(x)')
plt.show()

X = sm.add_constant(X)

model = sm.OLS(y, X).fit()
y_pred = model.predict(X)

print(model.summary())
pred_df = pd.DataFrame({'prediction': y_pred})
residuals = y - pred_df['prediction']
res_squared = [i**2 for i in residuals]
rmse = np.sqrt(np.sum(res_squared) / len(res_squared))
print(f'RMSE: {rmse:.4f}')

plt.scatter(X['abs(450nm)'], residuals)
plt.show()

# =======================================================
# Transform selection
# =======================================================
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

def grid_search(X, y, transform_func, col_name):
    df_transform = pd.DataFrame()
    for name, func in transform_func.items():
        X['xt'] = func(X[col_name])
        model = sm.OLS(y, X[['const', 'xt']]).fit()
        y_pred = model.predict(X[['const', 'xt']])
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        df_transform = df_transform._append({
            'transform': name,
            'rmse': rmse
        }, ignore_index=True)

    df_transform.sort_values(by=['rmse'], inplace=True)
    print(df_transform)
    return df_transform.iloc[0]['transform']

def evaluate(X_train, X_test, y_train, y_test,
             transform_func, best_transform, col_name):
    X_train['xt'] = transform_func[best_transform](X_train[col_name])
    model = sm.OLS(y_train, X_train[['const', 'xt']]).fit()
    X_test['xt'] = transform_func[best_transform](X_test[col_name])
    y_pred = model.predict(X_test[['const', 'xt']])
    print(model.summary())
    show_residual_rmse(X_test['xt'], y_test, y_pred)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3
)

best_transform = grid_search(X, y, transform_func, 'abs(450nm)')

# Evaluate based on train test split but the results
# depends closely on the splits, which is too small
# for them to be reliable
evaluate(X_train, X_test, y_train, y_test,
         transform_func, best_transform, 'abs(450nm)')
plt.show()

# Evaluate based on the whole dataset, the result produced
# here is more suitable for comparison with the initial RMSE
X['xt'] = transform_func[best_transform](X['abs(450nm)'])
model_t = sm.OLS(y, X[['const', 'xt']]).fit()
y_pred_2 = model_t.predict(X[['const', 'xt']])
show_residual_rmse(X['xt'], y, y_pred_2)
plt.show()

print(model_t.summary())