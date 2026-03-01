import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

from plot import show_X_y
from plot import show_residual_rmse

# Section A: Define the raw data.
x = [0.2, 0.5, 0.7, 0.9,1,2,3,4,5,6]
y = [-5.0, -2.0, -1.43, -1.2, -1.0, -0.5, -0.34, -0.25, -0.2, -0.16]

# show_X_y(x, y, 'x', 'Before')

df_X = pd.DataFrame({'x': x})
df_y = pd.DataFrame({'y': y})

X_train, X_test, y_train, y_test = train_test_split(
    df_X, df_y, test_size=0.3, random_state=42
)

X_train['xt'] = 1 / -X_train['x']
X_test['xt'] = 1 / -X_test['x']

X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

model = sm.OLS(y_train, X_train).fit()
model_t = sm.OLS(y_train, X_train[['const', 'xt']]).fit()

y_pred = model.predict(X_test)
y_pred_t = model_t.predict(X_test[['const', 'xt']])

plt.subplots(1, 2, figsize=(14, 7))
plt.subplot(1, 2, 1)
show_X_y(x, y, 'Original', xlab='X')
plt.subplot(1, 2, 2)
show_residual_rmse(X_test['x'], y_test['y'], y_pred)

plt.subplots(1, 2, figsize=(14, 7))
plt.subplot(1, 2, 1)
show_X_y(X_train['xt'], y_train, 'Neg Inv Transform', xlab='X')
plt.subplot(1, 2, 2)
show_residual_rmse(X_test['xt'], y_test['y'], y_pred_t)
plt.show()