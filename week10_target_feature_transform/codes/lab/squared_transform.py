import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

from plot import show_X_y
from plot import show_residual_rmse

# Section A: Define the raw data.
x = [-6,-5,-4,-3,-2,-1,-0.9,-0.7,-0.5,-0.2,0,0.2,0.5,0.7,0.9,1,2,3,4,5,6]
y = [36, 25, 16, 9, 4, 1, 0.81, 0.49, 0.25, 0.04,
     0, 0.04, 0.25, 0.49, 0.81, 1, 4, 9, 16, 25, 36]


# show_X_y(x, y, 'x', 'Before')

df_X = pd.DataFrame({'x': x})
df_y = pd.DataFrame({'y': y})

X_train, X_test, y_train, y_test = train_test_split(
    df_X, df_y, test_size=0.3, random_state=42
)


X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

model = sm.OLS(y_train, X_train).fit()
y_pred = model.predict(X_test)


X_train['xt'] = X_train['x'] ** 2
X_test['xt'] = X_test['x'] ** 2

model_t = sm.OLS(y_train, X_train[['const', 'xt']]).fit()
y_pred_t = model_t.predict(X_test[['const', 'xt']])

plt.subplots(1, 2, figsize=(14, 7))
plt.subplot(1, 2, 1)
show_X_y(x, y, 'Original', xlab='X')
plt.subplot(1, 2, 2)
show_residual_rmse(X_test['x'], y_test['y'], y_pred)

plt.subplots(1, 2, figsize=(14, 7))
plt.subplot(1, 2, 1)
show_X_y(X_train['xt'], y_train, 'y=$x^2$', xlab='X')
plt.subplot(1, 2, 2)
show_residual_rmse(X_test['xt'], y_test['y'], y_pred_t)
plt.show()