import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

from plot import show_X_y
from plot import show_residual_rmse

# Section A: Define the raw data.
x = [0.01, 0.2, 0.5, 0.7, 0.9,1,2,3,4,5,6,10,11,12,13,14,15,16,17,18,19,20]
y = [-4.61, -1.61, -0.693, -0.357, -0.1054, 0.0, 0.6931, 1.099, 1.3862, 1.6094,
     1.792, 2.303, 2.398, 2.48491, 2.56, 2.6390, 2.71, 2.772, 2.83, 2.89,
     2.94, 2.99]

df_X = pd.DataFrame({'x': x})
df_y = pd.DataFrame({'y': y})

X_train, X_test, y_train, y_test = train_test_split(
    df_X, df_y, test_size=0.3, random_state=42
)

X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

model = sm.OLS(y_train, X_train).fit()
y_pred = model.predict(X_test)

# Log Transform
X_train['xt'] = np.log(X_train['x'])
X_test['xt'] = np.log(X_test['x'])

model_t = sm.OLS(y_train, X_train[['const', 'xt']]).fit()
y_pred_t = model_t.predict(X_test[['const', 'xt']])

plt.subplots(1, 2, figsize=(14, 7))
plt.subplot(1, 2, 1)
show_X_y(x, y, 'Original', xlab='X')
plt.subplot(1, 2, 2)
show_residual_rmse(X_test['x'], y_test['y'], y_pred)

plt.subplots(1, 2, figsize=(14, 7))
plt.subplot(1, 2, 1)
show_X_y(X_train['xt'], y_train, 'y=log(X)', xlab='X')
plt.subplot(1, 2, 2)
show_residual_rmse(X_test['xt'], y_test['y'], y_pred_t)
plt.show()