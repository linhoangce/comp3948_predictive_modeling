import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def show_X_y_plot(X, y, x_title, title):
    plt.figure(figsize=(8, 4))
    plt.plot(X, y, color='blue')
    plt.title(title)
    plt.xlabel(x_title)
    plt.ylabel('y')
    plt.show()

def show_residual_plot_and_rmse(X, y, predictions):
    xmax = max(X)
    xmin = min(X)
    residuals = y - predictions

    plt.figure(figsize=(8, 3))
    plt.title('X and y')
    plt.plot([xmin, xmax], [0,0], '--', color='black')
    plt.title('Residuals')
    plt.scatter(X, residuals, color='red')
    plt.show()

    rmse = np.sqrt(mean_squared_error(y, predictions))
    print(f'RMSE: {rmse:.4f}')

# mock data
x = [0, 20, 50, 100,150,200,250,300,350,400]
y = [0.0, 4.47, 7.07, 10.0, 12.24, 14.14, 15.81, 17.32, 18.70, 20.0]

# show_X_y_plot(x, y, 'x', 'X and y')

df_X = pd.DataFrame({'x': x})
df_y = pd.DataFrame({'y': y})

X_train, X_test, y_train, y_test = train_test_split(
    df_X, df_y, test_size=0.3, random_state=42
)

X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

model = sm.OLS(y_train, X_train).fit()

y_pred = model.predict(X_test)

print(model.summary())
show_residual_plot_and_rmse(list(X_test['x']), list(y_test['y']), y_pred)

### SQUARE ROOT TRANSFORMATION
X_train['xt'] = np.sqrt(X_train['x'])
X_test['xt'] = np.sqrt(X_test['x'])

show_X_y_plot(X_train['xt'], y_train, 'x', 'y=sqrt(X)')

model_t = sm.OLS(y_train, X_train[['const', 'xt']]).fit()
y_pred_t = model_t.predict(X_test[['const', 'xt']])

print(model_t.summary())
show_residual_plot_and_rmse(X_test['xt'], y_test['y'], y_pred_t)