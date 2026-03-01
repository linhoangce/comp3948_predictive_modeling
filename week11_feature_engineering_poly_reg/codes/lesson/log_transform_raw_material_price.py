import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
path = r"C:\Users\linho\Desktop\CST\term3\pred_analytics\data\raw_material_prices.csv"

df = pd.read_csv(path)
X = df[['feature_strength', 'market_score']]
y = df['price']
print(y.describe())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25
)

offset = 1e-6
y_min = y.min()
if y_min <= 0:
    y_shift = y_min - offset
else:
    y_shift = 0

y_train_shifted = y_train - y_shift
y_train_logged = np.log(y_train_shifted)

model_normal = LinearRegression().fit(X_train, y_train)
model_log = LinearRegression().fit(X_train, y_train_logged)

y_pred = model_normal.predict(X_test)
y_pred_log = model_log.predict(X_test)
y_pred_orig = np.exp(y_pred_log) + offset

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
rmse_log = np.sqrt(mean_squared_error(y_test, y_pred_orig))

print(f"RMSE\nNormal={rmse:.4f} | Log={rmse_log:.4f}")

plt.subplots(1, 2, figsize=(14, 7))
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color='red')
plt.title('Prediction vs Actual')
plt.xlabel('Actual')
plt.ylabel('Prediction')

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_orig)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color='red')
plt.title('Prediction vs Actual (Log Transform)')
plt.xlabel('Actual')
plt.ylabel('Prediction')

plt.tight_layout()
plt.show()