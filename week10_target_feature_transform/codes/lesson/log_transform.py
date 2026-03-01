import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

def get_synthetic_data():
    np.random.seed(123)

    n = 400
    x = np.linspace(0, 5, n)
    a_true, b_true, sigma = 1.0, 0.6, 0.35
    eps = np.random.randn(n) * sigma
    y = np.exp(a_true + b_true * x + eps)
    X = x.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = get_synthetic_data()

model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

def shift_y_train(y_train):
    shift = 0.0
    y_min = np.min(y_train)
    if y_min <= 0:
        shift = y_min - 1e-6

    y_train_log = y_train + shift
    return y_train_log, shift

y_train_log, shift = shift_y_train(y_train)

y_log = np.log(y_train_log)

model_log = LinearRegression().fit(X_train, y_log)
y_pred_log = model_log.predict(X_test)
y_pred_orig = np.exp(y_pred_log) + shift

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

r2_log = r2_score(y_test, y_pred_orig)
rmse_log = np.sqrt(mean_squared_error(y_test, y_pred_orig))

plt.figure(figsize=(12, 8))

plt.subplot(1, 3, 1)
plt.scatter(X_test, y_test, s=10, alpha=0.6, label='Actual')
plt.scatter(X_test, y_pred, lw=2, label=f'y: RMSE={rmse:.2f}', color='green')
plt.scatter(X_test, y_pred_orig, lw=2, label=f'y_log: RMSE={rmse_log:.2f}', color='orange')
plt.title('Model Performance')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.subplot(1, 3, 2)
plt.scatter(y_pred, y_test - y_pred, s=10, alpha=0.7)
plt.axhline(0, linestyle='--')
plt.title('Residuals Raw Y')
plt.xlabel('Fitted')
plt.ylabel('Residuals')

plt.subplot(1, 3, 3)
plt.scatter(y_pred_orig, y_test - y_pred_orig, s=10, alpha=0.7)
plt.axhline(0, linestyle='--')
plt.title('Residuals ln(y)')
plt.xlabel('Fitted')
plt.ylabel('Residuals')

plt.subplots(1, 2, figsize=(14, 7))
plt.subplot(1, 2, 1)
plt.hist(y_train)
plt.title('y_train - Before')
plt.subplot(1, 2, 2)
plt.hist(y_log)
plt.title('y_train - After')

plt.tight_layout()
plt.show()