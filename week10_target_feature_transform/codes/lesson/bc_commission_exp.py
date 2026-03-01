import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from sklearn.model_selection import train_test_split

path = r"C:\Users\linho\Desktop\CST\term3\pred_analytics\data\commission_vs_experience.csv"

df = pd.read_csv(path)
X = df[['years_experience']]
y = df['commission']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model_plain = LinearRegression().fit(X_train, y_train)
y_pred_plain = model_plain.predict(X_test)
r2_plain = r2_score(y_test, y_pred_plain)
rmse_plain = np.sqrt(mean_squared_error(y_test, y_pred_plain))

# Boxcox transformation
OFFSET = 1e-6

y_min = y_train.min()
if y_min <= 0:
    y_shift = y_min - OFFSET
else:
    y_shift = 0

y_train_shifted = y_train - y_shift

y_train_transformed, lam = boxcox(y_train_shifted)
model_bc = LinearRegression().fit(X_train, y_train_transformed)

# Compare distributions
plt.hist(y_train.squeeze(), bins=15, label='y_train', color='blue')
plt.hist(y_train_transformed, bins=15, label='y_train_transformed', color='yellow')
plt.xlabel('commission')
plt.legend()
plt.title('Commission Rate Frequency')
plt.tight_layout()
plt.show()

# predict and inverse-transform
y_pred_bc = model_bc.predict(X_test)

y_pred_bc = np.maximum(y_pred_bc, -1/lam + OFFSET)
y_pred_boxcox = inv_boxcox(y_pred_bc, lam) + y_min - OFFSET

r2_bc = r2_score(y_test, y_pred_boxcox)
rmse_bc = np.sqrt(mean_squared_error(y_test, y_pred_boxcox))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_test, y_test, label='Actual')
plt.scatter(X_test, y_pred_plain, color='yellow', label='Pred')
plt.title(f'Without Box-Cox\n$R^2$={r2_plain:.3f}, RMSE={rmse_plain:.3f}')
plt.xlabel('years experience')
plt.ylabel('commission')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(X_test, y_test, label='Actual')
plt.scatter(X_test, y_pred_boxcox, color='yellow', label='Pred (BC)')
plt.title(f'Box-Cox Transformation\n$R^2$={r2_bc:.3f}. RMSE={rmse_bc:.3f}')
plt.xlabel('years experience')
plt.ylabel('commission')
plt.legend()

residuals_before = y_test - y_pred_plain
residuals_after = y_test - y_pred_bc

plt.figure(figsize=(14, 7))
plt.subplot(2, 2, 1)
stats.probplot(y_train, dist='norm', plot=plt)
plt.title('QQ Plot of y_train (before)')
plt.legend()

plt.subplot(2, 2, 2)
stats.probplot(y_train_transformed, dist='norm', plot=plt)
plt.title('QQ Plot of  y_train (after)')
plt.legend()

plt.subplot(2, 2, 3)
plt.scatter(X_test, y_pred_plain, label='Prediction', color='blue')
plt.scatter(X_test, residuals_before, label='Residuals', color='red')
plt.title('Predictions vs Residuals (before)')
plt.legend()

plt.subplot(2, 2, 4)
plt.scatter(X_test, y_pred_boxcox,label='Prediction', color='blue')
plt.scatter(X_test, residuals_after, label='Residual', color='red')
stats.probplot(y_train_transformed, dist='norm', plot=plt)
plt.title('Predictions vs Residuals (after)')
plt.legend()

plt.tight_layout()
plt.show()