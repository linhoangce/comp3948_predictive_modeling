import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import boxcox
from scipy.special import inv_boxcox
import scipy.stats as stats

X, y = fetch_california_housing(return_X_y=True, as_frame=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model_plain = LinearRegression()
model_plain.fit(X_train, y_train)
y_pred = model_plain.predict(X_test)

r2_plain = r2_score(y_test, y_pred)
rmse_plain = np.sqrt(mean_squared_error(y_test, y_pred))

# Box Cox Transformation
OFFSET = 1e-6

y_min = y_train.min()
if y_min <= 0:
    y_shift = y_min - OFFSET
else:
    y_shift = 0

y_train_shift = y_train - y_shift

y_train_transformed, lambda_val = boxcox(y_train_shift)

model_bc = LinearRegression().fit(X_train, y_train_transformed)
y_pred_bc = model_bc.predict(X_test)
y_pred_original = inv_boxcox(y_pred_bc, lambda_val) + y_shift

r2_bc = r2_score(y_test, y_pred_original)
rmse_bc = np.sqrt(mean_squared_error(y_test, y_pred_original))

print(f"R2 BC: {r2_bc:.4f}")
plt.subplots(1, 2, figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.rcParams['font.size'] = 20

plt.scatter(y_test, y_pred, s=5, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color='red', label='Perfect prediction')
plt.xlabel('Actual')
plt.ylabel('Prediction')
plt.title(f'Without Box-Cox\n$R^2$={r2_plain:.2f}, RMSE={rmse_plain:.2f}')

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_original, s=5, color='orange')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color='red')
plt.xlabel('Actual')
plt.ylabel('Prediction')
plt.title(f'With Box-Cox\n $\lambda$={lambda_val:.2f}, RMSE={rmse_bc:.2f}')

residuals_before = y_test - y_pred
residuals_after = y_test - y_pred_original

plt.subplots(2, 2, figsize=(14, 7))

plt.subplot(2, 2, 1)
plt.scatter(np.arange(len(residuals_before)), residuals_before)
plt.title("Residuals (before)")

plt.subplot(2, 2, 2)
stats.probplot(residuals_before, dist='norm', plot=plt)
plt.title("QQ Plot of residuals (before)")

plt.subplot(2, 2, 3)
plt.scatter(np.arange(len(residuals_before)), residuals_after)
plt.title('Residuals (after)')

plt.subplot(2, 2, 4)
stats.probplot(residuals_after, dist='norm', plot=plt)
plt.title("QQ Plot of residuals (after)")

plt.subplots(1, 2, figsize=(14, 7))
plt.subplot(1, 2, 1)
plt.hist(y_train)
plt.title("y_train")

plt.subplot(1, 2, 2)
plt.hist(y_train_transformed)
plt.title("y_train transformed")
plt.tight_layout()
plt.show()