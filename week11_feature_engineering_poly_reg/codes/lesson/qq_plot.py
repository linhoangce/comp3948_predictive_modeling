import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 3 * X.squeeze() + 5 + np.random.randn(100) * 2

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

residuals = y_test - y_pred

# Predicted vs Rediduals plot
plt.figure(figsize=(14, 7))
plt.rcParams['font.size'] = 17
plt.subplot(2, 2, 1)
plt.scatter(y_pred, residuals, color='blue', alpha=0.7)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Predicted vs Residuals')
plt.xlabel('Precited')
plt.ylabel('Residuals')

# QQ Plot
plt.subplot(2, 2, 2)
stats.probplot(residuals, dist='norm', plot=plt)
plt.title('Q-Q Plot')

plt.subplot(2, 2, 3)
plt.hist(residuals)
plt.title('Residuals')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()