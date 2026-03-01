from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from scipy.stats import boxcox
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
# low data
X, y = fetch_california_housing(return_X_y=True, as_frame=True)
# y = y.values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

y_min = y_train.min()
print(f"y min {y_min}")
# Shift to make data positive. Boxcox requires that this be positive.
if y_min <= 0:
    y_shift = y_min - 0.000001
else:
    y_shift = 0

y_train_shifted = y_train - y_shift

# Box cox transform
y_train_bc, lam = boxcox(y_train_shifted)
print(f'lambda value: {lam}')

plt.hist(y_train, color='blue', bins=15, label='train')
plt.hist(y_train_bc, color='orange', bins=15, alpha=0.5, label='boxcox')
plt.legend(prop={'family': 'serif', 'size': 20})
plt.show()