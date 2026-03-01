from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from scipy.stats import boxcox
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

y = np.array([1.2, 1.5, 1.8, 2.0, 2.3, 2.5, 2.9, 3.0, 3.1, 3.3,
              4.0, 4.5, 5.2, 6.8, 8.5, 10.2, 12.0, 15.5, 20.0, 25.0])

y_min = y.min()
if y_min <= 0:
    y_shift = y_min - 1e-6
else:
    y_shift = 0

y_shifted = y - y_shift
y_transformed, lambda_val = boxcox(y_shifted)
print(f'lamba: {lambda_val}')

plt.hist(y, color='blue', label='original')
plt.hist(y_transformed, color='yellow', label='transformed')
plt.legend()
plt.show()