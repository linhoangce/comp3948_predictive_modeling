import math
import numpy as np
from statsmodels.stats.power import TTestIndPower
from scipy import stats

mean_a = 10
mean_b = 5
std = 4.5

cohen_val = (mean_a - mean_b) / (np.sqrt((std**2 + std**2) / 2))
print(cohen_val)

alpha = 0.05
power = 0.95

analysis = TTestIndPower()

test_days_needed = analysis.solve_power(cohen_val, power=power, alpha=alpha)
print(f"Number of test days: {math.ceil(test_days_needed)}")

before = [10, 11, 6, 18, 11, 9, 13, 9, 3, 12, 3, 13, 14, 4, 12,
          8, 18, 17, 15, 18, 6, 1, 13, 9, 11, 15, 11, 7, 12, 14]
after = [ 5, 7, 3, 12,  0, 7,  0,  8,  9,  5,  5,  2,  2,  2,  4,
          6,  6,  7,  1,  6, 10,  1,  0,  2, 2,  4,  5,  1,  4,  3]

test_result = stats.ttest_ind(before, after, equal_var=False)

print(f"Hypothesis test summary: {test_result}")
print(f"New Covid cases mean: {np.mean(after)}")
print(f"New Covid cases std: {np.std(after)}")