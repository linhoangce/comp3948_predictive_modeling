from statsmodels.stats.power import TTestIndPower
from scipy import stats
import numpy as np

effect = -0.85
alpha = 0.05 # enables 95% confidence for the two tail test
power = 0.8  # 1 - prob of type II error / limits possibility of type II error to 20%

analysis = TTestIndPower()

num_samples_needed = analysis.solve_power(effect, power=power, alpha=alpha)
print(num_samples_needed)

old_menu_sales = [101, 110, 115, 136, 140, 108,
                  80, 89, 131, 98, 121, 117, 106,
                  141, 119, 153, 184, 127, 103,
                  139, 130, 146, 130]

new_menu_sales = [158, 145, 134, 130, 113, 135,
                  163, 128, 166, 154, 143, 147, 132,
                  132, 136, 99, 163, 106, 143, 168, 136,
                  123, 159]

test_result = stats.ttest_ind(new_menu_sales,
                              old_menu_sales, equal_var=False)

print(f"Hypothesis test p-value: {test_result}")
print(f"New sales mean: {np.mean(new_menu_sales)}")
print(f"New sales std: {np.std(new_menu_sales)}")