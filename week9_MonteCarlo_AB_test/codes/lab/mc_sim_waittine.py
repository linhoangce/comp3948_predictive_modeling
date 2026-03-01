from scipy.stats import norm
import numpy as np

WAIT_AVG = 23
WAIT_SD = 7
TOTAL_HOURS = 40

sim_runs = norm.rvs(loc=WAIT_AVG, scale=WAIT_SD, size=TOTAL_HOURS)

print(np.round(sim_runs, 0))
print(f"Number of times more than 30 customers waiting: {(sim_runs > 30).sum()}")

