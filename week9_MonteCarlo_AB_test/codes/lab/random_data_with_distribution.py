from matplotlib import pyplot as plt
from scipy.stats import genlogistic

NUM_SIMS = 500
loc = 20.30555
scale = 9.717675

random_samples = genlogistic.rvs(c=1, size=NUM_SIMS, loc=loc, scale=scale)

# Get samples greater than 0 since age cannot be negative
truncated_samples = [i for i in random_samples if i >= 0]

plt.hist(truncated_samples, bins=80)
plt.title("Randomly generated samples with rvs()")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()