from scipy.stats import norm
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize
import numpy as np

# Ratio of 1 says both samples have same count
# Ratio of 0.5 says B has half the samples of A
def min_n_two_prop(p1, p2, alpha=0.05, power=0.8, ratio=1.0):
    # Cohen's effect says how big difference is between two groups in stardard units
    eff = proportion_effectsize(p1, p2) # Cohen's h for proportions
    analysis = NormalIndPower()

    # Alpha split over two tails so 0.025 at each tail
    n_a = analysis.solve_power(effect_size=eff, alpha=alpha, power=power,
                               ratio=ratio, alternative='two-sided')
    return int(np.ceil(n_a))

# Detect lift from 9% to 12% with 5% alpha, 80% power
p1 = 0.09; p2 = 0.12
nA = min_n_two_prop(p1, p2, alpha=0.05, power=0.8, ratio=1.0)
nB = nA # equal size
print(f"Required per group: A={nA}, B={nA}, total={nA+nB}")


TOTAL_SAMPLES = 2000

# Generate some fake data
np.random.seed(41)
channelA = np.random.binomial(1, 0.10, TOTAL_SAMPLES)
channelB = np.random.binomial(1, 0.13, TOTAL_SAMPLES)
# print(f"channelA: {channelA}, channelB: {channelB}")

# Get stats for both channels
xA, nA = channelA.sum(), len(channelA)
xB, nB = channelB.sum(), len(channelB)
pA, pB = xA/nA, xB/nB
p_pool = (xA + xB) / (nA + nB)

# Get standard error
se = np.sqrt(p_pool * (1 - p_pool) * (1/nA + 1/nB))

# Calculate z
z_stat = (pB - pA) / se

# Note we are getting the combined alphas at both ends.
# So multiply by 2.
# Note there will be some slight rounding differences.

p_val = 2 * (1 - norm.cdf(abs(z_stat)))

print(f"pA={pA:.3f}, pB={pB:.3f}, pooled={p_pool:.3f}")
print(f"z={z_stat:.2f}, p={p_val:.4f}")

