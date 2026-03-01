import numpy as np
from scipy.stats import norm
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportions_ztest, proportion_effectsize

TOTAL_SAMPLES = 1000

# generate fake data
np.random.seed(22)
channel_a = np.random.binomial(1, 0.1, TOTAL_SAMPLES)
channel_b = np.random.binomial(1, 0.14, TOTAL_SAMPLES)

a_x = channel_a.sum(); a_n = len(channel_a)
b_x = channel_b.sum(); b_n = len(channel_b)
p_a = a_x/a_n; p_b = b_x/b_n
p_pool = (a_x + b_x) / (a_n + b_n)

print(f"a_x={a_x}, b_x={b_x}")

# Calculate SE
se = np.sqrt(p_pool * (1 - p_pool) * (1/a_n + 1/b_n))

# calculate z score
z = (p_b - p_a) / se

p_val = 2 * (1 - norm.cdf(abs(z)))

print(f"p_a={p_a:.4f}, p_b={p_b:.4f}, pooled={p_pool:.4f}")
print(f'z={z:.4f}, p={p_val:.4f}')


def min_n_two_prop(p1, p2, alpha=0.05, power=0.8, ratio=1.0):
    # Cohen's effect says how big difference is between two groups in stardard units
    eff = proportion_effectsize(p1, p2) # Cohen's h for proportions
    analysis = NormalIndPower()

    # Alpha split over two tails so 0.025 at each tail
    n_a = analysis.solve_power(effect_size=eff, alpha=alpha, power=power,
                               ratio=ratio, alternative='two-sided')
    return int(np.ceil(n_a))

# Detect lift from 9% to 12% with 5% alpha, 80% power
p1 = 0.09; p2 = 0.13
nA = min_n_two_prop(p1, p2, alpha=0.05, power=0.8, ratio=1.0)
nB = nA # equal size
print(f"Required per group: A={nA}, B={nA}, total={nA+nB}")
