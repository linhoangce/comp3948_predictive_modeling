import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import seaborn
from scipy.stats import kstest

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

dataset = seaborn.load_dataset('titanic')
samples = dataset.age.dropna()

def run_kolmogorov_smirnov_test(dist, loc, arg, scale, samples):
    d, pvalue = kstest(samples,
                       lambda x: dist.cdf(x, loc=loc, scale=scale, *arg),
                       alternative='two-sided')
    print(f"D value: {d}")
    print(f"p value: {pvalue}")
    return {'dist': dist.name,
            'D value': d,
            'p value': pvalue,
            'loc': loc,
            'scale': scale,
            'arg': arg}

def fit_and_plot(dist, samples, df):
    print(f"\n*** {dist.name} ***")
    params = dist.fit(samples)
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # generate 'x' values between 0 and 80
    x = np.linspace(0, 80, 80)

    # run test to see if generated data aligns properly to sample data
    dist_p_and_d = run_kolmogorov_smirnov_test(dist, loc, arg, scale, samples)
    df = df._append(dist_p_and_d, ignore_index=True)

    _, ax = plt.subplots(1, 1)
    plt.hist(samples, bins=80, range=(0, 80))
    ax2 = ax.twinx()
    ax2.plot(x, dist.pdf(x, loc=loc, scale=scale, *arg), 'r-', lw=2)
    plt.title(dist.name)
    plt.show()
    return df

distribution = scipy.stats.norm

df_dist = pd.DataFrame()
df_dist = fit_and_plot(distribution, samples, df_dist)

age_mean = np.mean(samples)
age_sd = np.std(samples)
age_min = np.min(samples)
age_max = np.max(samples)

df_dist.sort_values(by=['D value'], inplace=True)
print(df_dist)

NUM_SIMS = len(samples)

random_num = scipy.stats.norm.rvs`(loc=age_mean, scale=age_sd, size=NUM_SIMS)
plt.xlim([age_min, age_max])
plt.hist(random_num, bins=80)
plt.show()