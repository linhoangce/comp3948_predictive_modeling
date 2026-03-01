import numpy as np
import pandas as pd
import scipy
from matplotlib import pyplot as plt
from scipy.stats import kstest

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

path = r"C:\Users\linho\Desktop\CST\term3\pred_analytics\data\drugSales.csv"

df = pd.read_csv(path)
samples = np.array(df[['value']])
print(samples)

# prefer high p-value and low D score
def run_kolmogorov_smirnov_test(dist, loc, arg, scale, samples):
    d, pvalue = kstest(samples.tolist(),
                       lambda x: dist.cdf(x, loc=loc, scale=scale, *arg),
                       alternative='two-sided')
    print(f'D value: {d}')
    print(f'p value: {pvalue}')
    return {'dist': dist.name,
            'D score': d,
            'p value': pvalue,
            'loc': loc,
            'scale': scale,
            'arg': arg}

def fit_and_plot(dist, samples, df):
    print(f'\n*** {dist.name} ***')

    # fit and find best matched distribution
    params = dist.fit(samples)
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    dist_d_and_p = run_kolmogorov_smirnov_test(dist, loc, arg, scale, samples)
    df = df._append(dist_d_and_p, ignore_index=True)

    x_cdf = np.linspace(min(samples), max(samples), len(samples))
    x_pdf = np.linspace(0, len(samples), len(samples))
    sorted_samples = np.sort(samples.flatten())
    empirical_cdf = np.arange(1, len(sorted_samples) + 1) / len(sorted_samples)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # plot CDF
    ax[0].plot(sorted_samples, empirical_cdf, marker='.',
             linestyle='none', label='Empirical CDF')
    ax[0].plot(x_cdf, dist.cdf(x_cdf, loc=loc, scale=scale, *arg),
             'r-', lw=2, label=f'{dist.name} CDF')
    ax[0].set_title(f'Cumulative Distribution Function - {dist.name}')
    ax[0].set_xlabel('value')
    ax[0].set_ylabel('Cumulative Probability')
    ax[0].legend()
    ax[0].grid(True)

    # plot test and actual values
    ax[1].hist(samples, bins=len(samples), density=True, range=(0, len(samples)))
    # # ax[1] = ax[1].twinx()
    ax[1].plot(x_pdf, dist.pdf(x_pdf, loc=loc, scale=scale, *arg),
             'r-', lw=2, label='Fitted PDF')
    ax[1].set_xlabel('Value')
    ax[1].set_ylabel('Density')
    ax[1].set_title(f"PDF vs Samples - {dist.name}")
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()
    return df

distributions = [
    scipy.stats.norm,
    scipy.stats.gamma,
    scipy.stats.chi2,
    scipy.stats.wald,
    scipy.stats.uniform,
    scipy.stats.t,
    scipy.stats.chi,
    scipy.stats.dgamma,
    scipy.stats.ncx2,
    scipy.stats.burr,
    scipy.stats.bradford,
    scipy.stats.crystalball,
    scipy.stats.exponnorm,
    scipy.stats.pearson3,
    scipy.stats.exponpow,
    scipy.stats.invgauss,
    scipy.stats.argus,
    scipy.stats.gennorm,
    scipy.stats.expon,
    scipy.stats.gengamma,
    scipy.stats.genextreme,
    scipy.stats.genexpon,
    scipy.stats.genlogistic
]

df_dist = pd.DataFrame()
df_list = [fit_and_plot(dist, samples, df_dist) for dist in distributions]
df_dist = pd.concat(df_list, ignore_index=True)
df_dist.sort_values(by=['D score'], inplace=True)
print(df_dist)