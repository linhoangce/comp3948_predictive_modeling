import pandas as pd
import matplotlib.pyplot
import numpy as np
import scipy.stats
import seaborn
from matplotlib import pyplot as plt
from scipy.stats import kstest

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

dataset = seaborn.load_dataset('titanic')

samples = dataset.age.dropna()

# High p-values are preferred and low D scores (closer to 0)
# are preferred
def run_kolmogorov_smirnov_test(dist, loc, arg, scale, samples):
    d, pvalue = kstest(samples.tolist(),
                       lambda x: dist.cdf(x, loc=loc, scale=scale, *arg),
                       alternative="two-sided")
    print(f"D Value: {d}")
    print(f"P value: {pvalue}")
    dict = {"dist": dist.name,
            "D value": d,
            "p value": pvalue,
            "loc": loc,
            "scale": scale,
            "arg": arg}
    return dict

def fit_and_plot(dist, samples, df):
    print(f"\n** {dist.name} ***")

    # fit the distribution as best as possible to existing data
    params = dist.fit(samples)
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # generate 'x' values between 0 and 80
    x = np.linspace(0, 80, 80)

    # run test to see if generated data aligns properly
    # to see the sample data
    dist_d_and_p = run_kolmogorov_smirnov_test(dist, loc, arg, scale, samples)
    df = df._append(dist_d_and_p, ignore_index=True)

    # plot the test and actual values together
    _, ax = plt.subplots(1, 1)
    plt.hist(samples, bins=80, range=(0, 80))
    ax2 = ax.twinx()
    ax2.plot(x, dist.pdf(x, loc=loc, scale=scale, *arg),
             '-', color='r', lw=2)
    plt.title(dist.name)
    plt.show()
    return df

distributions = [
    scipy.stats.norm,
    scipy.stats.gamma,
    scipy.stats.chi2,
    scipy.stats.wald,
    scipy.stats.uniform,
    scipy.stats.norm,
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
    scipy.stats.genlogistic]

df_dist = pd.DataFrame()

# grid search the continuous distribution
df_dist = {fit_and_plot(dist, samples, df_dist)
           for dist in distributions}
df_dist.sort_values(by=['D value'], inplace=True)
print(df_dist)