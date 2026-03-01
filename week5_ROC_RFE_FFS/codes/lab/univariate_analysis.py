import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
from sklearn.model_selection import train_test_split
from scipy import stats

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

path = r"C:\Users\linho\Desktop\CST\term3\pred_analytics\data\USA_Housing.csv"

df = pd.read_csv(path)

# show statistics, boxplot, extreme values and
# returns DataFrame row indices where outliers exist
def view_and_get_outliers(df, col_name, threshold, plt):
    # show basic statistics
    df_sub = df[[col_name]]
    print("===== Statistics for " + col_name)
    print(df_sub.describe())

    # show boxplot
    df_sub.boxplot(column=[col_name])
    plt.title(col_name)
    plt.show()

    # `abs()` returns both high and low values
    z = np.abs(stats.zscore(df_sub))
    row_col_array = np.where(z > threshold)
    row_indices = row_col_array[0]

    print("\nOutliers row indices for " + col_name)
    print(row_indices)

    # show filtered and sorted DataFrame with outliers
    df_sub = df.iloc[row_indices]
    df_sorted = df_sub.sort_values([col_name], ascending=True)
    print("\nDataFrame rows containing outliers for " + col_name)
    print(df_sorted)

    return row_indices

def view_and_get_outliers_by_percentile(df, col_name, lower_per, upper_per, plt):
    df_sub = df[[col_name]]
    print("======= Statistics for =======" + col_name)
    print(df_sub.describe())

    df_sub.boxplot(column=[col_name])
    plt.title(col_name)
    plt.show()

    up = df[col_name].quantile(upper_per)
    lp = df[col_name].quantile(lower_per)
    outlier_df = df[(df[col_name] > up) | (df[col_name] < lp)]

    df_sorted = outlier_df.sort_values([col_name], ascending=True)
    print("\n\nDataFrame rows containing outliers for " + col_name)
    print(df_sorted)

    return lp, up

# lower_percentile = 0.00025
# upper_percentile = 0.99925

# lp, up = view_and_get_outliers_by_percentile(df, "Price",
#                                              lower_percentile,
#                                              upper_percentile,
#                                              plt)

THRESHOLD = 3
# price_outliers_rows = view_and_get_outliers(df, 'Avg. Area Income', THRESHOLD, plt)

path = r"C:\Users\linho\Desktop\CST\term3\pred_analytics\data\babysamp-98.txt"

df_babysamp = pd.read_csv(path, sep='\t')
# weight_outliers_rows = view_and_get_outliers(df_babysamp, 'weight', 2.33, plt)

lower_p = 0.002
upper_p = 0.98
#
# lp, up = view_and_get_outliers_by_percentile(df_babysamp, 'gestation',
#                                              lower_p, upper_p,
#                                              plt)

LOWER_PERCENTILE = 0.00025
UPPER_PERCENTILE = 0.99925

lp, up = view_and_get_outliers_by_percentile(df[['Avg. Area Income', 'Avg. Area House Age',
                                                 'Avg. Area Number of Rooms',
                                                 'Avg. Area Number of Bedrooms', "Area Population",
                                                 'Price']],
                                             'Price',
                                             LOWER_PERCENTILE,
                                             UPPER_PERCENTILE, plt)

print("Lower " + str(LOWER_PERCENTILE) + " percentile limit: " + str(lp))
print("Upper " + str(UPPER_PERCENTILE) + " percentile limit: " + str(up))
print("Dataframe length: " + str(len(df)))

df_filtered = df[(df["Price"] > lp) & (df["Price"] < up)]
print("\nDataframe length after filtering: " + str(len(df_filtered)))
print(df['Price'].min())
print(df['Price'].max())


# clip data range
df_adjusted = df['Price'].clip(1.520719e+05, 2.294648e+06)
print(df_adjusted)
df["PriceClipped"] = df_adjusted

price_outliers_rows = view_and_get_outliers(df[['Avg. Area Income', 'Avg. Area House Age',
                                                 'Avg. Area Number of Rooms',
                                                 'Avg. Area Number of Bedrooms', "Area Population",
                                                 'Price', 'PriceClipped']],
                                            'Price', 3, plt)