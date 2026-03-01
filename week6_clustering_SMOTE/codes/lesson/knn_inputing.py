import pandas as pd
from sklearn.impute import KNNImputer

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Import data into a DataFrame.
path = r"C:\Users\linho\Desktop\CST\term3\pred_analytics\data\babysamp-98.txt"
df   = pd.read_table(path, delim_whitespace=True)
numeric_col = [ 'MomAge','DadAge','MomEduc','MomMarital','numlive',
                    'dobmm','gestation','weight','prenatalstart']

df_numeric = df[numeric_col]

# Show data types for each columns.
print("\n*** Before imputing")
print(df.describe())
print(df_numeric.head(11))

# Show summaries for objects like dates and strings.
imputer   = KNNImputer(n_neighbors=5)
df_numeric = pd.DataFrame(imputer.fit_transform(df_numeric),
                         columns = df_numeric.columns)

# Show data types for each columns.
print("\n*** After imputing")
print(df_numeric.describe())
print(df_numeric.head(11))
