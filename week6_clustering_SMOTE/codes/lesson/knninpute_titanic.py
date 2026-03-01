import pandas as pd
from sklearn.impute import KNNImputer

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

path = r"C:\Users\linho\Desktop\CST\term3\pred_analytics\data\titanic_training_data.csv"

df = pd.read_csv(path)

# select only numeric columns
df_numeric = df.select_dtypes(include=['number'])
print(df_numeric.describe())

imputer = KNNImputer(n_neighbors=5)
df_numeric_imputed = pd.DataFrame(imputer.fit_transform(df_numeric),
                                  columns=df_numeric.columns)

print("\n\n\t\t=============== Imputed DataFrame ===============\n\n")
print(df_numeric_imputed.describe())