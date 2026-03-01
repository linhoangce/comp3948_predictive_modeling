import pandas as pd
import numpy as np

path = r"C:\Users\linho\Desktop\CST\term3\pred_analytics\data\babysamp-98.txt"

df = pd.read_table(path, sep='\t')
df_na = df['DadAge'].isna()
print(df_na)
imp = df['DadAge'].mean()
print(imp)

df_filled = df.copy()
df_filled['imp_DadAge'] = df['DadAge']
# print(df_filled.head())
df_filled.loc[df_na, 'imp_DadAge'] = imp

# print(df.head())
# print(df_filled.head())

df_filled['m_DadAge'] = df_na.astype(int)
print(df_filled.head())

