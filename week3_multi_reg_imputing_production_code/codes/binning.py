import pandas as pd

path = r"C:\Users\linho\Desktop\CST\term3\pred_analytics\data\babysamp-98.txt"

df = pd.read_table(path, sep='\t')

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

df['dadAgeBin'] = pd.cut(x=df['DadAge'], bins=[17, 27, 37, 48])
df['momAgeBin'] = pd.cut(x=df['MomAge'], bins=[13, 23, 33, 42])

tempDf = df[['dadAgeBin', 'momAgeBin', 'sex']]

dummyDf = pd.get_dummies(tempDf, columns=['dadAgeBin', 'momAgeBin', 'sex'], dtype=int)
df = pd.concat([df, dummyDf], axis=1)
print(df.head())