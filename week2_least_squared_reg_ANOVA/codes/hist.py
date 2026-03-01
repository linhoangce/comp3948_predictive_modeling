import pandas as pd
import matplotlib.pyplot as plt

path = r"C:\Users\linho\Desktop\CST\term3\pred_analytics\data\bodyfat.txt"

df = pd.read_csv(path, sep='\t')

plt.subplots(nrows=2, ncols=2, figsize=(14, 8))

plt.subplot(2, 2, 1)
plt.hist(df['Pct.BF'], bins=10)
plt.xlabel('Pct.BF')

plt.subplot(2, 2, 2)
plt.hist(df['Age'])
plt.xlabel('Age')

plt.subplot(2, 2, 3)
plt.hist(df['Weight'])
plt.xlabel('Weight')

plt.subplot(2, 2, 4)
plt.hist(df['Height'])
plt.xlabel('Height')

plt.show()