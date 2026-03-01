import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import pandas as pd
from string import ascii_letters

path = r"C:\Users\linho\Desktop\CST\term3\pred_analytics\data\babysamp-98.txt"

df = pd.read_csv(path, sep='\t')
sub_df = df[['MomAge', 'gestation', 'weight']]

# compute correlation matrix
corr = sub_df.corr()

# plot heatmap
sns.set_theme(rc={'figure.figsize': (6, 4)})
sns.heatmap(corr[['gestation']],
            linewidths=0.1, vmin=-1, vmax=1,
            cmap='YlGnBu')
plt.show()

