import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

path = r"C:\Users\linho\Desktop\CST\term3\pred_analytics\data\babysamp-98.txt"

df = pd.read_csv(path, sep='\t')

sub_df = df[['MomAge', 'gestation', 'weight']]

scatter_matrix(sub_df, figsize=(12, 12))
plt.show()