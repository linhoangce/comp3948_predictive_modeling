import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PATH = r"C:\Users\linho\Desktop\CST\term3\pred_analytics\data"
CSV_DATA = r'\winequality.csv'
dataset = pd.read_csv(PATH + CSV_DATA, sep=',')

# set font size
font = {"family": 'normal',
        'weight': 'bold',
        'size': 30}
plt.rc('font', **font)

# show all column
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print(dataset.head(5))
print(dataset.describe())

X = dataset[['fixed acidity', 'volatile acidity', 'citric acid',
             'residual sugar', 'chlorides', 'free sulfur dioxide',
             'total sulfur dioxide', 'density', 'pH', 'sulphates',
             'alcohol', 'quality']]

# compute the correlation matrix
corr = dataset.corr()

# sort dataframe by quality correlation
corr = corr.sort_values(by=['quality'], ascending=False)

sns.set_theme(rc={'figure.figsize': (6, 4)})
sns.heatmap(corr[['quality']], annot=True,
            linewidth=0.1, vmin=-1, vmax=1,
            cmap='YlGnBu')

plt.tight_layout() # prevent label truncation
# plt.show()

print('\nFrequency for Wine quality')
print(dataset['quality'].value_counts(ascending=True))