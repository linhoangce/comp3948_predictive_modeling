import pandas as pd
import matplotlib.pyplot as plt

path = r"C:\Users\linho\Desktop\CST\term3\pred_analytics\data\babysamp-98.txt"

df_baby = pd.read_csv(path, sep='\t')

def baby_sample_plot(df):
    plt.subplots(nrows=1, ncols=3, figsize=(14, 7))
    plt.xticks([])

    plt.subplot(1, 3, 1)
    plotplot = df.boxplot(column=['MomAge', 'DadAge'])

    plt.subplot(1, 3, 2)
    boxplot = df.boxplot(column=["MomEduc"])

    plt.subplot(1, 3, 3)
    boxplot = df.boxplot(column=['weight'])

    plt.show()

def body_fat_plot(df, features):
    plt.subplots(nrows=1, ncols=len(features), figsize=(14, 7))
    plt.xticks([])

    for i in range(len(features)):
        plt.subplot(1, len(features), i+1)
        df.boxplot(column=[features[i]])

    plt.show()

path_body_fat = r"C:\Users\linho\Desktop\CST\term3\pred_analytics\data\bodyfat.txt"
df_body_fat = pd.read_csv(path_body_fat, sep='\t')
body_fat_plot(df_body_fat, ['Pct.BF', 'Age', 'Weight', 'Height'])
