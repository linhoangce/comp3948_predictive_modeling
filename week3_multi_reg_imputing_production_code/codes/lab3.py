import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


path = r"C:\Users\linho\Desktop\CST\term3\pred_analytics\data\USA_Housing.csv"

dataset = pd.read_csv(path, encoding='ISO-8859-1', sep=',')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print(dataset.head(3))
print(dataset.describe())

# extract only numeric columns
numeric_dataset = dataset[['Avg. Area Income', 'Avg. Area House Age',
             'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms',
             'Area Population', 'Price']]

corr = numeric_dataset.corr()
corr = corr.sort_values(by=['Price'], ascending=False)

sns.set_theme(rc={'figure.figsize': (6, 4)})
sns.heatmap(corr[['Price']], annot=True,
            linewidth=0.1, vmin=-1, vmax=1, cmap='YlGnBu')

# plt.tight_layout()
# plt.show()

# extract only numeric features
X = numeric_dataset[['Avg. Area Income', 'Avg. Area House Age',
             'Avg. Area Number of Rooms',
             'Area Population']]

# add intercept
X = sm.add_constant(X)

y = numeric_dataset['Price']

# split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

model = sm.OLS(y_train, X_train).fit()
pred = model.predict(X_test)

print(model.summary())
print(f'RMSE: {np.sqrt(mean_squared_error(y_test, pred))}')

def plot_pred_vs_actual(plt, title, y_test, preds):
    plt.scatter(y_test, preds)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Predicted (Y) vs. Actual (X): ' + title)
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], 'k--')

def plot_residuals_vs_actual(plt, title, y_test, preds):
    residuals = y_test - preds
    plt.scatter(y_test, residuals, label='Residuals vs Actual')
    plt.xlabel('Actual')
    plt.ylabel('Residuals')
    plt.title('Error Residuals (Y) vs. Actual (X): ' + title)
    plt.plot([y_test.min(), y_test.max()], [0, 0], 'k--')

def plot_residuals_hist(plt, title, y_test, preds, bins):
    residuals = y_test - preds
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.hist(residuals, label='Residuals vs Actual', bins=bins)
    plt.title("Error Residuals vs Actual: " + title)
    plt.plot()

def draw_plots(title, bins, y_test, preds):
    plt.subplots(1, 3, figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plot_pred_vs_actual(plt, title, y_test, preds)
    plt.subplot(1, 3, 2)
    plot_residuals_vs_actual(plt, title, y_test, preds)
    plt.subplot(1, 3, 3)
    plot_residuals_hist(plt, title, y_test, preds, bins)
    plt.show()

BINS = 10
TITLE = 'House Prices'
draw_plots(TITLE, BINS, y_test, pred)
