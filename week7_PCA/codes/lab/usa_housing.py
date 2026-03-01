import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

path = r"C:\Users\linho\Desktop\CST\term3\pred_analytics\data\USA_Housing.csv"

df = pd.read_csv(path)
df2 = df._get_numeric_data()

X = df2.copy()
X.drop(['Price'], inplace=True, axis=1)
y = df2[['Price']]

vif = pd.DataFrame()
vif['VIF Score'] = [variance_inflation_factor(X.values, i)
                    for i in range(X.shape[1])]
vif['Feature'] = X.columns

print(f'\nOriginal Data VIF\n{vif}')

print(f'Price Stats\n{y.describe()}')
pca = PCA(0.8)
model = LinearRegression()

X_scaler = StandardScaler()
y_scaler = StandardScaler()
X_std = X_scaler.fit_transform(X)
# y_std = y_scaler.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_std, y, test_size=0.20
)

X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print(f'Principal Componenets:\n{pca.components_}')
print(f'\nExplained Variance:\n{pca.explained_variance_}')

model.fit(X_train_pca, y_train)
y_pred = model.predict(X_test_pca)
# y_pred = y_scaler.inverse_transform(y_pred)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'\nRMSE: {rmse:.4f}')
print(f'\nModel Coefficient\n{model.coef_}')
print(f'\nModel Intercept\n{model.intercept_}')
print(f'\nR2 score: {r2:.4f}')

vif = pd.DataFrame()
vif['VIF Factor'] = [variance_inflation_factor(X_train_pca, i)
                     for i in range(X_train_pca.shape[1])]

print(f'\nVIF Score\n{vif}')
