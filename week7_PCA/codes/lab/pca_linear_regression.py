import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler

path = r"C:\Users\linho\Desktop\CST\term3\pred_analytics\data\Hitters.csv"

df = pd.read_csv(path).dropna()
df.info()

dummies = pd.get_dummies(df[['League', 'Division', 'NewLeague']], dtype=int)
y = df.Salary

print(f"\nSalary Stats:\n{y.describe}")

# Drop columns with independent variable (Salary),
# and columns for which we created dummy variables
X_ = df.drop(['Salary', 'League', 'Division', 'NewLeague'], axis=1).astype('float64')

# define feature set X
X = pd.concat([X_, dummies[['League_N', 'Division_W', 'NewLeague_N']]], axis=1)

# calculate and show VIF scores for original data
vif = pd.DataFrame()
vif['VIF Factor'] = [variance_inflation_factor(X.values, i)
                     for i in range(X.shape[1])]
vif['features'] = X.columns
print(f'\nOriginal VIF Score:\n{vif}')

X_scaled = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=1
)

pca = PCA(.9)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print(f'\nPrincipal components\n{pca.components_}')
print(f'\nExplained variance:\n{pca.explained_variance_}')

model = LinearRegression()
model.fit(X_train_pca, y_train)

y_pred = model.predict(X_test_pca)

# show stats about regression prediction
mse = mean_squared_error(y_test, y_pred)
RMSE = np.sqrt(mse)
print(f'\nRMSE: {RMSE}')
print(f'\nModel Coefficients:\n{model.coef_}')
print(f'\nModel Intercept:\n{model.intercept_}')
print(f'R2 score: {r2_score(y_test, y_pred)}')

# Calculate VIF for each principal component
vif = pd.DataFrame()
vif['VIF Factor'] = [variance_inflation_factor(X_train_pca, i)
                     for i in range(X_train_pca.shape[1])]
print(f'\nPCA VIF\n{vif}')