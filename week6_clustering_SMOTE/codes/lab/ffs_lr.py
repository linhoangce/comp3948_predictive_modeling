import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn import metrics
from sklearn.feature_selection import f_regression, RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

path = r"C:\Users\linho\Desktop\CST\term3\pred_analytics\data\winequality.csv"

df = pd.read_csv(path)

X = df.copy()
del X['quality']

y = df['quality']

# ============================================================================
# FORWARD FEATURE SELECTION
# =============================================================================
ffs = f_regression(X, y)

feature_df = pd.DataFrame()

for i in range(len(X.columns)):
    feature_df = feature_df._append({"feature": X.columns[i],
                                    "ffs": ffs[0][i]},
                                   ignore_index=True)

feature_df = feature_df.sort_values(by=['ffs'], ascending=False)
print(feature_df)

X = df[['alcohol', 'volatile acidity', 'sulphates',
        'citric acid', 'total sulfur dioxide']]

X = sm.add_constant(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

model = sm.OLS(y_train, X_train).fit()
predictions = model.predict(X_test)

print(model.summary())

print(f'RMSE: {np.sqrt(metrics.mean_squared_error(y_test, predictions))}')


# ==============================================================================
# BACKWARD FEATURE SELECTION
# ==============================================================================
X = df.copy()
del X['quality']

model = LinearRegression()

rfe = RFE(model, n_features_to_select=5)
rfe.fit(X, y)

print("\n\n\n=======================================================================")

print("============= FEATURE SELECTED ==================")
# print(rfe.support_)

columns = list(X.keys())

for i in range(len(columns)):
    if(rfe.support_[i]):
        print(columns[i])