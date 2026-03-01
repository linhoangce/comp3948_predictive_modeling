import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
path = r"C:\Users\linho\Desktop\CST\term3\pred_analytics\data\USA_Housing.csv"

df = pd.read_csv(path)

print(df.head())
print(df.describe())

X = df.copy()
del X['Price']
del X['Address']
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

# Scale data
std_sc_X = StandardScaler()
mm_sc_X = MinMaxScaler()
std_sc_y = StandardScaler()
mm_sc_y = MinMaxScaler()
X_std= std_sc_X.fit_transform(X)
X_mm = mm_sc_X.fit_transform(X)
y_mm = mm_sc_y.fit_transform(np.array(y).reshape(-1, 1))
y_std = std_sc_y.fit_transform(np.array(y).reshape(-1, 1))

columns = list(X.keys())

for i in range(1, 6):
    model1 = LinearRegression()
    model2 = LinearRegression()

    rfe1 = RFE(model1, n_features_to_select=i)
    rfe2 = RFE(model2, n_features_to_select=i)

    rfe1.fit(X_std, y_std)
    rfe2.fit(X_mm, y_mm)
    cols = []

    print(f"\n\n{i} FEATURES SELECTED")
    print("========================================================================")

    print("************** STD Fit *********************")
    for i in range(len(columns)):
        if(rfe1.support_[i]):
            print(columns[i])
            cols.append(i)

    print("************** MinMax Fit *********************")
    for i in range(len(columns)):
        if(rfe2.support_[i]):
            print(columns[i])

    X_train_std, X_test_std, y_train_std, y_test_std = train_test_split(
        X_std[:, cols], y_std, test_size=0.2
    )

    X_train_mm, X_test_mm, y_train_mm, y_test_mm = train_test_split(
        X_mm[:, cols], y_mm, test_size=0.2
    )

    model1.fit(X_train_std, y_train_std)
    model2.fit(X_train_mm, y_train_mm)

    y_pred1 = model1.predict(X_test_std)
    y_pred2 = model2.predict(X_test_mm)

    print("----------------------------------------------------------------------------")
    print(f"Model 1 Coef: {model1.coef_}")
    print(f"Model 1 Intercept: {model1.intercept_}")
    print(f"Model 1 R-squared: {r2_score(y_test_std, y_pred1)}")
    print(f"Model 1 RMSE: {np.sqrt(metrics.mean_squared_error(y_test_std, y_pred1))}")
    print("----------------------------------------------------------------------------")
    print(f"Model 2 Coef: {model2.coef_}")
    print(f"Model 2 Intercept: {model2.intercept_}")
    print(f"Model 2 R-squared: {r2_score(y_test_mm, y_pred2)}")
    print(f"Model 2 RMSE: {np.sqrt(metrics.mean_squared_error(y_test_mm, y_pred2))}")
    print("========================================================================")
