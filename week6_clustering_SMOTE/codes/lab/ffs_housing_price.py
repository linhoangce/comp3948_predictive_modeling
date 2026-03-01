import numpy as np
import pandas as pd
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

path = r"C:\Users\linho\Desktop\CST\term3\pred_analytics\data\USA_Housing.csv"

df = pd.read_csv(path)

X = df.copy()
del X['Price']
del X['Address']

y = df['Price']

print(df.head())

std_sc_X = StandardScaler()
std_sc_y = StandardScaler()
mm_sc_X = MinMaxScaler()
mm_sc_y = MinMaxScaler()

X_std = std_sc_X.fit_transform(X)
X_mm = mm_sc_X.fit_transform(X)
y_std = std_sc_y.fit_transform(np.array(y).reshape(-1, 1))
y_mm = mm_sc_y.fit_transform(np.array(y).reshape(-1, 1))

ffs_std = f_regression(X_std, y_std)
ffs_mm = f_regression(X_mm, y_mm)

X_std_df = pd.DataFrame({'feature': X.columns,
                         'ffs_fstat': ffs_std[0]})
X_std_df = X_std_df.sort_values(['ffs_fstat'], ascending=False)
print(f"*** STD Scaler\n{X_std_df}")

X_mm_df = pd.DataFrame({'feature': X.columns,
                        'ffs_fstat': ffs_mm[0]})
X_mm_df = X_mm_df.sort_values(['ffs_fstat'], ascending=False)
print(f"\n*** MinMaxScaler\n{X_mm_df}")

model1 = LinearRegression()
model2 = LinearRegression()

print("\n\n")
for i in range(3, 6):
    top_features = X_mm_df['feature'][:i]
    print(f"*************** {i} TOP FEATURES *****************")
    X_top = X_mm[:, top_features.index]
    X_train, X_test, y_train, y_test = train_test_split(
        X_top, y_mm, test_size=0.2
    )

    model1.fit(X_train, y_train)
    y_pred = model1.predict(X_test)

    print(f"Coef: {model1.coef_}")
    print(f"Intercept: {model1.intercept_}")
    print(f"R-squared: {r2_score(y_test, y_pred)}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")
    print("========================================================================\n")


