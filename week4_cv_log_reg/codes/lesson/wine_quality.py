import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import RobustScaler
from sklearn import metrics
from sklearn import datasets
from sklearn.model_selection import train_test_split
from pickle import dump, load


pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

wine = datasets.load_wine()

df = pd.DataFrame(data=np.c_[wine['data'], wine['target']],
                  columns=wine['feature_names'] + ['target'])

print(df.head())
X = df[['alcalinity_of_ash', 'total_phenols', 'flavanoids',
        'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity',
        'od280/od315_of_diluted_wines', 'proline']]
y = df['target']

X = sm.add_constant(X)

# split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

# model = sm.OLS(y_train, X_train).fit()
# y_pred = model.predict(X_test)

sc_X = RobustScaler()
# scale features
X_train_scaled = sc_X.fit_transform(X_train)
X_test_scaled = sc_X.fit_transform(X_test)

# scale target
sc_y = RobustScaler()
y_train_scaled = sc_y.fit_transform(np.array(y_train).reshape(-1, 1))

model = sm.OLS(y_train_scaled, X_train_scaled).fit()
y_pred_scaled = model.predict(X_test_scaled)

# rescale predictions back to actual size range
y_pred = sc_y.inverse_transform(np.array(y_pred_scaled).reshape(-1, 1))


print(model.summary())
print(f"RMSE: {np.sqrt(metrics.mean_squared_error(y_test, y_pred))}")

####################################################################
### Saving and Loading models

# save scalers
dump(model, open('ols_model.pkl', 'wb'))

# load scalers
loaded_model = load(open('ols_model.pkl', 'rb'))

y_pred_loaded = loaded_model.predict(X_test_scaled)
y_pred = sc_y.inverse_transform(np.array(y_pred_loaded).reshape(-1, 1))

print(f"RMSE from loaded model: {np.sqrt(metrics.mean_squared_error(y_test, y_pred))}")

