from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
import pandas as pd

iris = datasets.load_iris()
X = iris.data
y = iris.target
X_std = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_std, y, test_size=0.25, random_state=0
)

pca = PCA(n_components=2)

# transform data with PCA
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

model = LogisticRegression(fit_intercept=True, solver='liblinear',
                           random_state=0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nIntercept:")
print(model.intercept_)

print("\n*** Model Coefficients:")
print(model.coef_)

con_mat = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])

print("\n*** Confusion Matrix")
print(con_mat)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

vif = pd.DataFrame()

vif['VIF Factor for Components'] = [variance_inflation_factor(X_train, i)
                                    for i in range(X_train.shape[1])]
print(f"\nVIF\n{vif}")