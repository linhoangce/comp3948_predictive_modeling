import pandas as pd
from mlxtend.data import loadlocal_mnist
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

mnist = fetch_openml('mnist_784')

X = mnist.data
y = mnist.target

print(f"Dimension: {X.shape[0]}x{X.shape[1]}")
print(f"\n1st row:\n {X.iloc[0]}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1/720, random_state=0
)

for i in range(3):
    image = np.array(X_train.iloc[i], dtype='float')
    pixels = image.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pca = PCA(0.95)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

model = LogisticRegression(solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
score = model.score(X_test, y_test)
print(f"Model Score: {score}")

conf_mat = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
print(f"Conf Matrix:\n{conf_mat}")

print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

