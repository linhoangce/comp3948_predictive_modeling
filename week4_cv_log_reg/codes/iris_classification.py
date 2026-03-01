import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import metrics

iris = datasets.load_iris()
X = iris.data
y = iris.target

# split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

# create one vs rest regression object
clf = LogisticRegression(multi_class='multinomial', solver='newton-cg',
                         random_state=0)

model = clf.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(y_test)
print(y_pred)

# view predicted probs
y_prob = model.predict_proba(X_test)
print(y_prob)

precision = metrics.precision_score(y_test, y_pred, average=None)
recall = metrics.recall_score(y_test, y_pred, average=None)
f1 = metrics.f1_score(y_test, y_pred, average=None)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1: {f1}")