import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, f1_score
from sklearn.metrics import recall_score, accuracy_score
from imblearn.over_sampling import SMOTE

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

path = r"C:\Users\linho\Desktop\CST\term3\pred_analytics\data\healthcare-dataset-stroke-data.csv"

df = pd.read_csv(path)

def show_y_plots(y_train, y_test, title):
    print("\n*** "+title)
    plt.subplots(1, 2)

    plt.subplot(1, 2, 1)
    plt.hist(y_train)
    plt.title(f"Train Y: {title}")

    plt.subplot(1, 2, 2)
    plt.hist(y_test)
    plt.title(f"Test Y: {title}")
    plt.show()


def evaluate_model(X_test, y_test, y_train, model, title):
    show_y_plots(y_train, y_test, title)

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    precision = precision_score(y_test, y_pred, average='binary')
    print(f"Pricision: {precision}")

    recall = recall_score(y_test, y_pred, average='binary')
    print(f"Recall: {recall}")

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")


print(df.head())
print(df.describe())

average_bmi = np.mean(df['bmi'])
df['bmi'] = df['bmi'].replace(np.nan, average_bmi)
print(df.describe())

X = df[['age','hypertension','heart_disease','avg_glucose_level','bmi']]
y = df['stroke']

X_train, X_test, y_train, y_test = train_test_split(X, y)

clf = LogisticRegression(solver='newton-cg', max_iter=1000)

clf.fit(X_train, y_train)
evaluate_model(X_test, y_test, y_train, clf, "Before SMOTE")
y_pred = clf.predict(X_test)

smt = SMOTE()

X_train_SMOTE, y_train_SMOTE = SMOTE().fit_resample(X_train, y_train)

clf2 = LogisticRegression(solver='newton-cg', max_iter=1000)
clf2.fit(X_train_SMOTE, y_train_SMOTE)
evaluate_model(X_test, y_test, y_train_SMOTE, clf2, 'After SMOTE')
y_pred_smote = clf2.predict(X_test)
