import pandas as pd
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

def show_y_plots(y_train, y_test, title):
    print(f"\n*** {title}")
    plt.subplots(1, 2)

    plt.subplot(1, 2, 1)
    plt.hist(y_train)
    plt.title(f"y_train {title}")

    plt.subplot(1, 2, 2)
    plt.hist(y_test)
    plt.title(f"y_test {title}")
    plt.show()

def evaluate_model(X_test, y_test, y_train, model, title):
    show_y_plots(y_train, y_test, title)

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Accuracy: {accuracy}")

path = r"C:\Users\linho\Desktop\CST\term3\pred_analytics\data\framingham_v2.csv"

df = pd.read_csv(path)

print(df.head())
print(df.describe())


X = df.copy()
del X['TenYearCHD']

y = df['TenYearCHD']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

clf1 = LogisticRegression(solver='newton-cg', max_iter=1000)
clf2 = LogisticRegression(solver='newton-cg', max_iter=1000)

smote = SMOTE()

X_train_SMOTE, y_train_SMOTE = smote.fit_resample(X_train, y_train)

clf1.fit(X_train, y_train)
clf2.fit(X_train_SMOTE, y_train_SMOTE)

evaluate_model(X_test, y_test, y_train, clf1, "Before SMOTE")
evaluate_model(X_test, y_test, y_train_SMOTE, clf2, "After SMOTE")
