import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

PATH = r"C:\Users\linho\Desktop\CST\term3\pred_analytics\data\glass.csv"

def classify_glass_types(path):
    df = pd.read_csv(path)
    print(df.head())

    X = df.copy()
    del X['Type']

    y = df['Type']

    print(f"Labels: {y.unique()}")
    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25
    )

    # instantiate logistic model
    logistic_model = LogisticRegression(solver='newton-cg')
    logistic_model.fit(X_train, y_train)

    # make predictions
    y_pred = logistic_model.predict(X_test)

    # Precision, recall, f1 and accuracy
    acc = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred, average=None)
    recall = metrics.recall_score(y_test, y_pred, average=None)
    f1 = metrics.f1_score(y_test, y_pred, average=None)

    print(f"Accuracy: {acc:.2f}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")

classify_glass_types(PATH)