import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold


data = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

#s split data into 3 randomized folds
kfold = KFold(n_splits=3, shuffle=True)

for train, test in kfold.split(data):
    print(f"train: {data[train]}, test: {data[test]}")


#################################################################
PATH = r"C:\Users\linho\Desktop\CST\term3\pred_analytics\data\computerPurchase.csv"

df = pd.read_csv(PATH)
label = 'Purchased'
features = ['EstimatedSalary', 'Age']

def get_test_and_train_data(train_idx, test_idx, df, label, features):
    df_train = df.iloc[train_idx, :]
    df_test = df.iloc[test_idx, :]
    X_train = df_train[features]
    X_test = df_test[features]
    y_train = df_train[label]
    y_test = df_test[label]
    return X_train, X_test, y_train, y_test

def make_predictions(df, label, features, scaler, splits):
    kfold = KFold(n_splits=splits, shuffle=True)
    accuracies = []
    precisions = []
    recalls = []
    fold_count = 0

    for train_idx, test_idx in kfold.split(df):
        X_train, X_test, y_train, y_test = \
            get_test_and_train_data(train_idx, test_idx, df,
                                    label=label, features=features)

        sc = scaler
        X_train_scaled = sc.fit_transform(X_train)
        X_test_scaled = sc.transform(X_test)

        # initialize model
        model = LogisticRegression()
        model.fit(X_train_scaled, y_train)

        # make preds
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)

        conf_mat = pd.crosstab(y_test, y_pred,
                               rownames=['Actual'], colnames=['Predicted'])

        print(f'\n***K-fold: {fold_count}')
        fold_count += 1

        # calculate metrics
        accuracy = metrics.accuracy_score(y_test, y_pred)
        precision = metrics.precision_score(y_test, y_pred, average=None)
        recall = metrics.recall_score(y_test, y_pred, average=None)

        precisions.append(precision)
        recalls.append(recall)
        accuracies.append(accuracy)

        print("\nAccuracy: ", accuracy)
        print("\nConfusion matrix: ", conf_mat)

    # calculate metrics
    auc = metrics.roc_auc_score(y_test, y_prob[:, 1])

    print("\n**** All Folds Metrics ****")
    print("================================================")
    print(f"Average acc: {np.mean(accuracies):.4f}")
    print(f"Accuracy std: {np.std(accuracies):.4f}")
    print(f"ROC AUC: {auc:.3f}")
    print(f"Precision: {np.mean(precisions):.4f}")
    print(f"Precision std: {np.std(precisions):.4f}")
    print(f"Recall: {np.mean(recalls):.4f}")
    print(f"Recall Std: {np.std(recalls):.4f}")

robust_sc = RobustScaler()
mm_sc = MinMaxScaler()
std_sc = StandardScaler()

make_predictions(df, label, features, std_sc, splits=3)