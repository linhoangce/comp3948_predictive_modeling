import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import RobustScaler


PATH = r"C:\Users\linho\Desktop\CST\term3\pred_analytics\data\computerPurchase.csv"

def load_dataset(path):
    df = pd.read_csv(path)

    X = df[['Age', 'EstimatedSalary']]
    y = df['Purchased']

    return X, y


def min_max_scaling(path):

    X, y = load_dataset

    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25
    )

    # instantiate model
    model = LogisticRegression(fit_intercept=True, solver='liblinear')

    # fit model
    model.fit(X_train, y_train)
    # make predictions
    y_pred = model.predict(X_test)

    conf_mat = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
    print("\n\n ======== Original Dataset ========")
    print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}")
    print(f'\nConfusion matrix:\n {conf_mat}')

    ### MinMax Scaler
    sc = MinMaxScaler()

    # scale data
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)

    model1 = LogisticRegression(fit_intercept=True, solver='liblinear')
    model1.fit(X_train_scaled, y_train)

    # make predictions
    y_pred = model1.predict(X_test_scaled)

    conf_mat1 = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])

    print("\n\n======= Scaled Data =======")
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print(f"Conf Mat:\n{conf_mat1}")

def show_automated_scaler_results(path):
    X, y = load_dataset(path)

    std_sc = StandardScaler()
    X_scaled = std_sc.fit_transform(X)
    salary = X.iloc[0][1]
    scaled_salary = X_scaled[0][1]

    print("First unscaled X:", salary)
    print(f"First scaled X: {scaled_salary}")

    print("\n*** Showing manually calculated results: ")
    sd = get_std_with_zero_degree_freedom(X)
    mean = X['EstimatedSalary'].mean()
    scaled = (19000 - mean) / sd

    print("$19,000 scaled manually is: " + str(scaled))


def get_std_with_zero_degree_freedom(X):
    mean = X['EstimatedSalary'].mean()

    # StandardScaler calculates the standard deviation with zero degree of freedom
    s1 = X['EstimatedSalary'].std(ddof=0)
    print(f"Std with 0 degree of freedom automated: {s1}")

    # manual calculation for standard scaling
    s2 = np.sqrt(np.sum(((X['EstimatedSalary'] - mean)**2) / len(X)))
    print(f"Std with 0 degree of freedom manually: {s2}")

    return s1


def robust_scaling(path):
    X, y = load_dataset(path)

    sc = RobustScaler()

    X_scaled = sc.fit_transform(X)

    print("\n\n======= Robust Scaler =======")
    return X_scaled, y


print("show automated results")
show_automated_scaler_results(PATH)



def make_predictions(path, dataset_scaler):
    X, y = dataset_scaler

    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2
    )

    model = LogisticRegression(fit_intercept=True, solver='liblinear')
    model.fit(X_train, y_train)

    # make predictions
    y_pred = model.predict(X_test)

    # confusion matrix
    conf_mat = pd.crosstab(y_test, y_pred,
                           rownames=['Actual'], colnames=['Predicted'])

    print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
    print("Confusion matrix:\n", conf_mat)

robust_scaler = robust_scaling(PATH)
make_predictions(PATH, robust_scaler)

