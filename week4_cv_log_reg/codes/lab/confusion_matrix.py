import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import metrics

PATH = r"C:\Users\linho\Desktop\CST\term3\pred_analytics\data\employee_turnover.csv"

def predict_employee_turnover():
    df = pd.read_csv(PATH)

    predictor_vars = [col for col in df.keys() if col != 'turnover']
    # print(predictor_vars)
    # Construct features and label
    X = df[predictor_vars]
    y = df['turnover']

    # evaluate predictors
    selector = SelectKBest(score_func=chi2)
    selector.fit(X, y)

    chi_scores = dict(zip(predictor_vars, selector.scores_))
    # extract only significant predictors, i.e chi scores > 3.8
    sig_predictors = {pred:score.item() for pred,score in chi_scores.items() if score >= 3.8}

    X = df[sig_predictors.keys()]

    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2
    )

    # instantiate logistic model
    logistic_model = LogisticRegression(fit_intercept=True, solver='liblinear')
    logistic_model.fit(X_train, y_train)

    # make predictions
    y_pred = logistic_model.predict(X_test)
    # retrieve accuracy
    acc = metrics.accuracy_score(y_test, y_pred)

    # create confusion matrix
    conf_mat = pd.crosstab(y_test, y_pred,
                           rownames=['Actual'], colnames=['Predicted'])
    print(f'Accuracy: {acc:.2f}')
    print(f'Confusion Matrix: \n{conf_mat}')

    ### Precision and Recall
    TN = conf_mat[0][0]
    FN = conf_mat[0][1]
    FP = conf_mat[1][0]
    TP = conf_mat[1][1]

    print(f'\nTrue Neg: {TN}')
    print(f'False Neg: {FN}')
    print(f'False Pos: {FP}')
    print(f'True Pos: {TP}')

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * (precision * recall) / (precision + recall)

    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1: {F1}')

predict_employee_turnover()