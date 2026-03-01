import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

def chi_squared_overview():
    candidates = {'gmat': [780,750,690,710,680,730,690,720,
     740,690,610,690,710,680,770,610,580,650,540,590,620,
     600,550,550,570,670,660,580,650,660,640,620,660,660,
     680,650,670,580,590,690],
                  'gpa': [4,3.9,3.3,3.7,3.9,3.7,2.3,3.3,
     3.3,1.7,2.7,3.7,3.7,3.3,3.3,3,2.7,3.7,2.7,2.3,
     3.3,2,2.3,2.7,3,3.3,3.7,2.3,3.7,3.3,3,2.7,4,
     3.3,3.3,2.3,2.7,3.3,1.7,3.7],
                  'work_experience': [3,4,3,5,4,6,1,4,5,
     1,3,5,6,4,3,1,4,6,2,3,2,1,4,1,2,6,4,2,6,5,1,2,4,6,
     5,1,2,1,4,5],
                  'admitted': [1,1,1,1,1,1,0,1,1,0,0,1,
     1,1,1,0,0,1,0,0,0,0,0,0,0,1,1,0,1,1,0,0,1,1,1,0,0,
     0,0,1]}

    df = pd.DataFrame(candidates)
    print(df)

    # extract features and labels
    predictor_vars = ['gmat', 'gpa', 'work_experience']
    X = df[predictor_vars]
    y = df['admitted']

    # show chi-squared scores for each feature
    # there is 1 degree of freedom since there is 1 predictor
    # during feature evaluation
    test = SelectKBest(score_func=chi2, k=3)
    chi_scores = test.fit(X, y)
    np.set_printoptions(precision=3)

    print("\nPredictor VariableS: " + str(predictor_vars))
    print(f"Predictor Chi-Squared Scores: {chi_scores.scores_}")

    # another technique for showing the most statistically
    # significant variables involves the get_support() function
    cols = chi_scores.get_support(indices=True)
    print(cols)
    features = X.columns[cols]
    print(np.array(features))

    print('\n\n*** LOGISTIC REGRESSION ***')
    # Building Linear Regression model with significant predictors
    X = df[['gmat', 'work_experience']]

    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )

    # instantiate model
    logistic_model = LogisticRegression(fit_intercept=True, solver='liblinear',
                                random_state=0)
    logistic_model.fit(X_train, y_train)
    y_pred = logistic_model.predict(X_test)
    print(f"y_test: {y_test.values}")
    print(f"y_pred: {y_pred}")


path = r"C:\Users\linho\Desktop\CST\term3\pred_analytics\data\employee_turnover.csv"

def predict_employee_turnover():

    df = pd.read_csv(path)
    print(df.head())

    predictor_vars = list(df.keys())
    predictor_vars.remove('turnover')
    # print(predictor_vars)

    # construct features and label
    X = df[predictor_vars]
    y = df['turnover']

    # Ex 2
    ### Naive Way
    # selector = SelectKBest(score_func=chi2)
    # chi_scores = selector.fit(X, y)
    # # print(chi_scores.scores_)
    #
    # all_chi_score_df = pd.DataFrame(chi_scores.scores_).T
    # all_chi_score_df.columns = predictor_vars
    #
    # result = all_chi_score_df[all_chi_score_df >= 3.8].dropna(axis=1)
    # print(all_chi_score_df)
    # print('\n*** Significant Variables ***\n')
    # print(result)

    ### More efficient way using zip()
    selector = SelectKBest(score_func=chi2, k=4)
    selector.fit(X, y)

    chi_scores = dict(zip(predictor_vars, selector.scores_))
    significant_vars = {var:score.item() for var,score in chi_scores.items() if score >= 3.8}
    print(significant_vars)

    print("\n\n=========== LOGISTIC REGRESSION Exercise 3==============")
    predictor_vars = significant_vars.keys();
    X = df[predictor_vars]
    y = df['turnover']

    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2
    )

    # instantiate model
    logistic_model = LogisticRegression(fit_intercept=True, solver='liblinear')
    logistic_model.fit(X_train, y_train)

    y_pred = logistic_model.predict(X_test)

    print(f"y_test: {y_test.values}")
    print(f"y_pred: {y_pred}")
    print(f"Accuracy: {np.sum((y_test == y_pred)) / len(y_test)}")

predict_employee_turnover()

# df = pd.read_csv(path)
# print(df[df.keys()])


# print(dict(zip(["var"], ["score"])))
# chi_squared_overview()