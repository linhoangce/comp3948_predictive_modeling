import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

path = r"C:\Users\linho\Desktop\CST\term3\pred_analytics\data\Divorce.csv"

df = pd.read_csv(path, header=0)
model = LogisticRegression()

def select_features_rfe(model, df, label, num_features):

    X = df.copy()
    del X[label]

    y = df[label]

    rfe = RFE(model, n_features_to_select = num_features)

    rfe = rfe.fit(X, y)

    print("\n\nFEATURE SELECTED\n\n")
    print(rfe.support_)

    for i in range(0, len(X.keys())):
        if(rfe.support_[i]):
            print(X.keys()[i])

def build_and_evaluate_classifier(features, X, y):
    X = X[features]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.25, random_state=0
    )

    model = LogisticRegression(fit_intercept=True, solver="liblinear", random_state=0)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n\nResult without scaling")

    conf_mat = pd.crosstab(y_test, y_pred,
                           rownames=['Actual'], colnames=['Predicted'])
    print("\nConfusion matrix")
    print(conf_mat)

    print(f"Recall: {recall_score(y_test, y_pred)}")
    print(f"Precision: {precision_score(y_test, y_pred)}")
    print(f"F1: {f1_score(y_test, y_pred)}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

select_features_rfe(model, df, "Divorce", 8)

features = ['Q3', 'Q6', 'Q17', 'Q18', 'Q26', 'Q39', 'Q40', 'Q49']
build_and_evaluate_classifier(features, df[features], df['Divorce'])

