import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

path = r"C:\Users\linho\Desktop\CST\term3\pred_analytics\data\bank-additional-full.csv"

df = pd.read_csv(path, sep=';', encoding="ISO-8859-1")

# add a target column for dummy variable yes/no
df['target'] = df['y'].apply(lambda val: 1 if val == 'yes' else 0)
print(df.head(5))


temp_df = df[["job", "marital", "education", "default",
              "housing", "loan", "contact", "month",
              "day_of_week", "poutcome"]]
dummy_df = pd.get_dummies(temp_df,
                          columns=["job", "marital", "education", "default",
                                    "housing", "loan", "contact", "month",
                                    "day_of_week", "poutcome"])
df = pd.concat([df, dummy_df], axis=1)

X = df[[
    "age", "duration", "campaign", "pdays", "previous", "emp.var.rate", "cons.price.idx",
    "cons.conf.idx", "euribor3m", "nr.employed", "job_admin.", "job_blue-collar",
    "job_entrepreneur", "job_housemaid", "job_management", "job_retired",
    "job_self-employed", "job_services", "job_student", "job_technician", "job_unemployed",
    "job_unknown", "marital_divorced", "marital_married", "marital_single",
    "marital_unknown", "education_basic.4y", "education_basic.6y", "education_basic.9y",
    "education_high.school", "education_illiterate", "education_professional.course",
    "education_university.degree", "education_unknown", "default_no",
    "default_unknown", "default_yes", "housing_no", "housing_unknown", "housing_yes",
    "loan_no", "loan_unknown", "loan_yes", "contact_cellular", "contact_telephone",
    "month_apr", "month_aug", "month_dec", "month_jul", "month_jun", "month_mar",
    "month_may", "month_nov", "month_oct", "month_sep", "day_of_week_fri",
    "day_of_week_mon", "day_of_week_thu", "day_of_week_tue", "day_of_week_wed",
    "poutcome_failure", "poutcome_nonexistent", "poutcome_success"
]]

y = df['target']

def select_k_top_features_chi_square(selector, X, y, k):
    X_scaled = MinMaxScaler().fit_transform(X)
    sel_fitted = selector.fit(X_scaled, y)

    score_dict = dict(zip(X.keys(), sel_fitted.scores_))
    feature_objects = [{'feature': feature, 'score': score}
                       for feature, score in score_dict.items()]

    df_features = (pd.DataFrame(feature_objects)
                   .sort_values(by=['score'], ascending=False))
    print('\n=========== Top Features ===========\n')
    print(df_features.head(15)['feature'])

    return df_features.head(15)

def build_evaluate_classifier(features, X, y):
    X = X[features]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25
    )

    model = LogisticRegression(fit_intercept=True, solver='liblinear')

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    print("============== Model Summary =================")
    print(f"Recall: {recall_score(y_test, y_pred)}")
    print(f"Precision: {precision_score(y_test, y_pred)}")
    print(f"F1: {f1_score(y_test, y_pred)}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

    return X_test, y_test, y_pred, y_prob


def select_k_features_rfe(X, y, k):
    model = LogisticRegression(solver='liblinear')
    rfe = RFE(model, n_features_to_select=k)
    rfe.fit(X, y)

    print("\n\n=============== Selected Features RFE ==================")
    selected_features = X.keys()[rfe.support_ == True]
    print(selected_features.values)

    return selected_features.values

K = 15
selector = SelectKBest(score_func=chi2, k=K)
top_features_chi2 = select_k_top_features_chi_square(
    selector, X, y, k=K
)
top_features_rfe = select_k_features_rfe(X, y, k=K)

X_test_chi2, y_test_chi2, y_pred_chi2, y_prob_chi2 = build_evaluate_classifier(
    top_features_chi2['feature'], X, y
)
print("\n\n=================================================================\n")
X_test_rfe, y_test_rfe, y_pred_rfe, y_prob_rfe = build_evaluate_classifier(
    top_features_rfe, X, y
)

auc_chi2 = roc_auc_score(y_test_chi2, y_prob_chi2[:, 1])
auc_rfe = roc_auc_score(y_test_rfe, y_prob_rfe[:, 1])
print(f"\n\nChi2 - Logistic ROC AUC: {auc_chi2}")
print(f"RFE - Logistic ROC AUC: {auc_rfe}")


lr_fpr, lr_tpr, _ = roc_curve(y_test_chi2, y_prob_chi2[:, 1])

plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
plt.plot([0, 1], [0, 1], '--', label='No Skill')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.title("Regression with top 15 features identified with Chi-square score")
plt.show()

print(df['target'].value_counts())