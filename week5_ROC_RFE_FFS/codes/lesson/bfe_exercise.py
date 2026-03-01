import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import recall_score, precision_score
from sklearn.metrics import f1_score, accuracy_score

path = r"C:\Users\linho\Desktop\CST\term3\pred_analytics\data\bank-additional-full.csv"

df = pd.read_csv(path, sep=';', encoding="ISO-8859-1")

pd.set_option("display.max_column", None)
pd.set_option("display.width", 1000)

print(df.head())
print(df.describe().transpose())
print(df.info())

df['target'] = df['y'].apply(lambda val : 1 if val == 'yes' else 0)

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

def select_features_rfe(model, X, y, num_features):

    rfe = RFE(model, n_features_to_select=num_features)
    rfe.fit(X, y)
    print("\n\n============================================================")
    print("Selected Features")
    # use the mask returned by RFE to apply on the column names array
    selected_features = X.keys()[rfe.support_ == True]
    print(f"here: {selected_features.values}")

# logistic_model = LogisticRegression(solver='liblinear')
# select_features_rfe(logistic_model, X, y, num_features=15)

print("\n\n================================================================")
np.set_printoptions(precision=3)

def select_k_best_features_chi_square(selector, X, y, k):
    # Scale data
    X_scaled = MinMaxScaler().fit_transform(X)
    chi_scores = selector.fit(X_scaled, y)

    print(f"Predictor Chi-Square Scores: {chi_scores.scores_}")

    # map chi_scores to their respective features
    chi_scores_zipped = dict(zip(X.keys(), chi_scores.scores_))
    feature_objects = [{"feature": feature, "chi-square score": score}
                       for feature, score in chi_scores_zipped.items()]

    df_features = (pd.DataFrame(feature_objects)
                   .sort_values(by=['chi-square score'], ascending=False))
    print(df_features.head(k))

    return df_features

selector = SelectKBest(score_func=chi2, k=15)


def build_evaluate_classifier(features, X, y):
    X = X[features]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25
    )

    model = LogisticRegression(fit_intercept=True, solver='liblinear')

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    print("========================== No Scaling ==========================")
    print(f"Recall: {recall_score(y_test, y_pred)}")
    print(f'Precision: {precision_score(y_test, y_pred)}')
    print(f"F1 score: {f1_score(y_test, y_pred)}")
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

    return X_test, y_test, y_pred, y_prob

df_features = select_k_best_features_chi_square(selector, X, y, k=15)
X_test, y_test, y_pred, y_prob = build_evaluate_classifier(df_features['feature'], X, y)

auc = roc_auc_score(y_test, y_prob[:, 1],)
print('Logistic: ROC AUC=%.3f' % (auc))

# calculate roc curves
lr_fpr, lr_tpr, _ = roc_curve(y_test, y_prob[:, 1])
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
plt.plot([0,1], [0,1], '--', label='No Skill')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

print(df['target'].value_counts())
