import numpy as np
import pandas as pd
from sklearn.feature_selection import f_regression
from sklearn.metrics import roc_auc_score

from lr_chi_square import build_evaluate_classifier

pd.set_option("display.max_column", None)
pd.set_option("display.width", 1000)

divorce_path = r"C:\Users\linho\Desktop\CST\term3\pred_analytics\data\Divorce.csv"
bank_path = r"C:\Users\linho\Desktop\CST\term3\pred_analytics\data\bank-additional-full.csv"

df_divorce = pd.read_csv(divorce_path, header=0)
df_bank = pd.read_csv(bank_path, sep=';')

X = df_divorce.copy()
del X['Divorce']
y = df_divorce['Divorce']

ffs = f_regression(X, y)

variables = []
for i in range(len(X.columns) - 1):
    if ffs[0][i] >= 700:
        print(ffs[0][i])
        variables.append(X.columns[i])

print(variables)

df_bank['target'] = df_bank['y'].apply(lambda val: 1 if val == 'yes' else 0)

temp_df = df_bank[["job", "marital", "education", "default",
              "housing", "loan", "contact", "month",
              "day_of_week", "poutcome"]]
dummy_df = pd.get_dummies(temp_df,
                          columns=["job", "marital", "education", "default",
                                    "housing", "loan", "contact", "month",
                                    "day_of_week", "poutcome"])
df = pd.concat([df_bank, dummy_df], axis=1)

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

ffs = f_regression(X, y)

top_features = []

for i in range(len(X.columns)-1):
    if ffs[0][i] >= 700: # comparing F-stats
        top_features.append(X.columns[i])

print(top_features)

X_test, y_test, y_pred, y_prob = build_evaluate_classifier(top_features, X, y)

print(f"AUC: {roc_auc_score(y_test, y_prob[:, 1])}")