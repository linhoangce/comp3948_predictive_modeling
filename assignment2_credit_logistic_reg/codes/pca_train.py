# ===========================================================================
# FEATURE SELECTIONS
# ==========================================================================
import numpy as np
import pandas as pd
import pickle
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from sklearn.feature_selection import RFE, f_regression, SelectKBest, chi2
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, \
    confusion_matrix
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler



def encode_ordinal_employment(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Encode 'employment' as an ordinal feature using a fixed mapping (no learning from data)."""
    if "employment" not in X_train.columns:
        return X_train, X_test

    employment_map = {
        "<1": 0,
        "unemployed": 1,
        "1<=X<4": 2,
        "4<=X<7": 3,
        ">=7": 4,
    }

    X_train["employment"] = X_train["employment"].map(employment_map)
    X_test["employment"] = X_test["employment"].map(employment_map)

    X_train["employment"] = pd.to_numeric(X_train["employment"])
    X_test["employment"] = pd.to_numeric(X_test["employment"])
    return X_train, X_test


def impute_train_test(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Leak-free imputation:
      - Categorical: mode computed from training only, applied to both.
      - Numeric: KNNImputer fitted on training numeric columns only.
    """
    # Categorical imputation (mode)
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        mode_val = X_train[col].mode(dropna=True)
        if len(mode_val) == 0:
            continue
        mode_val = mode_val[0]
        X_train[col] = X_train[col].fillna(mode_val)
        if col in X_test.columns:
            X_test[col] = X_test[col].fillna(mode_val)

    # Numeric imputation (KNN on all numeric columns)
    num_cols = X_train.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        imputer = KNNImputer(n_neighbors=5)
        X_train_num = imputer.fit_transform(X_train[num_cols])
        X_test_num = imputer.transform(X_test[num_cols])

        X_train[num_cols] = X_train_num
        X_test[num_cols] = X_test_num

    return X_train, X_test


def add_age_bins_fixed(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Fixed bins for 'age' (no learning):
        [18, 25, 35, 50, 65, 100]
    Then create dummy variables for these bins (drop_first=True).
    """
    if "age" not in X_train.columns:
        return X_train, X_test

    bins = [18, 25, 35, 50, 65, 100]
    labels = ["18-25", "25-35", "35-50", "50-65", "65+"]

    # Train
    age_bin_train = pd.cut(
        X_train["age"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )
    age_dummies_train = pd.get_dummies(
        age_bin_train, prefix="age_bin", drop_first=True, dtype=int
    )

    # Test
    age_bin_test = pd.cut(
        X_test["age"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )
    age_dummies_test = pd.get_dummies(
        age_bin_test, prefix="age_bin", drop_first=True, dtype=int
    )

    X_train = X_train.drop(columns=["age"])
    X_test = X_test.drop(columns=["age"])

    X_train = pd.concat([X_train.reset_index(drop=True), age_dummies_train.reset_index(drop=True)], axis=1)
    X_test = pd.concat([X_test.reset_index(drop=True), age_dummies_test.reset_index(drop=True)], axis=1)

    return X_train, X_test


def create_dummies_train_test(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Leak-free dummy creation:
      - get_dummies on training categorical columns,
      - get_dummies on test using same columns,
      - align columns (fill missing with 0).
    """
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns

    X_train_cat = pd.get_dummies(X_train[cat_cols], drop_first=True, dtype=int)
    X_test_cat = pd.get_dummies(X_test[cat_cols], drop_first=True, dtype=int)

    num_cols = [col for col in X_train.columns if col not in cat_cols]

    X_train_final = pd.concat(
        [X_train[num_cols].reset_index(drop=True), X_train_cat.reset_index(drop=True)],
        axis=1,
    )
    X_test_final = pd.concat(
        [X_test[num_cols].reset_index(drop=True), X_test_cat.reset_index(drop=True)],
        axis=1,
    )

    # Align columns: ensure same feature space
    X_train_final, X_test_final = X_train_final.align(
        X_test_final, join="left", axis=1, fill_value=0
    )

    return X_train_final, X_test_final


def preprocess_train_test(df: pd.DataFrame, target_col: str = "class"):
    """
    Full leak-free preprocessing:
        1. Split into X/y and train/test.
        2. Impute cat (mode) + numeric (KNN) on train, apply to test.
        3. Encode ordinal 'employment' with fixed mapping.
        4. Add fixed age bins -> dummy.
        5. Create dummy variables for all other categoricals.
    Returns: X_train, X_test, y_train, y_test (fully preprocessed).
    """
    print("\n*** READING & SPLITTING DATA ***\n")
    y = df[target_col].copy()
    X = df.drop(columns=[target_col]).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        shuffle=True,
        stratify=y,
        # random_state=42,
    )

    print("Imputing train/test ...")
    X_train, X_test = impute_train_test(X_train, X_test)

    print("Encoding ordinal 'employment' ...")
    X_train, X_test = encode_ordinal_employment(X_train, X_test)

    print("Adding fixed age bins ...")
    X_train, X_test = add_age_bins_fixed(X_train, X_test)

    print("Creating dummies (train/test) ...")
    X_train, X_test = create_dummies_train_test(X_train, X_test)

    # -------------------------------
    # Add interaction terms AFTER dummy creation
    # -------------------------------
    # -------------------------------
    # Add interaction terms AFTER dummy creation
    # -------------------------------
    for df_part in (X_train, X_test):

        # interaction of dummies
        if ("checking_status_<0" in df_part.columns
                and "credit_history_critical/other existing credit" in df_part.columns):
            df_part["very_low_liquid_assets"] = (
                    df_part["checking_status_<0"] *
                    df_part["credit_history_critical/other existing credit"]
            )

        if ("checking_status_<0" in df_part.columns
                and "savings_status_<100" in df_part.columns):
            df_part["liquidity_risk"] = (
                    df_part["checking_status_<0"] *
                    df_part["savings_status_<100"]
            )

        # continuous interactions
        df_part["monthly_payment"] = df_part["credit_amount"] / (df_part["duration"] + 1e-6)
        df_part["installment_x_amount"] = df_part["installment_commitment"] * df_part["credit_amount"]
        df_part["credit_density"] = df_part["credit_amount"] / (df_part["duration"] + 1e-6)
        df_part["debt_pressure"] = df_part["installment_commitment"] / (df_part["credit_amount"] + 1e-6)

    print("\nFinal preprocessed shapes:")
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

    X_train.to_csv('data/train.csv', index=False)
    y_train.to_csv('data/label_train.csv', index=False)
    X_test.to_csv('data/test.csv', index=False)
    y_test.to_csv('data/label_test.csv', index=False)
    return X_train, X_test, y_train, y_test


def evaluate_feature_selections(X, y, scaler, selected_features):

    k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    precision = []
    recall = []
    f1 = []
    acc = []

    for train_idx, test_idx in k_fold.split(X, y):
        X_train = X.iloc[train_idx].copy()
        y_train = y.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        y_test = y.iloc[test_idx].copy()

        X_train_scaled = scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = scaler.transform(X_test)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)


        X_train_fs = X_train_scaled[selected_features]
        X_test_fs = X_test_scaled[selected_features]

        X_train_sm, y_train_sm = SMOTE(random_state=42).fit_resample(X_train_fs, y_train)

        model = LogisticRegression(fit_intercept=True, solver='liblinear', max_iter=1000, random_state=42)
        model.fit(X_train_sm, y_train_sm)
        y_pred = model.predict(X_test_fs)

        precision.append(precision_score(y_test, y_pred))
        recall.append(recall_score(y_test, y_pred))
        f1.append(f1_score(y_test, y_pred))
        acc.append(accuracy_score(y_test, y_pred))
    print(f'Precision:{np.mean(precision)}',
            f'Recall: {np.mean(recall)}',
            f'F1: {np.mean(f1)}',
            f'Accuracy: {np.mean(acc)}')
    return {'Precision': np.mean(precision),
            'Recall': np.mean(recall),
            'F1': np.mean(f1),
            'Accuracy': np.mean(acc)}

def train_final_model(X_train, y_train, features, scaler):
    """
    Train final model:
      - scale on full training
      - SMOTE
      - logistic regression
    """
    X_selected = X_train[features]
    X_scaled = scaler.fit_transform(X_selected)
    X_sm, y_sm = SMOTE(random_state=42).fit_resample(X_scaled, y_train)

    # model = LogisticRegression(solver="liblinear", max_iter=2000, random_state=42)

    # model = LogisticRegression(
    #     solver="saga",
    #     penalty="l2",
    #     C=3.0,
    #     max_iter=5000,
    #     random_state=42
    # )

    model = LogisticRegression(
        solver="liblinear",
        max_iter=5000,
        random_state=42,
        class_weight={0: 1.0, 1: 3.0},
        penalty='l2',
    )

    model.fit(X_sm, y_sm)
    return model, scaler


def evaluate_final_model(model, scaler, X_test, y_test, features):
    """Evaluate on held-out test set"""
    X_selected = X_test[features]
    X_scaled = scaler.transform(X_selected)

    y_pred = model.predict(X_scaled)
    # probs = model.predict_proba(X_scaled)
    # y_pred = (probs[:, 1] >= 0.4).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    print('\nConf Mat')
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print(f"\nMacro F1 (TRUE performance metric): {f1_score(y_test, y_pred, average='macro'):.4f}")

    return {
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "Macro_F1": f1_score(y_test, y_pred, average='macro'),
        "Accuracy": accuracy_score(y_test, y_pred),
    }

df = pd.read_csv("data/Credit_Train.csv")



scaler = StandardScaler()

com1 = ['duration', 'credit_amount', 'employment', 'installment_commitment', 'existing_credits', 'age_bin_25-35', 'age_bin_35-50', 'age_bin_50-65', 'age_bin_65+', 'checking_status_<0', 'checking_status_no checking', 'credit_history_critical/other existing credit', 'credit_history_delayed previously', 'credit_history_existing paid', 'credit_history_no credits/all paid', 'purpose_education', 'purpose_new car', 'purpose_used car', 'savings_status_>=1000', 'savings_status_no known savings', 'personal_status_male single', 'property_magnitude_real estate', 'other_payment_plans_none', 'housing_own', 'own_telephone_yes', 'foreign_worker_yes']
com3 = ['checking_status_<0', 'duration', 'credit_amount', 'installment_commitment', 'age_bin_25-35', 'age_bin_35-50', 'age_bin_50-65', 'age_bin_65+', 'checking_status_no checking', 'credit_history_critical/other existing credit', 'credit_history_delayed previously', 'credit_history_existing paid', 'purpose_education', 'purpose_new car', 'savings_status_<100', 'housing_own', 'foreign_worker_yes']
com5 = ['duration', 'credit_amount', 'employment', 'installment_commitment', 'age_bin_25-35', 'age_bin_35-50', 'age_bin_50-65', 'checking_status_<0', 'checking_status_>=200', 'checking_status_no checking', 'credit_history_critical/other existing credit', 'credit_history_delayed previously', 'credit_history_existing paid', 'purpose_education', 'purpose_furniture/equipment', 'purpose_new car', 'purpose_radio/tv', 'purpose_used car', 'savings_status_<100', 'personal_status_male single', 'property_magnitude_no known property', 'property_magnitude_real estate', 'other_payment_plans_none', 'housing_rent', 'foreign_worker_yes']
com6 = ['duration', 'credit_amount', 'employment', 'installment_commitment', 'age_bin_25-35', 'age_bin_35-50', 'age_bin_50-65', 'checking_status_<0', 'checking_status_no checking', 'credit_history_critical/other existing credit', 'credit_history_delayed previously', 'credit_history_existing paid', 'purpose_education', 'purpose_furniture/equipment', 'purpose_new car', 'purpose_radio/tv', 'purpose_used car', 'savings_status_no known savings', 'personal_status_male single', 'housing_own', 'job_skilled', 'job_unemp/unskilled non res', 'job_unskilled resident', 'own_telephone_yes', 'foreign_worker_yes']
com7 = ['credit_amount', 'employment', 'installment_commitment', 'existing_credits', 'age_bin_25-35', 'age_bin_35-50', 'age_bin_50-65', 'age_bin_65+', 'checking_status_<0', 'checking_status_no checking', 'credit_history_critical/other existing credit', 'credit_history_delayed previously', 'credit_history_existing paid', 'purpose_education', 'purpose_furniture/equipment', 'purpose_new car', 'purpose_radio/tv', 'purpose_retraining', 'purpose_used car', 'savings_status_no known savings', 'personal_status_male single', 'property_magnitude_no known property', 'housing_own', 'own_telephone_yes', 'foreign_worker_yes']
com8 = ['duration', 'purpose_education', 'checking_status_<0', 'age_bin_35-50', 'checking_status_no checking', 'credit_amount', 'purpose_used car', 'property_magnitude_no known property', 'purpose_new car', 'savings_status_<100', 'credit_history_critical/other existing credit', 'employment', 'savings_status_500<=X<1000', 'other_payment_plans_stores', 'age_bin_65+', 'savings_status_no known savings', 'housing_rent', 'savings_status_>=1000', 'checking_status_>=200', 'other_payment_plans_none']

features_list = [com1, com3, com5, com6, com7, com8]

# ----------------------------------------------
# STORE METRICS OVER 50 RUNS
# ----------------------------------------------

results = {f"com{i+1}": {
                "Precision": [],
                "Recall": [],
                "F1": [],
                "Macro_F1": [],
                "Accuracy": []
            } for i in range(len(features_list))}

for run in range(50):
    print(f"\n\n========== RUN {run+1} / 50 ==========\n")

    X_train, X_test, y_train, y_test = preprocess_train_test(df, target_col="class")

    for i, feature in enumerate(features_list):
        key = f"com{i+1}"

        print(f"\n---- Feature Set {i+1} ----")
        # feature = feature + ['credit_density', 'debt_pressure',  'liquidity_risk']

        model, scaler = train_final_model(X_train, y_train, feature, StandardScaler())
        metrics = evaluate_final_model(model, scaler, X_test, y_test, feature)

        # Store metrics
        for metric_name, metric_value in metrics.items():
            results[key][metric_name].append(metric_value)


print("\n\n\n==================== FINAL MEAN METRICS ACROSS 50 RUNS ====================\n")

for i, feature in enumerate(features_list):
    key = f"com{i+1}"
    print(f"\n--- Feature Combination {i+1} ---")
    for metric_name, values in results[key].items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{metric_name:12s}: {mean_val:.4f} ± {std_val:.4f}")
