import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA  # kept in case you want later, not used below
from sklearn.feature_selection import RFE, f_regression, SelectKBest, chi2
from sklearn.impute import KNNImputer
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

DECORATOR = "=" * 120

# =============================================================================
# PREPROCESSING (LEAK-FREE)
# =============================================================================


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

    # Ensure numeric dtype
    X_train["employment"] = pd.to_numeric(X_train["employment"])
    X_test["employment"] = pd.to_numeric(X_test["employment"])
    return X_train, X_test


def impute_train_test(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Leak-free imputation:
      - Categorical: mode computed from training only, applied to both.
      - Numeric: single global KNNImputer fitted on training numeric columns only.
    """
    # ----- CATEGORICAL IMPUTATION (mode) -----
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        mode_val = X_train[col].mode(dropna=True)
        if len(mode_val) == 0:
            continue
        mode_val = mode_val[0]
        X_train[col] = X_train[col].fillna(mode_val)
        if col in X_test.columns:
            X_test[col] = X_test[col].fillna(mode_val)

    # ----- NUMERIC IMPUTATION (KNN on all numeric columns) -----
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

    X_train = pd.concat(
        [X_train.reset_index(drop=True), age_dummies_train.reset_index(drop=True)],
        axis=1,
    )
    X_test = pd.concat(
        [X_test.reset_index(drop=True), age_dummies_test.reset_index(drop=True)],
        axis=1,
    )

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
        test_size=0.2,
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

    print("\nFinal preprocessed shapes:")
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")

    return X_train, X_test, y_train, y_test


# =============================================================================
# RANKERS (NO CV INSIDE)
# =============================================================================


def get_rfe_ranks(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """
    RFE ranking: lower rank = more important
    """
    model = LogisticRegression(solver="liblinear", max_iter=2000)
    rfe = RFE(model, n_features_to_select=1, step=1)
    rfe.fit(X, y)
    return pd.Series(rfe.ranking_, index=X.columns)


def get_ffs_ranks(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """
    Forward feature selection-like ranking using f_regression scores.
    Higher F-statistic = better → lower rank number.
    """
    f_stat, _ = f_regression(X, y)
    scores = pd.Series(f_stat, index=X.columns)
    # rank: 1 = best (highest F-stat)
    ranks = scores.rank(ascending=False, method="dense")
    return ranks


def get_chi2_ranks(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """
    Chi² ranking: higher chi² score = better → lower rank number.
    Requires non-negative features.
    """
    X_pos = X - X.min() + 1e-6
    selector = SelectKBest(score_func=chi2, k=X_pos.shape[1])
    selector.fit(X_pos, y)
    scores = pd.Series(selector.scores_, index=X.columns)
    ranks = scores.rank(ascending=False, method="dense")
    return ranks


def aggregate_ranks(X: pd.DataFrame, y: pd.Series, scaler) -> pd.DataFrame:
    """
    Aggregate RFE, f_regression, chi² ranks into one combined rank.
    """
    print("\n" + DECORATOR)
    print("BUILDING RANK AGGREGATION (RFE + FFS + CHI²)")
    print(DECORATOR + "\n")

    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    rfe_ranks = get_rfe_ranks(X_scaled, y)
    ffs_ranks = get_ffs_ranks(X_scaled, y)
    chi2_ranks = get_chi2_ranks(X_scaled, y)

    rank_df = pd.DataFrame({
        "feature": X.columns,
        "rfe_rank": rfe_ranks,
        "ffs_rank": ffs_ranks,
        "chi2_rank": chi2_ranks,
    })

    rank_df["agg_rank"] = rank_df[["rfe_rank", "ffs_rank", "chi2_rank"]].mean(axis=1)
    rank_df = rank_df.sort_values("agg_rank").reset_index(drop=True)

    print("Top 15 aggregated features:\n")
    print(rank_df.head(15))

    return rank_df


# =============================================================================
# CV EVALUATION OF A FIXED FEATURE SUBSET (LEAK-FREE)
# =============================================================================


def evaluate_feature_subset_cv(X: pd.DataFrame, y: pd.Series, features, scaler_class=StandardScaler):
    """
    Leak-free CV evaluation of a fixed feature subset:
       - For each fold:
          - fit scaler on X_train[features]
          - transform train/test
          - SMOTE on train only
          - logistic regression
          - evaluate on test
    """
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    f1s = []
    precisions = []
    recalls = []
    accs = []

    for train_idx, test_idx in kf.split(X, y):
        X_train_fold = X.iloc[train_idx][features].copy()
        X_test_fold = X.iloc[test_idx][features].copy()
        y_train_fold = y.iloc[train_idx].copy()
        y_test_fold = y.iloc[test_idx].copy()

        # Fresh scaler instance per fold (no reuse outside)
        scaler = scaler_class()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_test_scaled = scaler.transform(X_test_fold)

        # SMOTE on training only
        X_sm, y_sm = SMOTE(random_state=42).fit_resample(X_train_scaled, y_train_fold)

        model = LogisticRegression(solver="liblinear", max_iter=2000)
        model.fit(X_sm, y_sm)

        y_pred = model.predict(X_test_scaled)

        f1s.append(f1_score(y_test_fold, y_pred))
        precisions.append(precision_score(y_test_fold, y_pred))
        recalls.append(recall_score(y_test_fold, y_pred))
        accs.append(accuracy_score(y_test_fold, y_pred))

    return {
        "F1": np.mean(f1s),
        "Precision": np.mean(precisions),
        "Recall": np.mean(recalls),
        "Accuracy": np.mean(accs),
    }


def search_best_topK(X: pd.DataFrame, y: pd.Series, rank_df: pd.DataFrame,
                     k_min: int = 5, k_max: int = 30):
    """
    Use aggregated ranking as GUIDE, and search top-K for K in [k_min, k_max]
    using 10-fold CV F1.
    """
    print("\n" + DECORATOR)
    print("SEARCHING BEST TOP-K USING CV")
    print(DECORATOR + "\n")

    features_sorted = rank_df["feature"].tolist()
    max_k = min(k_max, len(features_sorted))

    results = []

    for k in range(k_min, max_k + 1):
        subset = features_sorted[:k]
        print(f"Evaluating K = {k} with first {k} aggregated features...")
        metrics = evaluate_feature_subset_cv(X, y, subset, scaler_class=StandardScaler)
        print(f"  -> F1={metrics['F1']:.4f}, "
              f"Precision={metrics['Precision']:.4f}, "
              f"Recall={metrics['Recall']:.4f}, "
              f"Accuracy={metrics['Accuracy']:.4f}\n")
        results.append((k, metrics["F1"], subset))

    # Pick best K by F1
    best_k, best_f1, best_features = max(results, key=lambda t: t[1])

    print(DECORATOR)
    print(f"BEST TOP-K FOUND: K={best_k}, CV F1={best_f1:.4f}")
    print("Best feature subset:")
    print(best_features)
    print(DECORATOR + "\n")

    return best_k, best_features, results


# =============================================================================
# FINAL TRAINING / EVAL (USING BEST SUBSET)
# =============================================================================


def train_final_model(X_train: pd.DataFrame, y_train: pd.Series, features):
    """
    Train final model on full training data:
      - scale on full training subset
      - SMOTE
      - logistic regression
    """
    X_selected = X_train[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)

    X_sm, y_sm = SMOTE(random_state=42).fit_resample(X_scaled, y_train)

    model = LogisticRegression(solver="liblinear", max_iter=2000)
    model.fit(X_sm, y_sm)

    return model, scaler


def evaluate_final_model(model, scaler, X_test: pd.DataFrame, y_test: pd.Series, features):
    X_selected = X_test[features]
    X_scaled = scaler.transform(X_selected)
    y_pred = model.predict(X_scaled)

    return {
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "Accuracy": accuracy_score(y_test, y_pred),
    }


# =============================================================================
# MAIN
# =============================================================================


def main():
    # ---- Load raw data ----
    df = pd.read_csv("data/Credit_Train.csv")

    # ---- Preprocess (leak-free) ----
    X_train, X_test, y_train, y_test = preprocess_train_test(df, target_col="class")

    # ---- Rank aggregation on TRAIN only ----
    rank_df = aggregate_ranks(X_train, y_train, StandardScaler())

    # ---- Search K using CV on TRAIN only ----
    best_k, best_features, _ = search_best_topK(
        X_train, y_train, rank_df, k_min=5, k_max=30
    )

    # ---- Train final model on TRAIN with best K ----
    print("\n" + DECORATOR)
    print("TRAINING FINAL MODEL WITH BEST FEATURE SUBSET")
    print(DECORATOR + "\n")
    print(f"Using K={best_k} features:\n{best_features}\n")

    model, scaler = train_final_model(X_train, y_train, best_features)

    # ---- Evaluate on HELD-OUT TEST ----
    print("\nEVALUATING FINAL MODEL ON TEST SET...\n")
    test_metrics = evaluate_final_model(model, scaler, X_test, y_test, best_features)

    print("\n" + DECORATOR)
    print("FINAL MODEL TEST METRICS")
    print(DECORATOR)
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
