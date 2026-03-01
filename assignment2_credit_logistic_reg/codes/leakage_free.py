import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pandas.core.dtypes.common import is_numeric_dtype

from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE, RFECV, f_regression, SelectKBest, chi2
from sklearn.impute import KNNImputer
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from imblearn.over_sampling import SMOTE

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

DECORATOR = "=" * 120


# =============================================================================
#  PREPROCESSING (LEAK-FREE: ALWAYS FIT ON TRAIN, APPLY TO TEST)
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

    print("\nFinal preprocessed shapes:")
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

    X_train.to_csv('data/train.csv', index=False)
    y_train.to_csv('data/label_train.csv', index=False)
    X_test.to_csv('data/test.csv', index=False)
    y_test.to_csv('data/label_test.csv', index=False)
    return X_train, X_test, y_train, y_test


# =============================================================================
# OPTIMAL FEATURE SELECTION METHODS
# =============================================================================
def select_features_rfe(X, y, n_features):
    rfe = RFE(LogisticRegression(solver='liblinear', max_iter=1000, random_state=42),
              n_features_to_select=n_features)
    rfe.fit(X, y)
    selected_features = X.keys()[rfe.support_]
    return selected_features

def select_features_ffs(X, y, n_features):
    f_stat, _ = f_regression(X, y)
    df_ffs = pd.DataFrame({
        'feature': X.columns,
        'f_stat': f_stat
    })
    df_ffs.sort_values(by=['f_stat'], ascending=False, inplace=True)
    top_features = df_ffs['feature'][:n_features]
    return list(top_features)

def select_features_chi2(X, y, n_features):
    # shift to avoid negative input as chi2 require non-negative values
    X_pos = X - X.min() + 1e-6
    selector = SelectKBest(score_func=chi2, k=n_features)
    selector.fit(X_pos, y)
    return list(X.columns[selector.get_support()])


def rfecv_feature_selection(X, y, scaler):
    """
    Use RFECV to automatically find optimal number of features.
    This performs CV internally with no leakage.
    """
    print("\n" + DECORATOR)
    print("RFECV: AUTOMATIC OPTIMAL FEATURE SELECTION")
    print(DECORATOR + "\n")

    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    # Apply SMOTE once on full training data for RFECV
    X_smote, y_smote = SMOTE(random_state=42).fit_resample(X_scaled_df, y)

    rfecv = RFECV(
        estimator=LogisticRegression(solver='liblinear', max_iter=2000, random_state=42),
        step=1,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring='f1',
        n_jobs=-1,
        min_features_to_select=5
    )

    rfecv.fit(X_smote, y_smote)

    selected_features = list(X.columns[rfecv.support_])

    print(f"Optimal number of features: {rfecv.n_features_}")
    print(f"Grid scores: {rfecv.cv_results_['mean_test_score'][:20]}")
    print(f"\nSelected features: {selected_features}\n")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1),
             rfecv.cv_results_['mean_test_score'])
    plt.xlabel('Number of Features')
    plt.ylabel('Cross-Validation F1 Score')
    plt.title('RFECV: Optimal Feature Selection')
    plt.axvline(rfecv.n_features_, color='r', linestyle='--',
                label=f'Optimal = {rfecv.n_features_}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return selected_features


def sequential_forward_selection(X, y, scaler, max_features=25):
    """
    Sequential Forward Selection: greedily add features that improve CV score.
    No leakage - features selected via proper CV.
    """
    print("\n" + DECORATOR)
    print("SEQUENTIAL FORWARD SELECTION")
    print(DECORATOR + "\n")

    selected_features = []
    remaining_features = list(X.columns)

    best_score = 0
    scores_history = []

    for iteration in range(min(max_features, len(X.columns))):
        scores = []

        print(f"Iteration {iteration + 1}: Testing {len(remaining_features)} candidates...")

        for feature in remaining_features:
            candidate_features = selected_features + [feature]

            # Evaluate via proper CV
            score = evaluate_feature_subset_cv(X, y, candidate_features, scaler)['F1']
            scores.append((feature, score))

        # Pick best feature
        best_feature, best_new_score = max(scores, key=lambda x: x[1])

        # Stop if no improvement (with small tolerance)
        if best_new_score <= best_score + 0.001:
            print(f"\nStopping: No improvement (current={best_new_score:.4f}, previous={best_score:.4f})")
            break

        selected_features.append(best_feature)
        remaining_features.remove(best_feature)
        best_score = best_new_score
        scores_history.append(best_score)

        print(f"  Added '{best_feature}', F1={best_new_score:.4f}, Total features={len(selected_features)}")

    # Plot progress
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(scores_history) + 1), scores_history, marker='o')
    plt.xlabel('Number of Features')
    plt.ylabel('Cross-Validation F1 Score')
    plt.title('Sequential Forward Selection Progress')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"\nFinal selected features ({len(selected_features)}): {selected_features}\n")

    return selected_features


def evaluate_feature_subset_cv(X: pd.DataFrame, y: pd.Series, features, scaler):
    """
    Leak-free CV evaluation of a fixed feature subset.
    For each fold:
      - fit scaler on X_train[features]
      - transform train/test
      - SMOTE on train only
      - logistic regression
      - evaluate on test
    """
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    f1s = []
    precisions = []
    recalls = []
    accs = []

    for train_idx, test_idx in kf.split(X, y):
        X_train_fold = X.iloc[train_idx][features].copy()
        X_test_fold = X.iloc[test_idx][features].copy()
        y_train_fold = y.iloc[train_idx].copy()
        y_test_fold = y.iloc[test_idx].copy()

        # Scale using training fold only
        scaler_fold = type(scaler)()  # Create fresh scaler instance
        X_train_scaled = scaler_fold.fit_transform(X_train_fold)
        X_test_scaled = scaler_fold.transform(X_test_fold)

        # SMOTE on training only
        X_sm, y_sm = SMOTE(random_state=42).fit_resample(X_train_scaled, y_train_fold)

        model = LogisticRegression(solver="liblinear", max_iter=2000, random_state=42)
        model.fit(X_sm, y_sm)

        y_pred = model.predict(X_test_scaled)

        f1s.append(f1_score(y_test_fold, y_pred, average='macro'))
        precisions.append(precision_score(y_test_fold, y_pred))
        recalls.append(recall_score(y_test_fold, y_pred))
        accs.append(accuracy_score(y_test_fold, y_pred))

    return {
        "F1": np.mean(f1s),
        "Precision": np.mean(precisions),
        "Recall": np.mean(recalls),
        "Accuracy": np.mean(accs),
    }


def compare_feature_selection_methods(X_train, y_train, scaler):
    """
    Compare RFECV vs Sequential Forward Selection.
    Returns results from both methods.
    """
    print("\n" + DECORATOR)
    print("COMPARING FEATURE SELECTION METHODS")
    print(DECORATOR + "\n")

    # Method 1: RFECV
    rfecv_features = rfecv_feature_selection(X_train, y_train, scaler)
    rfecv_score = evaluate_feature_subset_cv(X_train, y_train, rfecv_features, scaler)

    print(f"RFECV Results:")
    print(f"  Features: {len(rfecv_features)}")
    print(f"  Macro F1: {rfecv_score['F1']:.4f}")
    print(f"  Precision: {rfecv_score['Precision']:.4f}")
    print(f"  Recall: {rfecv_score['Recall']:.4f}\n")

    # Method 2: Sequential Forward Selection
    sfs_features = sequential_forward_selection(X_train, y_train, scaler, max_features=25)
    sfs_score = evaluate_feature_subset_cv(X_train, y_train, sfs_features, scaler)

    print(f"Sequential Forward Selection Results:")
    print(f"  Features: {len(sfs_features)}")
    print(f"  Macro F1: {sfs_score['F1']:.4f}")
    print(f"  Precision: {sfs_score['Precision']:.4f}")
    print(f"  Recall: {sfs_score['Recall']:.4f}\n")

    # Choose best
    if rfecv_score['F1'] >= sfs_score['F1']:
        print(f"Winner: RFECV (F1={rfecv_score['F1']:.4f})")
        return rfecv_features, rfecv_score, 'RFECV'
    else:
        print(f"Winner: Sequential Forward Selection (F1={sfs_score['F1']:.4f})")
        return sfs_features, sfs_score, 'SFS'


# =============================================================================
# FINAL TRAINING / EVALUATION
# =============================================================================

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
    probs = model.predict_proba(X_scaled)

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


def build_feature_pool_guide(X_train, y_train, scaler, top_k=20):
    """
    Use RFE, FFS, Chi2 as GUIDES (not in CV) to build a pool of promising features.
    This runs ONCE on full training data - these are just recommendations.
    """
    print("\n" + DECORATOR)
    print("BUILDING FEATURE POOL (GUIDES ONLY - NO CV)")
    print(DECORATOR + "\n")

    # Scale once for all selectors
    X_scaled = scaler.fit_transform(X_train)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X_train.columns)

    pool = {}

    # Guide 1: RFE top features
    print(f"Running RFE (top {top_k})...")
    rfe_features = select_features_rfe(X_scaled_df, y_train, top_k)
    pool['RFE'] = rfe_features
    print(f"  RFE suggests: {rfe_features}\n")

    # Guide 2: FFS top features
    print(f"Running FFS (top {top_k})...")
    ffs_features = select_features_ffs(X_scaled_df, y_train, top_k)
    pool['FFS'] = ffs_features
    print(f"  FFS suggests: {ffs_features}\n")

    # Guide 3: Chi2 top features
    print(f"Running Chi2 (top {top_k})...")
    chi2_features = select_features_chi2(X_scaled_df, y_train, top_k)
    pool['Chi2'] = chi2_features
    print(f"  Chi2 suggests: {chi2_features}\n")

    # Combined pool (union of all suggestions)
    all_features = set(rfe_features) | set(ffs_features) | set(chi2_features)

    # Feature frequency analysis
    feature_counts = {}
    for feature in all_features:
        count = sum([
            1 if feature in rfe_features else 0,
            1 if feature in ffs_features else 0,
            1 if feature in chi2_features else 0
        ])
        feature_counts[feature] = count

    # Sort by frequency (how many methods selected it)
    sorted_pool = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)

    print("Feature Pool Analysis:")
    print("-" * 80)
    print(f"{'Feature':<40s} {'Selected by (count)'}")
    print("-" * 80)
    for feature, count in sorted_pool:
        methods = []
        if feature in rfe_features:
            methods.append('RFE')
        if feature in ffs_features:
            methods.append('FFS')
        if feature in chi2_features:
            methods.append('Chi2')
        print(f"{feature:<40s} {', '.join(methods)} ({count}/3)")

    print(f"\nTotal features in pool: {len(all_features)}")
    print(f"Features selected by all 3 methods: {sum(1 for c in feature_counts.values() if c == 3)}")
    print(f"Features selected by 2+ methods: {sum(1 for c in feature_counts.values() if c >= 2)}")

    return pool, list(all_features), sorted_pool


def experiment_with_combinations(X_train, y_train, feature_pool, sorted_pool, scaler):
    """
    Manually experiment with different feature combinations from the pool.
    """
    print("\n" + DECORATOR)
    print("EXPERIMENTING WITH FEATURE COMBINATIONS")
    print(DECORATOR + "\n")

    experiments = []

    # Experiment 1: Features selected by all 3 methods (highest confidence)
    consensus_features = [f for f, count in sorted_pool if count == 3]
    if len(consensus_features) >= 3:
        print(f"Experiment 1: Consensus features (selected by all 3 methods, n={len(consensus_features)})")
        score = evaluate_feature_subset_cv(X_train, y_train, consensus_features, scaler)
        experiments.append({
            'name': 'Consensus (3/3)',
            'features': consensus_features,
            'score': score['F1']
        })
        print(f"  Macro F1: {score['F1']:.4f}\n")

    # Experiment 2: Features selected by 2+ methods (medium confidence)
    high_confidence = [f for f, count in sorted_pool if count >= 2]
    if len(high_confidence) >= 5:
        print(f"Experiment 2: High confidence features (2+ methods, n={len(high_confidence)})")
        score = evaluate_feature_subset_cv(X_train, y_train, high_confidence, scaler)
        experiments.append({
            'name': 'High Confidence (2+/3)',
            'features': high_confidence,
            'score': score['F1']
        })
        print(f"  Macro F1: {score['F1']:.4f}\n")

    # Experiment 3: Top 10 from pool (by frequency)
    top_10 = [f for f, count in sorted_pool[:10]]
    print(f"Experiment 3: Top 10 by frequency")
    score = evaluate_feature_subset_cv(X_train, y_train, top_10, scaler)
    experiments.append({
        'name': 'Top 10',
        'features': top_10,
        'score': score['F1']
    })
    print(f"  Macro F1: {score['F1']:.4f}\n")

    # Experiment 4: Top 15 from pool
    top_15 = [f for f, count in sorted_pool[:15]]
    print(f"Experiment 4: Top 15 by frequency")
    score = evaluate_feature_subset_cv(X_train, y_train, top_15, scaler)
    experiments.append({
        'name': 'Top 15',
        'features': top_15,
        'score': score['F1']
    })
    print(f"  Macro F1: {score['F1']:.4f}\n")

    # Experiment 5: Top 20 from pool
    top_20 = [f for f, count in sorted_pool[:20]]
    print(f"Experiment 5: Top 20 by frequency")
    score = evaluate_feature_subset_cv(X_train, y_train, top_20, scaler)
    experiments.append({
        'name': 'Top 20',
        'features': top_20,
        'score': score['F1']
    })
    print(f"  Macro F1: {score['F1']:.4f}\n")

    # Experiment 6: RFE suggestions only
    print(f"Experiment 6: RFE features only (n={len(feature_pool['RFE'])})")
    score = evaluate_feature_subset_cv(X_train, y_train, feature_pool['RFE'], scaler)
    experiments.append({
        'name': 'RFE Only',
        'features': feature_pool['RFE'],
        'score': score['F1']
    })
    print(f"  Macro F1: {score['F1']:.4f}\n")

    # Experiment 7: FFS suggestions only
    print(f"Experiment 7: FFS features only (n={len(feature_pool['FFS'])})")
    score = evaluate_feature_subset_cv(X_train, y_train, feature_pool['FFS'], scaler)
    experiments.append({
        'name': 'FFS Only',
        'features': feature_pool['FFS'],
        'score': score['F1']
    })
    print(f"  Macro F1: {score['F1']:.4f}\n")

    # Experiment 8: Chi2 suggestions only
    print(f"Experiment 8: Chi2 features only (n={len(feature_pool['Chi2'])})")
    score = evaluate_feature_subset_cv(X_train, y_train, feature_pool['Chi2'], scaler)
    experiments.append({
        'name': 'Chi2 Only',
        'features': feature_pool['Chi2'],
        'score': score['F1']
    })
    print(f"  Macro F1: {score['F1']:.4f}\n")

    # Experiment 9: Intersection of RFE and FFS
    rfe_ffs_intersection = list(set(feature_pool['RFE']) & set(feature_pool['FFS']))
    if len(rfe_ffs_intersection) >= 5:
        print(f"Experiment 9: RFE ∩ FFS (n={len(rfe_ffs_intersection)})")
        score = evaluate_feature_subset_cv(X_train, y_train, rfe_ffs_intersection, scaler)
        experiments.append({
            'name': 'RFE ∩ FFS',
            'features': rfe_ffs_intersection,
            'score': score['F1']
        })
        print(f"  Macro F1: {score['F1']:.4f}\n")

    # Experiment 10: Union of top 10 from each method
    rfe_top10 = feature_pool['RFE'][:20]
    ffs_top10 = feature_pool['FFS'][:20]
    chi2_top10 = feature_pool['Chi2'][:20]
    union_top10 = list(set(rfe_top10) | set(ffs_top10) | set(chi2_top10))
    print(f"Experiment 10: Union of top 10 from each method (n={len(union_top10)})")
    score = evaluate_feature_subset_cv(X_train, y_train, union_top10, scaler)
    experiments.append({
        'name': 'Union Top 10',
        'features': union_top10,
        'score': score['F1']
    })
    print(f"  Macro F1: {score['F1']:.4f}\n")

    # Sort experiments by score
    experiments.sort(key=lambda x: x['score'], reverse=True)

    # Summary
    print(DECORATOR)
    print("EXPERIMENT RESULTS SUMMARY")
    print(DECORATOR + "\n")

    for i, exp in enumerate(experiments, 1):
        print(f"{i}. {exp['name']:25s} | F1={exp['score']:.4f} | n={len(exp['features'])}")

    print(f"\nBEST: {experiments[0]['name']} (F1={experiments[0]['score']:.4f})")

    return experiments[0]['features'], experiments[0]['score'], experiments


# In main(), replace with:
def main():
    # Load and preprocess
    df = pd.read_csv("data/Credit_Train.csv")
    X_train, X_test, y_train, y_test = preprocess_train_test(df, target_col="class")

    scaler = StandardScaler()

    # Step 1: Build feature pool using guides (NO CV HERE)
    pool, all_features, sorted_pool = build_feature_pool_guide(X_train, y_train, scaler, top_k=25)

    # Step 2: Experiment with different combinations (CV HAPPENS HERE)
    best_features, cv_score, all_experiments = experiment_with_combinations(
        X_train, y_train, pool, sorted_pool, scaler
    )

    # best_features = ['duration', 'credit_amount', 'employment', 'installment_commitment', 'existing_credits', 'age_bin_25-35', 'age_bin_35-50', 'age_bin_50-65', 'age_bin_65+', 'checking_status_<0', 'checking_status_no checking', 'credit_history_critical/other existing credit', 'credit_history_delayed previously', 'credit_history_existing paid', 'credit_history_no credits/all paid', 'purpose_education', 'purpose_new car', 'purpose_used car', 'savings_status_>=1000', 'savings_status_no known savings', 'personal_status_male single', 'property_magnitude_real estate', 'other_payment_plans_none', 'housing_own', 'own_telephone_yes', 'foreign_worker_yes']
    # best_features = ['checking_status_no checking', 'housing_own', 'checking_status_<0', 'savings_status_>=1000', 'savings_status_no known savings', 'credit_amount', 'savings_status_<100', 'foreign_worker_yes', 'credit_history_no credits/all paid', 'purpose_education', 'installment_commitment', 'age_bin_35-50', 'credit_history_critical/other existing credit', 'purpose_new car', 'property_magnitude_real estate', 'employment', 'other_payment_plans_none', 'own_telephone_yes', 'duration', 'personal_status_male single']
    # best_features = ['credit_amount', 'installment_commitment', 'age_bin_25-35', 'age_bin_35-50', 'age_bin_50-65', 'age_bin_65+', 'checking_status_no checking', 'credit_history_critical/other existing credit', 'credit_history_delayed previously', 'credit_history_existing paid', 'purpose_education', 'purpose_new car', 'savings_status_<100', 'housing_own', 'foreign_worker_yes']
    # best_features = ['purpose_education', 'housing_own', 'credit_amount', 'credit_history_critical/other existing credit', 'duration', 'savings_status_<100', 'checking_status_<0', 'age_bin_50-65', 'checking_status_no checking', 'age_bin_35-50', 'savings_status_no known savings', 'employment', 'purpose_new car', 'property_magnitude_no known property', 'other_parties_guarantor']


    # Step 3: Train final model with best combination
    print("\n" + DECORATOR)
    print("TRAINING FINAL MODEL WITH BEST FEATURE COMBINATION")
    print(DECORATOR + "\n")

    print(f"Using {len(best_features)} features: {best_features}\n")

    final_model, fitted_scaler = train_final_model(X_train, y_train, best_features, StandardScaler())
    final_test_metrics = evaluate_final_model(final_model, fitted_scaler, X_test, y_test, best_features)

    print("\n" + DECORATOR)
    print("FINAL MODEL TEST METRICS")
    print(DECORATOR)
    for k, v in final_test_metrics.items():
        print(f"{k}: {v:.4f}")

    # Optional: PCA comparison
    # ... (keep PCA code if you want)


if __name__ == "__main__":
    main()