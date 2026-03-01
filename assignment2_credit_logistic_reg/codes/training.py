import numpy as np
import pandas as pd
import pickle
from sklearn.feature_selection import RFE, RFECV, f_regression, SelectKBest, chi2
from sklearn.impute import KNNImputer
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, OneHotEncoder
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
        ">=7": 4
    }

    X_train['employment'] = X_train['employment'].map(employment_map)
    X_test['employment'] = X_test['employment'].map(employment_map)

    X_train['employment'] = pd.to_numeric(X_train['employment'])
    X_test['employment'] = pd.to_numeric(X_test['employment'])
    return X_train, X_test


def impute_nan_columns(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    - Categorical: mode computed from training only, applied to both.
    - Numeric: KNNImputer fitted on training numeric columns only.
    """

    # Categorical imputation (mode)
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns
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


def bin_age_column(X_train: pd.DataFrame, X_test: pd.DataFrame):
    if "age" not in X_train.columns:
        return X_train, X_test

    bins = [18, 25, 35, 50, 65, 100]
    labels = ["18-25", "25-35", "35-50", "50-65", "65+"]

    age_bin_train = pd.cut(
        X_train['age'],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True
    )
    age_dummies_train = pd.get_dummies(
        age_bin_train, prefix='age_bin', drop_first=True, dtype=int
    )

    age_bin_test = pd.cut(
        X_test['age'],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True
    )
    age_dummies_test = pd.get_dummies(
        age_bin_test, prefix='age_bin', drop_first=True, dtype=int
    )

    X_train = X_train.drop(columns=['age'])
    X_test = X_test.drop(columns=['age'])

    X_train = pd.concat([X_train.reset_index(drop=True), age_dummies_train.reset_index(drop=True)], axis=1)
    X_test = pd.concat([X_test.reset_index(drop=True), age_dummies_test.reset_index(drop=True)], axis=1)

    return X_train, X_test


def create_dummies(X_train: pd.DataFrame, X_test: pd.DataFrame):
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns

    encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    encoder.fit(X_train[cat_cols])

    train_cat_encoded = encoder.transform(X_train[cat_cols])
    test_cat_encoded = encoder.transform(X_test[cat_cols])
    encoded_cols = encoder.get_feature_names_out(cat_cols)

    # create DataFrame with encoded features
    df_train_cat = pd.DataFrame(train_cat_encoded, columns=encoded_cols)
    df_test_cat = pd.DataFrame(test_cat_encoded, columns=encoded_cols)

    num_cols = [col for col in X_train.columns if col not in cat_cols]

    X_train_final = pd.concat(
        [X_train[num_cols].reset_index(drop=True), df_train_cat.reset_index(drop=True)],
        axis=1
    )
    X_test_final = pd.concat(
        [X_test[num_cols].reset_index(drop=True), df_test_cat.reset_index(drop=True)],
        axis=1
    )

    X_train_final, X_test_final = X_train_final.align(
        X_test_final, join='left', axis=1, fill_value=0
    )

    return X_train_final, X_test_final


def preprocess_full_dataset(df: pd.DataFrame, target_col: str = 'class'):
    """
        2. Impute cat (mode) + numeric (KNN) on train, apply to test.
        3. Encode ordinal 'employment' with fixed mapping.
        4. Add fixed age bins -> dummy.
        5. Create dummy variables for all other categoricals.
    Returns full dataset as features and target
    """
    df = df.copy()
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    # ==================================================================
    # Encode ordinal "employment"
    # ==================================================================
    # X.setdefault("employment", "unemployed")

    employment_map = {
        "<1": 0,
        "unemployed": 1,
        "1<=X<4": 2,
        "4<=X<7": 3,
        ">=7": 4
    }
    X['employment'] = X['employment'].map(employment_map)
    X['employment'] = pd.to_numeric(X['employment'])

    # ==================================================================
    # Impute NaN columns
    # ==================================================================
    # Categorical imputation (mode)
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        mode_val = X[col].mode(dropna=True)
        if len(mode_val) == 0:
            continue
        mode_val = mode_val[0]
        X[col] = X[col].fillna(mode_val)

    # Numeric imputation (KNN on all numeric columns)
    num_cols = X.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        imputer = KNNImputer(n_neighbors=5)
        X_num = imputer.fit_transform(X[num_cols])

        X[num_cols] = X_num

    # ==================================================================
    # Bin "age"
    # ==================================================================
    if "age" in X.columns:
        bins = [18, 25, 35, 50, 65, 100]
        labels = ["18-25", "25-35", "35-50", "50-65", "65+"]

        age_bin = pd.cut(
            X['age'],
            bins=bins,
            labels=labels,
            include_lowest=True,
            right=True
        )
        age_dummies = pd.get_dummies(
            age_bin, prefix='age_bin', drop_first=True, dtype=int
        )

        X = X.drop(columns=['age'])
        X = pd.concat([X.reset_index(drop=True), age_dummies.reset_index(drop=True)],
                      axis=1)

    # ==================================================================
    # Create dummies
    # ==================================================================
    cat_cols = X.select_dtypes(include=['object', 'category']).columns

    encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    encoder.fit(X[cat_cols])

    cat_encoded = encoder.transform(X[cat_cols])
    encoded_cols = encoder.get_feature_names_out(cat_cols)

    # create DataFrame with encoded features
    df_cat = pd.DataFrame(cat_encoded, columns=encoded_cols)

    num_cols = [col for col in X.columns if col not in cat_cols]

    X_final = pd.concat(
        [X[num_cols].reset_index(drop=True), df_cat.reset_index(drop=True)],
        axis=1
    )

    # X_final['credit_density'] = X_final['credit_amount'] * X_final['duration']

    df_new = pd.concat([y, X_final], axis=1)
    df_new.to_csv('data/df_new.csv', index=False)

    return X_final, y, encoder, imputer, employment_map, mode_val



def preprocess_train_test(df: pd.DataFrame, target_col: str = 'class'):
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

    X = df.drop(columns=[target_col]).copy()
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, shuffle=True, stratify=y
    )

    print('Imputing train/test...')
    X_train, X_test = impute_nan_columns(X_train, X_test)

    print('Encoding ordinal "employment"....')
    X_train, X_test = encode_ordinal_employment(X_train, X_test)

    print('Binning "age"....')
    X_train, X_test = bin_age_column(X_train, X_test)

    print('Creating dummies (train/test)....')
    X_train, X_test = create_dummies(X_train, X_test)

    # X_train['credit_density'] = X_train['credit_amount'] * (X_train['duration'])
    # X_test['credit_density'] = X_test['credit_amount'] * (X_test['duration'])

    print(f'\nFinal processing shape: X_train={X_train.shape} | X_test={X_test.shape}')

    X_train.to_csv('data/train_features.csv', index=False)
    X_test.to_csv('data/test_features.csv', index=False)
    y_train.to_csv('data/train_label.csv', index=False)
    y_test.to_csv('data/test_label.csv', index=False)
    return X_train, X_test, y_train, y_test


# =============================================================================
# OPTIMAL FEATURE SELECTION METHODS
# =============================================================================

def select_features_rfe(X, y, n_features):
    rfe = RFE(LogisticRegression(solver='liblinear', max_iter=2000, random_state=42),
              n_features_to_select=n_features)
    rfe.fit(X, y)
    return list(X.columns[rfe.support_])


def select_features_ffs(X, y, n_features):
    f_stats, _ = f_regression(X, y)
    df_ffs = pd.DataFrame({'features': X.columns, 'f_stats': f_stats})
    df_ffs.sort_values(by=['f_stats'], ascending=False, inplace=True)
    top_features = df_ffs['features'].head(n_features)
    return list(top_features)


def select_features_chi2(X, y, n_features):
    # Scale X values to avoid negative values as required by chi2
    X_scaled = X - X.min() + 1e-6
    selector = SelectKBest(score_func=chi2, k=n_features)
    selector.fit(X_scaled, y)
    return list(X.columns[selector.get_support()])


def build_features_pool(X, y, scaler, k):
    """
    Build a feature pool from top-ks of RFE, FFS, and Chi2
    """
    print('\n' + DECORATOR)
    print("BUILDING FEATURE POOL FROM RFE, FFS, AND CHI2")
    print(DECORATOR + '\n')

    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    top_k = k
    pool = set()

    rfe_features = select_features_rfe(X_scaled, y, top_k)
    print(f'RFE top {top_k} features:\n{rfe_features}\n')
    pool.update(rfe_features)

    ffs_features = select_features_ffs(X_scaled, y, top_k)
    print(f'FFS top {top_k} features:\n{ffs_features}\n')
    pool.update(ffs_features)

    chi2_features = select_features_chi2(X_scaled, y, top_k)
    print(f'Chi2 top {top_k} features:\n{chi2_features}\n')
    pool.update(chi2_features)

    pool = list(pool)
    print(f'\nTotal unique features in pool: {len(pool)}\n')
    return pool


def evaluate_feature_subset_cv(X, y, features, scaler):
    k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    f1s, precisions, recalls, accuracies = [], [], [], []

    for train_idx, test_idx in k_fold.split(X, y):
        X_train_fold = X.iloc[train_idx][features].copy()
        X_test_fold = X.iloc[test_idx][features].copy()
        y_train_fold = y.iloc[train_idx].copy()
        y_test_fold = y.iloc[test_idx].copy()

        # create new instance of scaler for cross-fold validation
        scaler_fold = type(scaler)()

        X_train_scaled = scaler_fold.fit_transform(X_train_fold)
        X_test_scaled = scaler_fold.transform(X_test_fold)

        X_smote, y_smote = SMOTE(random_state=42).fit_resample(X_train_scaled, y_train_fold)

        model = LogisticRegression(
            solver='liblinear',
            max_iter=2000,
            random_state=42,
            penalty='l2',
            class_weight={0: 1.0, 1: 3.0},
        )
        model.fit(X_smote, y_smote)

        y_pred = model.predict(X_test_scaled)

        f1s.append(f1_score(y_test_fold, y_pred))
        precisions.append(precision_score(y_test_fold, y_pred))
        recalls.append(recall_score(y_test_fold, y_pred))
        accuracies.append(accuracy_score(y_test_fold, y_pred))

    return {
        "F1": np.mean(f1s),
        "Precision": np.mean(precisions),
        "Recall": np.mean(recalls),
        "Accuracy": np.mean(accuracies),
    }


def build_feature_pool_experiments(X_train, y_train, scaler, top_k=20):
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
    print("\n" + DECORATOR)
    print("EXPERIMENTING WITH FEATURE COMBINATIONS")
    print(DECORATOR + "\n")

    experiments = []

    # Experiment 1: Features selected by all 3 methods (highest confidence)
    features_common = [f for f, count in sorted_pool if count == 3]
    if len(features_common) >= 3:
        print(f'Experiment 1: Most common features (selected by all 3 selection algorithm), '
              f'n={len(features_common)}')
        score = evaluate_feature_subset_cv(X_train, y_train, features_common, scaler)
        experiments.append({
            'name': 'Consensus (3/3)',
            'features': features_common,
            'score': score['F1']
        })
        print(f'   Class 1 F1: {score["F1"]:.4f}\n')

    # Experiment 2: Features selected by 2+ algorithms
    high_confidence = [f for f, count in sorted_pool if count >= 2]
    if len(high_confidence) >= 5:
        print(f'Experiment 2: High Confidence features (2+ algorithms), '
              f'n={len(high_confidence)}')
        score = evaluate_feature_subset_cv(X_train, y_train, high_confidence, scaler)
        experiments.append({
            'name': 'High Confidence (2+/3)',
            'features': high_confidence,
            'score': score['F1']
        })
        print(f'   Class 1 F1: {score["F1"]:.4f}\n')

    # Experiment 3: top 10 from pool
    top_10 = [f for f, count in sorted_pool[:10]]
    print(f'Experiment 3: Top 10 Features from Feature Pool')
    score = evaluate_feature_subset_cv(X_train, y_train, top_10, scaler)
    experiments.append({
        'name': 'Top 10',
        'features': top_10,
        'score': score['F1']
    })
    print(f"  Macro F1: {score['F1']:.4f}\n")

    # Experiment 4: Top 15
    top_15 = [f for f, count in sorted_pool[:15]]
    print(f'Experiment 4: Top 15 features from Feature Pool')
    score = evaluate_feature_subset_cv(X_train, y_train, top_15, scaler)
    experiments.append({
        'name': 'Top 15',
        'features': top_15,
        'score': score['F1']
    })
    print(f"  Macro F1: {score['F1']:.4f}\n")

    # Experiment 5: Top 20
    top_20 = [f for f, count in sorted_pool[:20]]
    print(f"Experiment 5: Top 20 by frequency")
    score = evaluate_feature_subset_cv(X_train, y_train, top_20, scaler)
    experiments.append({
        'name': 'Top 20',
        'features': top_20,
        'score': score['F1']
    })
    print(f"  Macro F1: {score['F1']:.4f}\n")

    # Experiment 6: RFE only
    print(f'Experiment 6: RFE features only (n={len(feature_pool["RFE"])})')
    score = evaluate_feature_subset_cv(X_train, y_train, feature_pool['RFE'], scaler)
    experiments.append({
        'name': 'RFE only',
        'features': feature_pool['RFE'],
        'score': score['F1']
    })
    print(f'   Class 1 F1: {score["F1"]:.4f}\n')

    # Experiment 7: FFS only
    print(f'Experiment 7: FFS features only (n={len(feature_pool["FFS"])})')
    score = evaluate_feature_subset_cv(X_train, y_train, feature_pool['FFS'], scaler)
    experiments.append({
        'name': 'FFS only',
        'features': feature_pool['FFS'],
        'score': score['F1']
    })
    print(f'   Class 1 F1: {score["F1"]:.4f}\n')

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
        print(f'Experiment 9: RFE intersects FFS (n={len(rfe_ffs_intersection)})')
        score = evaluate_feature_subset_cv(X_train, y_train, rfe_ffs_intersection, scaler)
        experiments.append({
            'name': 'RFE intersects FFS',
            'features': rfe_ffs_intersection,
            'score': score['F1']
        })
        print(f'   Class 1 F1: {score["F1"]:.4f}\n')

    # Experiment 10: Union of top 20 from each method
    rfe_top20 = feature_pool['RFE'][:20]
    ffs_top20 = feature_pool['FFS'][:20]
    chi2_top20 = feature_pool['Chi2'][:20]
    union_top20 = list(set(rfe_top20) | set(ffs_top20) | set(chi2_top20))

    print(f'Experiment 10: Union of top 20 from each algorithm (n={len(chi2_top20)})')
    score = evaluate_feature_subset_cv(X_train, y_train, union_top20, scaler)
    experiments.append({
        'name': 'Union Top 20',
        'features': union_top20,
        'score': score['F1']
    })
    print(f'   Class 1 F1: {score["F1"]:.4f}\n')

    experiments.sort(key=lambda x: x['score'], reverse=True)

    print(DECORATOR)
    print("EXPERIMENT RESULTS SUMMARY")
    print(DECORATOR + "\n")

    for i, exp in enumerate(experiments, 1):
        print(f' {i}. {exp["name"]:.25s} | F1={exp["score"]:.4f} | n={len(exp["features"])}')

    print(f'\n\nBEST COMBINATION: {experiments[0]["features"]} (F1={experiments[0]["score"]:.4f})')

    return experiments[0]['features'], experiments[0]['score'], experiments



def train_final_model(X_train, y_train, features, scaler):
    X_selected = X_train[features]
    X_scaled = scaler.fit_transform(X_selected)
    X_smote, y_smote = SMOTE(random_state=42).fit_resample(X_scaled, y_train)

    model = LogisticRegression(
        solver='liblinear',
        max_iter=5000,
        random_state=42,
        class_weight={0: 1.0, 1: 3.0},
        penalty='l2'
    )
    model.fit(X_smote, y_smote)
    return model, scaler


def evaluate_final_model(model, scaler, X_test, y_test, features):
    X_selected = X_test[features]
    X_scaled = scaler.transform(X_selected)

    y_pred = model.predict(X_scaled)

    print(f'\nClassification Report:\n{classification_report(y_test, y_pred)}')
    print(f'\nMacro F1 (TRUE performance): {f1_score(y_test, y_pred, average="macro"):.4f}')

    return {
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "Macro_F1": f1_score(y_test, y_pred, average='macro'),
        "Accuracy": accuracy_score(y_test, y_pred)
    }


def main():
    df = pd.read_csv('data/Credit_Train.csv')
    X_train, X_test, y_train, y_test = preprocess_train_test(df, target_col='class')

    scaler = StandardScaler()

    print("\n" + DECORATOR)
    print(DECORATOR)
    print("PERFORMING FEATURE SELECTION & EXPERIMENT WITH CV EVALUATION")
    print(DECORATOR)
    print(DECORATOR + "\n")

    pool, all_features, sorted_pool = build_feature_pool_experiments(X_train, y_train, scaler, top_k=25)

    best_features, cv_score, all_experiments = experiment_with_combinations(
        X_train, y_train, pool, sorted_pool, scaler
    )

    print("\n" + DECORATOR)
    print(DECORATOR)
    print("RUNNING ITERATIONS OF EVALUATION OF SELECTED FEATURE COMBINATIONS ")
    print(DECORATOR)
    print(DECORATOR + "\n")

    com1 = ['duration', 'credit_amount', 'employment', 'installment_commitment', 'existing_credits', 'age_bin_25-35',
            'age_bin_35-50', 'age_bin_50-65', 'age_bin_65+', 'checking_status_<0', 'checking_status_no checking',
            'credit_history_critical/other existing credit', 'credit_history_delayed previously',
            'credit_history_existing paid', 'credit_history_no credits/all paid', 'purpose_education',
            'purpose_new car', 'purpose_used car', 'savings_status_>=1000', 'savings_status_no known savings',
            'personal_status_male single', 'property_magnitude_real estate', 'other_payment_plans_none', 'housing_own',
            'own_telephone_yes', 'foreign_worker_yes']
    com2 = ['checking_status_<0', 'duration', 'credit_amount', 'installment_commitment', 'age_bin_25-35',
            'age_bin_35-50', 'age_bin_50-65', 'age_bin_65+', 'checking_status_no checking',
            'credit_history_critical/other existing credit', 'credit_history_delayed previously',
            'credit_history_existing paid', 'purpose_education', 'purpose_new car', 'savings_status_<100',
            'housing_own', 'foreign_worker_yes']
    com3 = ['duration', 'credit_amount', 'employment', 'installment_commitment', 'age_bin_25-35', 'age_bin_35-50',
            'age_bin_50-65', 'checking_status_<0', 'checking_status_no checking',
            'credit_history_critical/other existing credit', 'credit_history_delayed previously',
            'credit_history_existing paid', 'purpose_education', 'purpose_furniture/equipment', 'purpose_new car',
            'purpose_radio/tv', 'purpose_used car', 'savings_status_no known savings', 'personal_status_male single',
            'housing_own', 'job_skilled', 'job_unemp/unskilled non res', 'job_unskilled resident', 'own_telephone_yes',
            'foreign_worker_yes']

    com4 = [ 'credit_amount', 'checking_status_<0', 'age_bin_35-50', 'checking_status_no checking',
             'property_magnitude_no known property', 'purpose_new car','savings_status_<100',
             'credit_history_critical/other existing credit', 'employment',
             'savings_status_no known savings', 'housing_rent', 'credit_history_no credits/all paid'
            ]

    features_list = [com1, com2, com3, com4]

    # ----------------------------------------------
    # STORE METRICS OVER 50 RUNS
    # ----------------------------------------------
    results = {f"com{i + 1}": {
        "Precision": [],
        "Recall": [],
        "F1": [],
        "Macro_F1": [],
        "Accuracy": []
    } for i in range(len(features_list))}

    for run in range(50):
        print(f"\n\n========== RUN {run + 1} / 50 ==========\n")

        X_train, X_test, y_train, y_test = preprocess_train_test(df, target_col="class")

        for i, feature in enumerate(features_list):
            key = f"com{i + 1}"

            print(f"\n---- Feature Set {i + 1} ----")

            model, scaler = train_final_model(X_train, y_train, feature, StandardScaler())
            metrics = evaluate_final_model(model, scaler, X_test, y_test, feature)

            # Store metrics
            for metric_name, metric_value in metrics.items():
                results[key][metric_name].append(metric_value)

    print("\n\n\n==================== FINAL MEAN METRICS ACROSS 50 RUNS ====================\n")

    for i, feature in enumerate(features_list):
        key = f"com{i + 1}"
        print(f"\n--- Feature Combination {i + 1} ---")
        for metric_name, values in results[key].items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"{metric_name:12s}: {mean_val:.4f} ± {std_val:.4f}")

    print("\n" + DECORATOR)
    print("TRAINING FINAL MODEL WITH BEST FEATURE COMBINATION ON FULL DATASET")
    print(DECORATOR + "\n")

    print(f'Using {len(com4)} features:\n{com4}\n')

    X, y, encoder, imputer, employment_map, mode_val = \
        preprocess_full_dataset(df, target_col='class')

    final_model, fitted_scaler = train_final_model(X, y, com4, scaler)
    full_columns = X.columns

    print(f'\n{DECORATOR}')
    print('Saving final model....')
    with open('model_training_resources.pkl', 'wb') as f:
        pickle.dump({
            'model': final_model,
            'scaler': fitted_scaler,
            'encoder': encoder,
            'imputer': imputer,
            'employment_map': employment_map,
            'mode_val': mode_val,
            'features': com4,
            'full_columns': full_columns
        }, f)
    print('Model Saved!')


if __name__ == '__main__':
    main()