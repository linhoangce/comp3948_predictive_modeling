import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.dtypes.common import is_numeric_dtype
from sklearn.feature_selection import RFE, f_regression, SelectKBest, chi2
from sklearn.impute import KNNImputer
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from imblearn.over_sampling import SMOTE

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

DECORATOR = '='*10
path = r'data/Credit_Train.csv'

df = pd.read_csv(path)
# print(df)
print(df.describe())
print(df.isna().sum())


# ==============================================================
# Impute Missing Value columns
# ==============================================================
def impute_nan_column(df, col_name, measure_type):
    mask = df[col_name].isna()

    if measure_type == 'mean':
        imputed_val = df[col_name].mean()
    elif measure_type == 'median':
        imputed_val = df[col_name].median()
    elif measure_type == 'mode':
        imputed_val = df[col_name].mode()[0]
    else:
        raise Exception('Invalid measure type')

    df.loc[mask, col_name] = imputed_val
    return df

na_cols = df.columns[df.isna().any()]
# print(na_cols)
for col in na_cols:
    if is_numeric_dtype(df[col]):
        df[col] = KNNImputer(n_neighbors=5).fit_transform(df[[col]])
    else:
        df = impute_nan_column(df, col, 'mode')

print(df.describe())

df_copy = df.copy()


 # ====================================================================================
 # Process ordinal features
 # ====================================================================================
def encode_ordinal_features(df):
    """
       Encode ordinal features with meaningful numeric values
       Higher values = better financial situation = lower risk
    """
    checking_status_map = {
        '<0': -1,
        'no checking': -2,
        '0<=X<200': 1,
        '>=200': 2
    }

    savings_status_map = {
        'no known savings': -2,
        '<100': 1,
        '100<=X<500': 2,
        '500<=X<1000': 3,
        '>=1000': 4
    }

    employment_map = {
        'unemployed': -1,
        '<1': 1,
        '1<=X<4': 2,
        '4<=X<7': 3,
        '>=7': 4
    }

    df['checking_status'] = df['checking_status'].map(checking_status_map)
    df['savings_status'] = df['savings_status'].map(savings_status_map)
    df['employment'] = df['employment'].map(employment_map)
    return df


# ========================================================================================
# Feature Interaction
# ========================================================================================
def create_interactions_with_ordinal_features(df):
    """Create interaction features using ordinal encodings"""
    df = df.copy()

    df = encode_ordinal_features(df)

    # === HIGH-VALUE Interaction ===

    # 1. Monthly payment - most important
    df['monthly_payment'] = df['credit_amount'] / (df['duration'] + 1e-10)

    # 2. Financial health composite score
    df['financial_health'] = (
        df['checking_status'] + df['savings_status'] + df['employment']
    )

    # 3. Payment burden adjusted by financial health
    df['adjusted_payment_burden'] = df['credit_amount'] / (df['financial_health'] + df['financial_health'].max())

    # 4. Credit amount risk (amount adjusted by checking status)
    df['credit_risk_score'] = (
        df['credit_amount'] / (df['checking_status'])
    )

    # === Employment Stability Interactions ====

    # 6. Career stability
    df['career_stability'] = df['age'] * df['employment']

    # ====== RISK Flags ======
    df['high_risk_flag'] = (
        (df['credit_amount'] > df['credit_amount'].median()) &
        (df['financial_health'] <= 2)
    ).astype(int)

    # 9. Poor saving + long duration
    df['low_savings_flag'] = (
        (df['savings_status'] <= 1) &
        (df['duration'] > 24)
    ).astype(int)

    # 10. Unstable employment: short employment + high payment burden
    df['unstable_employment_flag'] = (
        (df['employment'] <= 1) &
        (df['installment_commitment'] >= 3)
    )

    return df


# df_copy = create_interactions_with_ordinal_features(df_copy)
df_copy = encode_ordinal_features(df_copy)


# ==================================================================
# Bin age column
# ==================================================================
# print(f'\n** Age **\n {df_copy['age'].value_counts()}\n')
# print(f'\nMin age: {df["age"].min()}')
bins = [df_copy['age'].min(), 25, 35, 50, 65, df_copy['age'].max()]
# print(f'\nAge Bins: {bins}')
df_age_bin = pd.cut(df_copy['age'], bins=bins,
                    labels=['young_adults', 'adults', 'middle-aged', 'senior', 'retired']
                    )
# print(f'\nAge Bins\n{df_age_bin}\n')

df_age_dummy = pd.get_dummies(df_age_bin, drop_first=True, dtype=int)
# print(f'\nDummy Age bins\n{df_age_dummy}')

# drop age colum
df_copy = df_copy.drop(['age'], axis=1)


# =============================================================================
# Create dummy variables for Categorical Data Columns
# =============================================================================
### extract categorical columns first
df_categorical = df_copy.select_dtypes(include=['object'])
# print(df_categorical)

df_categorical = pd.get_dummies(df_categorical, drop_first=True,
                                dtype=int)


### Extract numeric columns before concatenate both into final dataframe
df_numeric = df_copy.select_dtypes(include=[np.number])

# combine both numeric and categorical dataframe
df_new = pd.concat([df_numeric, df_age_dummy, df_categorical], axis=1)
df_new.to_csv('data/df_new.csv', index=False)
print('Saved df_new.csv to folder data')

# ===========================================================================
# FEATURE SELECTIONS
# ==========================================================================
def select_features_rfe(X, y, n_features, scaler_X):
    X_scaled = scaler_X.fit_transform(X)
    rfe = RFE(LogisticRegression(), n_features_to_select=n_features)
    rfe.fit(X_scaled, y)
    selected_features = X.keys()[rfe.support_ == True]
    return selected_features

def select_features_ffs(X, y, n_features, scaler_X):
    X_scaled = scaler_X.fit_transform(X)
    ffs = f_regression(X_scaled, y)
    df_ffs = pd.DataFrame({
        'feature': X.columns,
        'f_stat': ffs[0]
    })
    df_ffs.sort_values(by=['f_stat'], ascending=False, inplace=True)
    top_features = df_ffs['feature'][:n_features]
    return list(top_features)

def select_features_chi2(X, y, n_features, scaler_X):
    X_scaled = scaler_X.fit_transform(X)

    # Shift for negative values
    if isinstance(scaler_X, (StandardScaler, RobustScaler)):
        X_scaled = X_scaled - X_scaled.min() + 1e-6

    selector = SelectKBest(score_func=chi2, k=n_features)
    selector.fit(X_scaled, y)
    chi_scores = dict(zip(X.columns, selector.scores_))
    significant_features = {col:score.item() for col,score in chi_scores.items()
                            if score >= 3.8}
    # print(significant_features)
    return list(significant_features.keys())

def evaluate_feature_selection(selected_features, X, y, scaler_X):
    X = X[selected_features]
    X_scaled = scaler_X.fit_transform(X)

    k_fold = KFold(n_splits=5, shuffle=True)
    precision = []
    recall = []
    f1 = []
    acc = []
    for train_idx, test_idx in k_fold.split(X_scaled):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        model = LogisticRegression(fit_intercept=True, solver='liblinear', max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        precision.append(precision_score(y_test, y_pred))
        recall.append(recall_score(y_test, y_pred))
        f1.append(f1_score(y_test, y_pred))
        acc.append(accuracy_score(y_test, y_pred))

    return {'Precision': np.mean(precision),
            'Recall': np.mean(recall),
            'F1': np.mean(f1),
            'Accuracy': np.mean(acc)}

def plot_feature_selection_result(precision, recall, f1, accuracy, selector, scaler):
    best_idx = np.argmax(f1)
    x = range(2, len(f1) + 2)
    plt.plot(x, precision, '-', alpha=0.6, label=f'Precision: {np.mean(accuracy):.4f}')
    plt.plot(x, recall, '-', color='green', label=f'Recall: {np.mean(recall):.4f}')
    plt.plot(x, f1, 'r-^', label=f'F1-score: {np.mean(f1)}')
    plt.plot(x, accuracy, '--', color='blue', label=f'Accuracy: {np.mean(accuracy):.4f}')
    plt.axhline(f1[best_idx], color='g', linestyle='--', label=f'Best F1={f1[best_idx]:.4f}')
    plt.xlabel('Number of features')
    plt.title(f'{selector} - {scaler}')
    plt.legend()
    plt.grid(alpha=0.3)


def perform_feature_selection(X, y, scaler, select_features_algo, n_features_range):
    precision = []
    recall = []
    f1 = []
    accuracy = []
    selected_features = []

    for i in range(1, n_features_range):
        features = select_features_algo(X, y, i, scaler)
        results = evaluate_feature_selection(features, X, y, scaler)
        precision.append(results['Precision'])
        recall.append(results['Recall'])
        f1.append(results['F1'])
        accuracy.append(results['Accuracy'])
        selected_features.append(features)

    return precision, recall, f1, accuracy, selected_features


DECORATOR = '='*10
df_new = pd.read_csv('data/df_new.csv')
X = df_new.copy()
X = X.drop('class', axis=1)
y = df_new['class']

mm_scaler = MinMaxScaler()
std_scaler = StandardScaler()
rb_scaler = RobustScaler()

def plot_feature_selection_with_scaler(X, y, scaler, scaler_name):
    precision_rfe, recall_rfe, f1_rfe, accuracy_rfe, selected_features_rfe = \
        perform_feature_selection(X, y, scaler, select_features_rfe, len(X.columns))

    precision_ffs, recall_ffs, f1_ffs, accuracy_ffs, selected_features_ffs = \
        perform_feature_selection(X, y, scaler, select_features_ffs, len(X.columns))

    precision_chi2, recall_chi2, f1_chi2, accuracy_chi2, selected_features_chi2 = \
        perform_feature_selection(X, y, scaler, select_features_chi2, len(X.columns))

    plt.figure(figsize=(12, 12))
    plt.subplot(3, 1, 1)
    plot_feature_selection_result(precision_rfe, recall_rfe, f1_rfe, accuracy_rfe, 'RFE', scaler_name)
    plt.subplot(3, 1, 2)
    plot_feature_selection_result(precision_ffs, recall_ffs, f1_ffs, accuracy_ffs, 'FFS',scaler_name)
    plt.subplot(3, 1, 3)
    plot_feature_selection_result(precision_chi2, recall_chi2, f1_chi2, accuracy_chi2, 'Chi2',scaler_name)
    plt.tight_layout()
    plt.show()

    best_f1_idx_rfe = np.argmax(f1_rfe)
    best_f1_idx_ffs = np.argmax(f1_ffs)
    best_f1_idx_chi2 = np.argmax(f1_chi2)
    return  selected_features_rfe[best_f1_idx_rfe], selected_features_ffs[best_f1_idx_ffs], \
            selected_features_chi2[best_f1_idx_chi2]



# =======================================================================================================
# TRAINING WITH SELECTED FEATURES RETURNED ABOVE
# =======================================================================================================

def train(X, y, features_dict, scalers_dict, random_state=42):

    k_fold = StratifiedKFold(n_splits=10, shuffle=True)
    splits = list(k_fold.split(X, y))
    results = {}

    for name, features in features_dict.items():
        scaler = scalers_dict[name]

        X_selected = X[features]
        X_scaled = scaler.fit_transform(X_selected)

        precisions = []
        recalls = []
        f1s = []
        accuracies = []
        models = []

        for train_idx, test_idx in splits:
            X_train_fold = X_scaled[train_idx]
            X_test_fold = X_scaled[test_idx]
            y_train_fold = y.iloc[train_idx]
            y_test_fold = y.iloc[test_idx]

            X_train_SMOTE, y_train_SMOTE = SMOTE(random_state=random_state).fit_resample(
                X_train_fold, y_train_fold
            )

            model = LogisticRegression(
                fit_intercept=True,
                solver='liblinear',
                max_iter=1000,
            )
            model.fit(X_train_SMOTE, y_train_SMOTE)
            y_pred = model.predict(X_test_fold)

            precisions.append(precision_score(y_test_fold, y_pred))
            recalls.append(recall_score(y_test_fold, y_pred))
            f1s.append(f1_score(y_test_fold, y_pred))
            accuracies.append(accuracy_score(y_test_fold, y_pred))

        results[name] = {
            'Precision': np.mean(precisions),
            'Recall': np.mean(recalls),
            'F1': np.mean(f1s),
            'Accuracy': np.mean(accuracies),
            'features': features,  # Store the features list
            'scaler_type': type(scaler)
        }

        print(f'\n{DECORATOR} {name} {DECORATOR}')
        print(f'CV Precision: {results[name]["Precision"]:.4f}')
        print(f'CV Recall: {results[name]["Recall"]:.4f}')
        print(f'CV F1: {results[name]["F1"]:.4f}')
        print(f'CV Accuracy: {results[name]["Accuracy"]:.4f}')

    # find best approach based on F1 score
    best_name = max(results.items(), key=lambda x: x[1]['F1'])[0]
    best_result = results[best_name]

    print(f'\n{"="*100}')
    print(f'BEST APPROACH: {best_name}')
    print(f'Best F1: {best_result["F1"]:.4f}')
    print(f'\n{"="*100}')

    return results, best_name, best_result


def train_final_model(X, y, features, scaler_class, random_state=42):
    """
    Train final model on ALL training data with best approach
    """
    print(f'\n{DECORATOR} TRAINING FINAL MODEL {DECORATOR}')

    X_selected = X[features]

    # Fit scaler on all training data
    scaler = scaler_class()
    X_scaled = scaler.fit_transform(X_selected)

    X_train_SMOTE, y_train_SMOTE = SMOTE(random_state=random_state).fit_resample(
        X_scaled, y
    )

    final_model = LogisticRegression(
        fit_intercept=True,
        solver='liblinear',
        max_iter=1000
    )

    final_model.fit(X_train_SMOTE, y_train_SMOTE)

    print(f'Model trained on {len(X_train_SMOTE)} samples (after SMOTE)')
    print(f'Using {len(features)} features: {list(features)}')

    return final_model, scaler

def evaluate_final_model(model, scaler, X_test, y_test, features):
    """
    Evaluate final model on test set
    """
    print(f'\n{DECORATOR} EVALUATING ON TEST SET {DECORATOR}')

    X_test_selected = X_test[features]
    X_test_scaled = scaler.transform(X_test_selected)

    y_pred = model.predict(X_test_scaled)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    print(f'Test Precision: {precision:.4f}')
    print(f'Test Recall: {recall:.4f}')
    print(f'Test F1: {f1:.4f}')
    print(f'Test Accuracy: {accuracy:.4f}')

    return {
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'Accuracy': accuracy
    }


X_train, X_test_final, y_train, y_test_final = train_test_split(
    X, y, test_size=0.2, shuffle=True, stratify=y
)

# feature selection on training data
best_rfe_mm, best_ffs_mm, best_chi2_mm = \
    plot_feature_selection_with_scaler(X_train, y_train, MinMaxScaler(), 'MM Scaler')
best_rfe_std, best_ffs_std, best_chi2_std = \
    plot_feature_selection_with_scaler(X_train, y_train, StandardScaler(), 'STD Scaler')
best_rfe_rb, best_ffs_rb, best_chi2_rb = \
    plot_feature_selection_with_scaler(X_train, y_train, RobustScaler(), 'RB Scaler')

features_dict = {
    'MM_RFE': best_rfe_mm,
    'MM_FFS': best_ffs_mm,
    'MM_CHI2': best_chi2_mm,
    'STD_RFE': best_rfe_std,
    'STD_FFS': best_ffs_std,
    'STD_CHI2': best_chi2_std,
    'RB_RFE': best_rfe_rb,
    'RB_FFS': best_ffs_rb,
    'RB_CHI2': best_chi2_rb,
}

scalers_dict = {
    'MM_RFE': MinMaxScaler(),
    'MM_FFS': MinMaxScaler(),
    'MM_CHI2': MinMaxScaler(),
    'STD_RFE': StandardScaler(),
    'STD_FFS': StandardScaler(),
    'STD_CHI2': StandardScaler(),
    'RB_RFE': RobustScaler(),
    'RB_FFS': RobustScaler(),
    'RB_CHI2': RobustScaler(),
}

cv_results, best_name, best_result = train(X_train, y_train, features_dict, scalers_dict)


final_model, scaler = train_final_model(
    X_train, y_train,
    best_result['features'],
    best_result['scaler_type']
)

test_results = evaluate_final_model(
    final_model, scaler, X_test_final, y_test_final, best_result['features']
)


# with open('final_model_4.pkl', 'wb') as f:
#     pickle.dump({
#         'model': final_model,
#         'scaler': scaler,
#         'features': best_result['features'],
#         'approach': best_name
#     }, f)
#
# print(f'\n{DECORATOR} FINAL MODEL SAVED {DECORATOR}')
# print(f'Approach: {best_name}')
# print(f'Features: {list(best_result["features"])}')