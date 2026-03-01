# ===========================================================================
# FEATURE SELECTIONS
# ==========================================================================
import numpy as np
import pandas as pd
import pickle
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from sklearn.feature_selection import RFE, f_regression, SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler


def select_features_rfe(X, y, n_features):
    rfe = RFE(LogisticRegression(solver='liblinear', max_iter=1000),
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

def evaluate_feature_selection(X, y, scaler, feature_selector, n_features):

    k_fold = StratifiedKFold(n_splits=10, shuffle=True)
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

        selected_features = feature_selector(X, y, n_features)

        X_train_fs = X_train_scaled[selected_features]
        X_test_fs = X_test_scaled[selected_features]

        X_train_sm, y_train_sm = SMOTE().fit_resample(X_train_fs, y_train)

        model = LogisticRegression(fit_intercept=True, solver='liblinear', max_iter=1000)
        model.fit(X_train_sm, y_train_sm)
        y_pred = model.predict(X_test_fs)

        precision.append(precision_score(y_test, y_pred))
        recall.append(recall_score(y_test, y_pred))
        f1.append(f1_score(y_test, y_pred))
        acc.append(accuracy_score(y_test, y_pred))

    return {'Precision': np.mean(precision),
            'Recall': np.mean(recall),
            'F1': np.mean(f1),
            'Accuracy': np.mean(acc)}

def perform_feature_selection(X, y, scaler, select_features_algo, n_features_range):
    precision = []
    recall = []
    f1 = []
    accuracy = []

    for k in range(1, n_features_range+1):
        results = evaluate_feature_selection(X, y, scaler, select_features_algo, k)
        precision.append(results['Precision'])
        recall.append(results['Recall'])
        f1.append(results['F1'])
        accuracy.append(results['Accuracy'])

    best_k = np.argmax(f1) + 1
    return precision, recall, f1, accuracy, best_k

def plot_feature_selection(X, y, scaler, scaler_name):
    """
    Evaluate RFE, FFS, Chi2 across feature counts and plot all.
    Returns best_k for each selector.
    :param X:
    :param y:
    :param scaler:
    :param scaler_name:
    :return:
    """
    selectors = {
        'RFE': select_features_rfe,
        'FFS': select_features_ffs,
        'Chi2': select_features_chi2
    }

    best_ks = {}

    plt.figure(figsize=(12, 12))

    for i, (name, selector) in enumerate(selectors.items(), start=1):
        precision, recall, f1, accuracy, best_k = \
            perform_feature_selection(X, y, scaler, selector, len(X.columns))

        best_ks[name] = best_k

        k_vals = np.arange(1, len(f1) + 1)

        plt.subplot(3, 1, i)
        plt.plot(k_vals, precision, label='Precision', marker='.')
        plt.plot(k_vals, recall, label='Recall', marker='.')
        plt.plot(k_vals, f1, label='F1', marker='^')
        plt.plot(k_vals, accuracy, label='Accuracy', linestyle='--')

        plt.axvline(best_k, color='red', linestyle='--',
                    label=f'Best k={best_k} (F1={f1[best_k-1]:.4f}')

        plt.title(f'{name} - {scaler_name}')
        plt.xlabel('Number of Features (k)')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    return best_ks


def get_final_feature_set(X, y, scaler, selector, best_k):
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    final_features = selector(X_scaled, y, best_k)
    return final_features, scaler

# =======================================================================================================
# TRAINING WITH SELECTED FEATURES RETURNED ABOVE
# =======================================================================================================
def train_final_model(X_train, y_train, features, scaler):
    X_selected = X_train[features]
    X_scaled = scaler.fit_transform(X_selected)

    X_SMOTE, y_SMOTE = SMOTE().fit_resample(X_scaled, y_train)

    model = LogisticRegression(solver='liblinear', max_iter=1000)
    model.fit(X_SMOTE, y_SMOTE)
    return model

def evaluate_final_model(model, scaler, X_test, y_test, features):
    X_selected = X_test[features]
    X_scaled = scaler.transform(X_selected)

    y_pred = model.predict(X_scaled)

    return {
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "Accuracy": accuracy_score(y_test, y_pred)
    }


# def train(X, y, features_dict, scalers_dict):
#
#     k_fold = StratifiedKFold(n_splits=10, shuffle=True)
#     splits = list(k_fold.split(X, y))
#     results = {}
#
#     for name, features in features_dict.items():
#         scaler = scalers_dict[name]
#
#         X_selected = X[features]
#         X_scaled = scaler.fit_transform(X_selected)
#
#         precisions = []
#         recalls = []
#         f1s = []
#         accuracies = []
#
#         for train_idx, test_idx in splits:
#             X_train_fold = X_scaled[train_idx]
#             X_test_fold = X_scaled[test_idx]
#             y_train_fold = y.iloc[train_idx]
#             y_test_fold = y.iloc[test_idx]
#
#             X_train_SMOTE, y_train_SMOTE = SMOTE(random_state=random_state).fit_resample(
#                 X_train_fold, y_train_fold
#             )
#
#             model = LogisticRegression(
#                 fit_intercept=True,
#                 solver='liblinear',
#                 max_iter=1000,
#             )
#             model.fit(X_train_SMOTE, y_train_SMOTE)
#             y_pred = model.predict(X_test_fold)
#
#             precisions.append(precision_score(y_test_fold, y_pred))
#             recalls.append(recall_score(y_test_fold, y_pred))
#             f1s.append(f1_score(y_test_fold, y_pred))
#             accuracies.append(accuracy_score(y_test_fold, y_pred))
#
#
#         results[name] = {
#             'Precision': np.mean(precisions),
#             'Recall': np.mean(recalls),
#             'F1': np.mean(f1s),
#             'Accuracy': np.mean(accuracies),
#             'features': features,  # Store the features list
#             'scaler_type': type(scaler),
#         }
#
#         print(f'CV Precision: {results[name]["Precision"]:.4f}')
#         print(f'CV Recall: {results[name]["Recall"]:.4f}')
#         print(f'CV F1: {results[name]["F1"]:.4f}')
#         print(f'CV Accuracy: {results[name]["Accuracy"]:.4f}')
#
#     # find best approach based on F1 score
#     best_name = max(results.items(), key=lambda x: x[1]['F1'])[0]
#     best_result = results[best_name]
#
#     print(f'\n{"="*100}')
#     print(f'BEST APPROACH: {best_name}')
#     print(f'Best F1: {best_result["F1"]:.4f}')
#     print(f'\n{"="*100}')
#
#     return results, best_name, best_result

#
# def train_final_model(X, y, features, scaler_class, random_state=42):
#     """
#     Train final model on ALL training data with best approach
#     """
#     print(f'\n{DECORATOR} TRAINING FINAL MODEL {DECORATOR}')
#
#     X_selected = X[features]
#
#     # Fit scaler on all training data
#     scaler = scaler_class()
#     X_scaled = scaler.fit_transform(X_selected)
#
#     X_train_SMOTE, y_train_SMOTE = SMOTE(random_state=random_state).fit_resample(
#         X_scaled, y
#     )
#
#     final_model = LogisticRegression(
#         fit_intercept=True,
#         solver='liblinear',
#         max_iter=1000
#     )
#
#     final_model.fit(X_train_SMOTE, y_train_SMOTE)
#
#     print(f'Model trained on {len(X_train_SMOTE)} samples (after SMOTE)')
#     print(f'Using {len(features)} features: {list(features)}')
#
#     return final_model, scaler
#
# def evaluate_final_model(model, scaler, X_test, y_test, features):
#     """
#     Evaluate final model on test set
#     """
#     print(f'\n{DECORATOR} EVALUATING ON TEST SET {DECORATOR}')
#
#     X_test_selected = X_test[features]
#     X_test_scaled = scaler.transform(X_test_selected)
#
#     y_pred = model.predict(X_test_scaled)
#
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)
#     accuracy = accuracy_score(y_test, y_pred)
#
#
#     print(f'Test Precision: {precision:.4f}')
#     print(f'Test Recall: {recall:.4f}')
#     print(f'Test F1: {f1:.4f}')
#     print(f'Test Accuracy: {accuracy:.4f}')
#
#     return {
#         'Precision': precision,
#         'Recall': recall,
#         'F1': f1,
#         'Accuracy': accuracy
#     }

def main():

    DECORATOR = '='*200
    df_new = pd.read_csv('data/df_new.csv')
    X = df_new.copy()
    X = X.drop('class', axis=1)
    y = df_new['class']

    mm_scaler = MinMaxScaler()
    std_scaler = StandardScaler()
    rb_scaler = RobustScaler()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, stratify=y,
    )

    best_k_mm = plot_feature_selection(X_train, y_train, MinMaxScaler(), 'MinMax')
    best_k_std = plot_feature_selection(X_train, y_train, StandardScaler(), "Standard")
    best_k_rb = plot_feature_selection(X_train, y_train, RobustScaler(), "Robust")

    final_results = {}

    for scaler_name, scaler in [
        ('MM', MinMaxScaler()),
        ('STD', StandardScaler()),
        ('RB', RobustScaler())
    ]:
        print(DECORATOR)
        print(scaler_name)
        print(DECORATOR)
        for selector_name, selector_func in [
            ('RFE', select_features_rfe),
            ('FFS', select_features_ffs),
            ('Chi2', select_features_chi2)
        ]:
            best_k = locals()[f'best_k_{scaler_name.lower()}'][selector_name]

            features, fitted_scaler = get_final_feature_set(
                X_train, y_train, scaler, selector_func, best_k
            )

            model = train_final_model(X_train, y_train, features, fitted_scaler)
            test_metrics = evaluate_final_model(model, fitted_scaler, X_test, y_test, features)

            print(f'\n{selector_name}\n')
            print(f'K: {best_k}\n'
                  f'Features: {features}\n'
                  f'Metrics: {test_metrics}\n')
            print(DECORATOR + '\n\n')

            final_results[f'{scaler_name}'] = {
                'k': best_k,
                'features': features,
                'metrics': test_metrics
            }


if __name__ == "__main__":
    main()
