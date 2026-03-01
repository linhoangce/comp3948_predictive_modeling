
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.feature_selection import RFE, SelectKBest, f_regression, r_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, KFold
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold
import statsmodels.api as sm

def select_features_rfe(X, y, n_features, scaler_X, scaler_y):
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(np.array(y).reshape(-1, 1))

    rfe = RFE(LinearRegression(), n_features_to_select=n_features)
    rfe.fit(X_scaled, y_scaled)
    selected_features = X.keys()[rfe.support_ == True]
    # print(selected_features.values)
    return selected_features

def select_features_ffs(X, y, n_features, scaler_X):
    X_scaled = scaler_X.fit_transform(X)

    ffs = f_regression(X_scaled, y)
    df_ffs = pd.DataFrame({'feature': X.columns,
                          'f_stat': ffs[0]})
    df_ffs.sort_values(['f_stat'], ascending=False, inplace=True)
    top_features = df_ffs['feature'][:n_features]
    return top_features

def build_evaluate_predictor(features, X, y, scaler_X, scaler_y):
    X = X[features]
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(np.array(y).reshape(-1, 1))

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.25)

    X_train = sm.add_constant(X_train, has_constant='add')
    X_test = sm.add_constant(X_test, has_constant='add')

    model = sm.OLS(y_train, X_train).fit()
    y_pred = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(np.array(y_pred).reshape(-1, 1))
    y_test = scaler_y.inverse_transform(np.array(y_test).reshape(-1, 1))

    # print(model.summary())
    print(f"RMSE: {np.sqrt(metrics.mean_squared_error(y_test, y_pred))}")
    print(f"R-squared: {r2_score(y_test, y_pred)}\n\n")
    return {'rmse': np.sqrt(metrics.mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'model': model.params}


# ========================================================================================================

# df_new = pd.read_csv('data/df_new.csv')
#
# # Remove irrelevant features identified by several rounds of running
# # feature selection algorithms to improve running time
# df_new = df_new.drop(columns=df_new.loc[:, '24_hour_check_in':'ev_charger'].columns)
# df_new = df_new.drop(columns=df_new.loc[:, 'elevator_in_building':'stove'].columns)
# df_new = df_new.drop(columns=df_new.loc[:, 'tv':'translation_missing:_en_hosting_amenity_49'].columns)
# mm_scaler_X = MinMaxScaler()
# mm_scaler_y = MinMaxScaler()
# std_scaler_X = StandardScaler()
# std_scaler_y = StandardScaler()
# rb_scaler_X = RobustScaler()
# rb_scaler_y = RobustScaler()
#
# X = df_new.copy()
# del X['price']
# y = df_new['price']
#
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=10000, shuffle=True
# )


def plot_feature_selection_result(rmse_bfs, features_rfe, bfs_params,
                                  rmse_ffs, features_ffs, ffs_params, scaler_name):
    # get min rmse index
    idx = np.argmin(rmse_bfs)

    # Plot results
    plt.figure(figsize=(24, 12))

    plt.subplot(2, 2, 1)
    plt.plot(rmse_bfs, '-', alpha=0.6)
    plt.axhline(np.mean(rmse_bfs), color='r', linestyle='--', label=f'Mean: {np.mean(rmse_bfs):.2f}')
    plt.axhline(rmse_bfs[idx], color='g', linestyle='--', label=f'Best: {rmse_bfs[idx]:.2f}')
    plt.xlabel('Number of Features')
    plt.ylabel('RMSE')
    plt.title('RMSE - BFS')
    plt.legend()
    plt.grid(alpha=0.3)

    idx = np.argmin(rmse_ffs)

    plt.subplot(2, 2, 2)
    plt.plot(rmse_ffs, '-', alpha=0.6)
    plt.axhline(np.mean(rmse_ffs), color='r', linestyle='--', label=f'Mean: {np.mean(rmse_ffs):.3f}')
    plt.axhline(rmse_ffs[idx], color='g', linestyle='--', label=f'Best: {rmse_ffs[idx]:.3f}')
    plt.xlabel('Number of Features')
    plt.ylabel('RMSE')
    plt.title('RMSE - FFS')
    plt.legend()
    plt.grid(alpha=0.3)

    coef_importance = abs(np.sort(bfs_params[idx]))

    plt.subplot(2, 2, 3)
    plt.barh(range(len(coef_importance)), coef_importance)
    plt.yticks(range(len(coef_importance)), features_rfe[:len(coef_importance)][:len(coef_importance)], fontsize=8)
    plt.xlabel('Absolute Coefficient Value')
    plt.title(f'Top {len(coef_importance)} Feature Importance - BFS')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    coef_importance = abs(np.sort(ffs_params[idx]))

    plt.subplot(2, 2, 4)
    plt.barh(range(len(coef_importance)), coef_importance)
    plt.yticks(range(len(coef_importance)), features_rfe[:len(coef_importance)][:len(coef_importance)], fontsize=8)
    plt.xlabel('Absolute Coefficient Value')
    plt.title(f'Top {len(coef_importance)}  Feature Importance - FFS')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    plt.tight_layout()
    plt.show()


# ======================================================================================
# Feature Selection - Scaling
# ======================================================================================
# rmse_bfs_mm = []
# r2_bfs_mm = []
# best_bfs_mm = []
# rmse_ffs_mm = []
# r2_ffs_mm = []
# best_ffs_mm = []
# ffs_params = []
# bfs_params = []
# print("\n*************** MM Scaler *************")
# for i in range(6, 30):
#         features_rfe = select_features_rfe(X, y, i, mm_scaler_X, std_scaler_y)
#         result = build_evaluate_predictor(features_rfe, X, y, mm_scaler_X, std_scaler_y)
#         rmse_bfs_mm.append(result['rmse'])
#         r2_bfs_mm.append(result['r2'])
#         best_bfs_mm.append(features_rfe)
#         bfs_params.append(result['model'])
#
#         features_ffs = select_features_ffs(X, y, i, mm_scaler_X)
#         result = build_evaluate_predictor(features_ffs, X, y, mm_scaler_X, std_scaler_y)
#         rmse_ffs_mm.append(result['rmse'])
#         r2_ffs_mm.append(result['r2'])
#         best_ffs_mm.append(features_rfe)
#         ffs_params.append(result['model'])
#
#
# plot_feature_selection_result(rmse_bfs_mm, features_rfe, bfs_params,
#                               rmse_ffs_mm, features_ffs, ffs_params, "MinMax Scaling")
#
# # ================================================================================================================
#
# print("\n*************** STD Scaler *************")
# rmse_bfs_std = []
# r2_bfs_std = []
# best_bfs_std = []
# rmse_ffs_std = []
# r2_ffs_std = []
# best_ffs_std = []
# ffs_params = []
# bfs_params = []
#
# for i in range(6, 30):
#         features_rfe = select_features_rfe(X, y, i, std_scaler_X, std_scaler_y)
#         result = build_evaluate_predictor(features_rfe, X, y, std_scaler_X, std_scaler_y)
#         rmse_bfs_std.append(result['rmse'])
#         r2_bfs_std.append(result['r2'])
#         best_bfs_std.append(features_rfe)
#         bfs_params.append(result['model'])
#
#         features_ffs = select_features_ffs(X, y, i, std_scaler_X)
#         result = build_evaluate_predictor(features_ffs, X, y, std_scaler_X, std_scaler_y)
#         rmse_ffs_std.append(result['rmse'])
#         r2_ffs_std.append(result['r2'])
#         best_ffs_std.append(features_rfe)
#         ffs_params.append(result['model'])
#
# plot_feature_selection_result(rmse_bfs_std, features_rfe, bfs_params,
#                               rmse_ffs_std, features_ffs, ffs_params, "MinMax Scaling")
#
# # =============================================================================================================
#
# print("\n*************** Robust Scaler *************")
# rmse_bfs_rb = []
# r2_bfs_rb = []
# best_bfs_rb = []
# rmse_ffs_rb = []
# r2_ffs_rb = []
# best_ffs_rb = []
# ffs_params = []
# bfs_params = []
# for i in range(6, 30):
#         features_rfe = select_features_rfe(X, y, i, rb_scaler_X, rb_scaler_X)
#         result = build_evaluate_predictor(features_rfe, X, y, rb_scaler_X, rb_scaler_X)
#         rmse_bfs_rb.append(result['rmse'])
#         r2_bfs_rb.append(result['r2'])
#         best_bfs_rb.append(features_rfe)
#         bfs_params.append(result['model'])
#
#         features_ffs = select_features_ffs(X, y, i, rb_scaler_X)
#         result = build_evaluate_predictor(features_ffs, X, y, rb_scaler_X, rb_scaler_X)
#         rmse_ffs_rb.append(result['rmse'])
#         r2_ffs_rb.append(result['r2'])
#         best_ffs_rb.append(features_rfe)
#         ffs_params.append(result['model'])
#
#
# plot_feature_selection_result(rmse_bfs_rb, features_rfe, bfs_params,
#                               rmse_ffs_rb, features_ffs, ffs_params, "MinMax Scaling")