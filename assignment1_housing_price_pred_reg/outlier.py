# ===================================================================================
# Outliers
# ===================================================================================
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import zscore
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import statsmodels.api as sm

from feature_sel import build_evaluate_predictor, select_features_rfe, select_features_ffs, \
    plot_feature_selection_result


def get_outliers_by_zscore(df, col_name, threshold, plt):
    df_sub = df[[col_name]]
    print(df_sub.describe())
    df_sub.boxplot(column=[col_name])
    plt.title(col_name)
    plt.show()

    z = np.abs(zscore(df_sub))
    row_col_array = np.where(z > threshold)
    row_indices = row_col_array[0]
    return row_indices

df_new = pd.read_csv('data/df_new.csv')
# row_indices = get_outliers_by_zscore(df_new, 'price', 3, plt)
# df_outliers = df_new.iloc[row_indices]
# df_outliers.sort_values(['price'], ascending=False, inplace=True)
# print("Outlier DataFrame:")
# print(df_outliers.head(40))

df_new = df_new.drop(columns=df_new.loc[:, '24_hour_check_in':'ev_charger'].columns)
df_new = df_new.drop(columns=df_new.loc[:, 'elevator_in_building':'stove'].columns)
df_new = df_new.drop(columns=df_new.loc[:, 'tv':'translation_missing:_en_hosting_amenity_49'].columns)


df_clip = df_new.copy()
df_c = df_new['price']
df_clip['price'] = df_c


X = df_clip.copy()
del X['price']
y = df_clip['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=10000, shuffle=True
)

y_train = y_train.clip(None, 1500)

mm_scaler_X = MinMaxScaler()
mm_scaler_y = MinMaxScaler()
std_scaler_X = StandardScaler()
std_scaler_y = StandardScaler()
rb_scaler_X = RobustScaler()
rb_scaler_y = RobustScaler()



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
#     k_fold = KFold(n_splits=5, shuffle=True, random_state=0)
#     cv_rmse_list = []
#
#     for train_idx, val_idx in k_fold.split(X_train):
#         X_tr = X_train.iloc[train_idx]
#         y_tr = y_train.iloc[train_idx]
#         X_val = X_train.iloc[val_idx]
#         y_val = y_train.iloc[val_idx]
#
#         # <-- feature selection happens ONLY on X_tr / y_tr
#         features_rfe = select_features_rfe(X_tr, y_tr, n_features=i, scaler_X=std_scaler_X, scaler_y=std_scaler_y)
#
#         # transform X_tr and X_val using ONLY those features
#         X_tr2 = X_tr[features_rfe]
#         X_val2 = X_val[features_rfe]
#
#         X_tr2 = std_scaler_X.fit_transform(X_tr2)
#         X_val2 = std_scaler_X.transform(X_val2)
#         y_tr2 = std_scaler_y.fit_transform(y_tr.values.reshape(-1, 1))
#
#         X_tr2 = sm.add_constant(X_tr2, has_constant='add')
#         X_val2 = sm.add_constant(X_val2, has_constant='add')
#
#         # train model on fold
#         model = sm.OLS(y_tr2, X_tr2).fit()
#         y_pred = model.predict(X_val2)
#         y_pred = std_scaler_y.inverse_transform(y_pred.reshape(-1, 1))
#
#         rmse = np.sqrt(mean_squared_error(y_val, y_pred))
#         cv_rmse_list.append(rmse)
#         bfs_params.append(model.params)
#         best_bfs_std.append(features_rfe)
#
#     # average CV rmse for this k
#     rmse_k = np.mean(cv_rmse_list)
#     rmse_bfs_std.append(rmse_k)
#     print(f"k={i} average rmse = {rmse_k}")
#
# idx = np.argmin(rmse_bfs_std)

best_features = ['accommodates', 'bathrooms', 'cleaning_fee', 'review_scores_rating',
       'bedrooms', 'beds', 'elevator', 'neighbourhood_avg_price',
       'room_type_Private room', 'room_type_Shared room']

rmses_test = []
r2s_test = []

rmses_train = []
r2s_train = []
r2s_adj_train = []
aics_train = []
bics_train = []

model_coefs = []

for it in range(100):
    # split first
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=10000, shuffle=True
    )

    # fold CV on TRAIN ONLY
    kf = KFold(n_splits=5, shuffle=True)

    fold_rmses = []
    fold_r2s = []
    fold_r2s_adj = []
    fold_aics = []
    fold_bics = []

    for train_idx, val_idx in kf.split(X_train):
        X_tr = X_train.iloc[train_idx][best_features]
        y_tr = y_train.iloc[train_idx]

        X_val = X_train.iloc[val_idx][best_features]
        y_val = y_train.iloc[val_idx]

        # scale inside fold
        X_tr_scaled = std_scaler_X.fit_transform(X_tr)
        y_tr_scaled = std_scaler_y.fit_transform(y_tr.values.reshape(-1, 1))

        X_val_scaled = std_scaler_X.transform(X_val)
        X_tr_scaled = sm.add_constant(X_tr_scaled)
        X_val_scaled = sm.add_constant(X_val_scaled)

        m = sm.OLS(y_tr_scaled, X_tr_scaled).fit()
        y_pred_val = m.predict(X_val_scaled)
        y_pred_val = std_scaler_y.inverse_transform(y_pred_val.reshape(-1, 1))

        fold_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
        fold_r2 = r2_score(y_val, y_pred_val)

        fold_rmses.append(fold_rmse)
        fold_r2s.append(fold_r2)
        fold_r2s_adj.append(m.rsquared_adj)
        fold_aics.append(m.aic)
        fold_bics.append(m.bic)

    # store avg training CV metrics for this iteration
    rmses_train.append(np.mean(fold_rmses))
    r2s_train.append(np.mean(fold_r2s))
    r2s_adj_train.append(np.mean(fold_r2s_adj))
    aics_train.append(np.mean(fold_aics))
    bics_train.append(np.mean(fold_bics))

    # now train final model on full train set
    X_train_sel = X_train[best_features]
    X_test_sel = X_test[best_features]

    X_train_scaled = std_scaler_X.fit_transform(X_train_sel)
    y_train_scaled = std_scaler_y.fit_transform(y_train.values.reshape(-1, 1))
    X_test_scaled = std_scaler_X.transform(X_test_sel)

    X_train_scaled = sm.add_constant(X_train_scaled)
    X_test_scaled = sm.add_constant(X_test_scaled)

    final_model = sm.OLS(y_train_scaled, X_train_scaled).fit()

    y_pred_test = final_model.predict(X_test_scaled)
    y_pred_test = std_scaler_y.inverse_transform(y_pred_test.reshape(-1, 1))

    rmses_test.append(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    r2s_test.append(r2_score(y_test, y_pred_test))
    model_coefs.append(final_model.params)

### BEST model by lowest test RMSE
idx = np.argmin(rmses_test)

print("BEST model coefficients:\n", model_coefs[idx])
print(f"Best test RMSE: {rmses_test[idx]:.4f}")
print(f"Best test R2:   {r2s_test[idx]:.4f}")

print("\n=== AVERAGE TRAIN CV METRICS ===")
print(f"Train CV RMSE: {np.mean(rmses_train):.4f} (+/- {np.std(rmses_train):.4f})")
print(f"Train CV R2:   {np.mean(r2s_train):.4f} (+/- {np.std(r2s_train):.4f})")
print(f"Train CV AdjR2:{np.mean(r2s_adj_train):.4f} (+/- {np.std(r2s_adj_train):.4f})")
print(f"Train CV AIC:  {np.mean(aics_train):.4f}")
print(f"Train CV BIC:  {np.mean(bics_train):.4f}")

print("\n=== AVERAGE TEST METRICS ===")
print(f"Test RMSE: {np.mean(rmses_test):.4f} (+/- {np.std(rmses_test):.4f})")
print(f"Test R2:   {np.mean(r2s_test):.4f} (+/- {np.std(r2s_test):.4f})")

# Plot results
plt.figure(figsize=(24, 12))

# plt.subplot(1, 2, 1)
plt.plot(rmses_test, '-', alpha=0.6)
plt.axhline(np.mean(rmses_test), color='r', linestyle='--', label=f'Mean: {np.mean(rmses_test):.2f}')
plt.axhline(rmses_test[idx], color='g', linestyle='--', label=f'Best: {rmses_test[idx]:.2f}')
plt.xlabel('Number of Features')
plt.ylabel('RMSE')
plt.title('RMSE - Outlier Clipping')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()