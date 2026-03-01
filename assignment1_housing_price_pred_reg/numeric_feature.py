import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import statsmodels.api as sm

from feature_sel import select_features_rfe, build_evaluate_predictor, select_features_ffs

df_new = pd.read_csv('data/df_new.csv')

selected_features = ['accommodates', 'bathrooms', 'cleaning_fee',
                     'review_scores_rating',
                     'bedrooms', 'beds',]

X = df_new[selected_features]
# del X['price']
y = df_new['price']


mm_scaler_X = MinMaxScaler()
mm_scaler_y = MinMaxScaler()
std_scaler_X = StandardScaler()
std_scaler_y = StandardScaler()


# print("\n*************** MM Scaler *************")
# for i in range(6, 30):
#         features_rfe = select_features_rfe(X, y, i, mm_scaler_X, std_scaler_y)
#         build_evaluate_predictor(features_rfe, X, y, mm_scaler_X, std_scaler_y)
#
#         features_ffs = select_features_ffs(X, y, i, mm_scaler_X)
#         build_evaluate_predictor(features_ffs, X, y, mm_scaler_X, std_scaler_y)
#
# print("\n*************** STD Scaler *************")
# for i in range(6, 30):
#         features_rfe = select_features_rfe(X, y, i, std_scaler_X, std_scaler_y)
#         build_evaluate_predictor(features_rfe, X, y, std_scaler_X, std_scaler_y)
#
#         features_ffs = select_features_ffs(X, y, i, std_scaler_X)
#         build_evaluate_predictor(features_ffs, X, y, std_scaler_X, std_scaler_y)



for i in range(100):
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=10000, shuffle=True
    # )
    # # selected_features = ['accommodates', 'bathrooms', , 'number_of_reviews',
    # #                      'review_scores_rating', 'bedrooms', 'beds', 'neighbourhood_avg_price',
    # #                      ]
    # # X_train = X_train[selected_features]
    # # X_test = X_test[selected_features]
    #
    # k_fold = KFold(n_splits=5, shuffle=True)
    # cv_scores = []
    # rmses = []
    #
    # for train_idx, val_idx in k_fold.split(X_train):
    #     X_tr = sm.add_constant(X_train.iloc[train_idx, :], has_constant='add')
    #     X_val = sm.add_constant(X_train.iloc[val_idx, :], has_constant='add')
    #     y_tr = y_train.iloc[train_idx]
    #     y_val = y_train.iloc[val_idx]
    #
    #     model = sm.OLS(y_tr, X_tr).fit()
    #     y_pred = model.predict(X_val)
    #
    #     cv_scores.append(r2_score(y_val, y_pred))
    #     rmses.append(np.sqrt(mean_squared_error(y_val, y_pred)))
    #
    # print("=" * 50)
    # print(f"CV R2: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
    # print(f"RMSE : {np.mean(rmses):.4f} (+/-{np.std(rmses):.4f})")
    #
    # # Final model on full training set
    # X_train_scaled = sm.add_constant(X_train, has_constant='add')
    # X_test_scaled = sm.add_constant(X_test, has_constant='add')
    # final_model = sm.OLS(y_train, X_train_scaled).fit()
    # y_pred = final_model.predict(X_test_scaled)
    # # print(f"y_pred: {y_pred}")
    # print(f"Test R2: {r2_score(y_test, y_pred)}")
    # print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}\n\n")
    # print(final_model.summary())
    # print("\n\n")
    # print("@" * 100)
    # print("\n\n\n")


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=10000, shuffle=True
    )
    # selected_features = ['accommodates', 'bathrooms', 'cleaning_fee', 'number_of_reviews',
    #        'review_scores_rating', 'bedrooms', 'beds', 'elevator',
    #        'suitable_for_events', 'neighbourhood_avg_price',
    #        'room_type_Private room', 'room_type_Shared room',
    #        'cancellation_policy_super_strict_60', 'city_Chicago', 'city_LA'
    #      ]
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(np.array(y_train).reshape(-1, 1)).ravel()

    k_fold = KFold(n_splits=5, shuffle=True)
    cv_scores = []
    rmses = []

    for train_idx, val_idx in k_fold.split(X_train_scaled):
        X_tr = sm.add_constant(X_train_scaled[train_idx], has_constant='add')
        X_val = sm.add_constant(X_train_scaled[val_idx], has_constant='add')
        y_tr = y_train_scaled[train_idx]
        y_val = y_train_scaled[val_idx]

        model = sm.OLS(y_tr, X_tr).fit()
        y_pred = model.predict(X_val)
        y_pred_orig = scaler_y.inverse_transform(np.array(y_pred).reshape(-1, 1)).ravel()
        # print(f"Y_pred_org : {y_pred_orig}")
        y_val_orig = scaler_y.inverse_transform(np.array(y_val).reshape(-1, 1)).ravel()
        # print(f"y_val_org: {y_val_orig}")
        cv_scores.append(r2_score(y_val_orig, y_pred_orig))
        rmses.append(np.sqrt(mean_squared_error(y_val_orig, y_pred_orig)))

    print("="*50)
    print(f"CV R2: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
    print(f"RMSE : {np.mean(rmses):.4f} (+/-{np.std(rmses):.4f})")

    # Final model on full training set
    X_train_scaled = sm.add_constant(X_train_scaled, has_constant='add')
    X_test_scaled = sm.add_constant(X_test_scaled, has_constant='add')
    final_model = sm.OLS(y_train, X_train_scaled).fit()
    y_pred = final_model.predict(X_test_scaled)
    # print(f"y_pred: {y_pred}")
    print(f"Test R2: {r2_score(y_test, y_pred)}")
    print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}\n\n")
    print(final_model.summary())
    print("\n\n")
    print("@"*100)
    print("\n\n\n")





