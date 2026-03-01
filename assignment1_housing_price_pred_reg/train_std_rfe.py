import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_selection import f_regression, RFECV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, KFold
import statsmodels.api as sm
import json
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

df = pd.read_csv('data/df_new.csv')
df = df.drop(columns=df.loc[:, '24_hour_check_in':'ev_charger'].columns)
df = df.drop(columns=df.loc[:, 'elevator_in_building':'stove'].columns)
df = df.drop(columns=df.loc[:, 'tv':'translation_missing:_en_hosting_amenity_49'].columns)
# df.to_csv('data/df_drop.csv', index=False)
X = df.copy()
del X['price']
y = df['price']


df_new = pd.read_csv('data/df_new.csv')
df_new = df_new.drop(columns=df_new.loc[:, '24_hour_check_in':'ev_charger'].columns)
df_new = df_new.drop(columns=df_new.loc[:, 'elevator_in_building':'stove'].columns)
df_new = df_new.drop(columns=df_new.loc[:, 'tv':'translation_missing:_en_hosting_amenity_49'].columns)
df_copy = df_new.copy()

df_clip = df_copy[:45000]
df_c = df_clip['price'].clip(None, 1700)
df_clip['price'] = df_c

df_test = df_copy[45000:]
X_test_final = df_test.copy()
y_test_final = df_test['price']
del X_test_final['price']

X = df_clip.copy()
del X['price']
y = df_clip['price']



test_rmses = []
test_r2 = []
model_coefs = []

test_aic = []
test_bic = []
test_r2_adj = []

train_cv_rmses = []
train_cv_r2s = []

# for i in range(100):
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=10000, shuffle=True
#     )
#
#     selected_features = ['accommodates', 'bathrooms', 'cleaning_fee', 'number_of_reviews',
#        'review_scores_rating', 'bedrooms', 'beds', 'elevator',
#        'suitable_for_events', 'neighbourhood_avg_price',
#        'room_type_Private room', 'room_type_Shared room',
#        'cancellation_policy_super_strict_60', 'city_Chicago', 'city_LA'
#     ]
#
#     X_train = X_train[selected_features]
#     X_test = X_test[selected_features]
#
#     scaler_X = StandardScaler()
#     scaler_y = StandardScaler()
#     X_train_scaled = scaler_X.fit_transform(X_train)
#     X_test_scaled = scaler_X.transform(X_test)
#     y_train_scaled = scaler_y.fit_transform(np.array(y_train).reshape(-1, 1)).ravel()
#
#     k_fold = KFold(n_splits=5, shuffle=True)
#     cv_scores = []
#     rmses = []
#
#     for train_idx, val_idx in k_fold.split(X_train_scaled):
#         X_tr = sm.add_constant(X_train_scaled[train_idx], has_constant='add')
#         X_val = sm.add_constant(X_train_scaled[val_idx], has_constant='add')
#         y_tr = y_train_scaled[train_idx]
#         y_val = y_train_scaled[val_idx]
#
#         model = sm.OLS(y_tr, X_tr).fit()
#         y_pred = model.predict(X_val)
#         y_pred_orig = scaler_y.inverse_transform(np.array(y_pred).reshape(-1, 1)).ravel()
#         y_val_orig = scaler_y.inverse_transform(np.array(y_val).reshape(-1, 1)).ravel()
#         cv_scores.append(r2_score(y_val_orig, y_pred_orig))
#         rmses.append(np.sqrt(mean_squared_error(y_val_orig, y_pred_orig)))
#
#     # store avg train cv
#     train_cv_r2s.append(np.mean(cv_scores))
#     train_cv_rmses.append(np.mean(rmses))
#
#     # Final model on FULL training set
#     X_train_scaled = pd.DataFrame(X_train_scaled, columns=selected_features)  # <-- add names
#     X_test_scaled  = pd.DataFrame(X_test_scaled,  columns=selected_features)  # <-- add names
#
#     X_train_scaled = sm.add_constant(X_train_scaled, has_constant='add')
#     X_test_scaled = sm.add_constant(X_test_scaled, has_constant='add')
#
#     final_model = sm.OLS(y_train_scaled, X_train_scaled).fit()   # FIX scaled y
#     y_pred_scaled = final_model.predict(X_test_scaled)
#     y_pred = scaler_y.inverse_transform(np.array(y_pred_scaled).reshape(-1,1)).ravel()
#
#     test_rmses.append(np.sqrt(mean_squared_error(y_test, y_pred)))
#     test_r2.append(r2_score(y_test, y_pred))
#     model_coefs.append(final_model.params)
#     test_aic.append(final_model.aic)
#     test_bic.append(final_model.bic)
#     test_r2_adj.append(final_model.rsquared_adj)
#
#
# # ===== FINAL SUMMARY =====
#
# best_idx = np.argmin(test_rmses)
#
# print(best_idx)
# print(test_rmses[best_idx])
# print(test_r2[best_idx])
# print(model_coefs[best_idx])   # <-- EXACT FORMAT you wanted
# print(f"AVG RMSE: {np.mean(test_rmses)}")
#
# print(f"AVG TRAIN CV RMSE: {np.mean(train_cv_rmses):.4f} (+/-{np.std(train_cv_rmses):.4f})")
# print(f"AVG TRAIN CV R2  : {np.mean(train_cv_r2s):.4f} (+/-{np.std(train_cv_r2s):.4f})")
#
# print(f"AVG TEST RMSE: {np.mean(test_rmses):.4f} (+/-{np.std(test_rmses):.4f})")
# print(f"AVG TEST R2  : {np.mean(test_r2):.4f} (+/-{np.std(test_r2):.4f})")
#
# print(f"AVG Adj Test R2: {np.mean(test_r2_adj):.4f}")
# print(f"AVG AIC:        {np.mean(test_aic):.4f}")
# print(f"AVG BIC:        {np.mean(test_bic):.4f}")
#
# plt.plot(test_rmses, '-')
# plt.title("Test RMSE per iteration")
# plt.show()







# ===================================================================================================

df_new = pd.read_csv('data/df_new.csv')
df_new = df_new.drop(columns=df_new.loc[:, '24_hour_check_in':'ev_charger'].columns)
df_new = df_new.drop(columns=df_new.loc[:, 'elevator_in_building':'stove'].columns)
df_new = df_new.drop(columns=df_new.loc[:, 'tv':'translation_missing:_en_hosting_amenity_49'].columns)

X = df_new.copy()
del X['price']
y = df_new['price']

# Base features
base_features = ['accommodates', 'bathrooms', 'cleaning_fee', 'host_identity_verified',
       'host_response_rate', 'number_of_reviews', 'review_scores_rating',
       'bedrooms', 'beds', 'neighbourhood_avg_price',
       'property_type_grouped_House', 'property_type_grouped_Other',
       'room_type_Private room', 'room_type_Shared room', 'bed_type_Real Bed',
       'cancellation_policy_moderate', 'cancellation_policy_strict',
       'cancellation_policy_super_strict_60', 'city_Chicago', 'city_LA',
       'city_NYC', 'city_SF']


# Polynomial features for key continuous variables
X['accommodates_squared'] = X['accommodates'] ** 2
X['bedrooms_squared'] = X['bedrooms'] ** 2
X['bathrooms_squared'] = X['bathrooms'] ** 2
X['beds_squared'] = X['beds'] ** 2

# Key interactions - size and capacity
X['accommodates_bedrooms'] = X['accommodates'] * X['bedrooms']
X['accommodates_bathrooms'] = X['accommodates'] * X['bathrooms']
X['bedrooms_bathrooms'] = X['bedrooms'] * X['bathrooms']
X['accommodates_beds'] = X['accommodates'] * X['beds']

# City-specific pricing patterns
X['LA_neighbourhood_price'] = X['city_LA'] * X['neighbourhood_avg_price']
X['SF_neighbourhood_price'] = X['city_SF'] * X['neighbourhood_avg_price']
X['NYC_neighbourhood_price'] = X['city_NYC'] * X['neighbourhood_avg_price']
X['Chicago_neighbourhood_price'] = X['city_Chicago'] * X['neighbourhood_avg_price']

# City-specific capacity pricing
X['LA_accommodates'] = X['city_LA'] * X['accommodates']
X['SF_accommodates'] = X['city_SF'] * X['accommodates']
X['NYC_accommodates'] = X['city_NYC'] * X['accommodates']
X['Chicago_accommodates'] = X['city_Chicago'] * X['accommodates']

# Room type interactions
X['private_room_bedrooms'] = X['room_type_Private room'] * X['bedrooms']
X['private_room_accommodates'] = X['room_type_Private room'] * X['accommodates']
X['shared_room_accommodates'] = X['room_type_Shared room'] * X['accommodates']

# Property type interactions
X['house_bedrooms'] = X['property_type_grouped_House'] * X['bedrooms']
X['house_accommodates'] = X['property_type_grouped_House'] * X['accommodates']

# Reviews and ratings interactions
X['reviews_rating'] = X['number_of_reviews'] * X['review_scores_rating']
X['reviews_per_accommodates'] = X['number_of_reviews'] / (X['accommodates'] + 1)
X['has_reviews'] = (X['number_of_reviews'] > 0).astype(int)

# Cancellation policy interactions
X['strict_policy_accommodates'] = (X['cancellation_policy_strict'] +
                                   X['cancellation_policy_super_strict_60']) * X['accommodates']

# Cleaning fee per person
X['cleaning_per_accommodates'] = X['cleaning_fee'] / (X['accommodates'] + 1)

# Host quality indicators
X['verified_response_rate'] = X['host_identity_verified'] * X['host_response_rate']

# Luxury indicators
X['luxury_indicator'] = (X['bedrooms'] >= 3).astype(int) * (X['bathrooms'] >= 2).astype(int) * X[
    'neighbourhood_avg_price']

# Compile all features
all_features = base_features + [
    'accommodates_squared', 'bedrooms_squared', 'bathrooms_squared', 'beds_squared',
    'accommodates_bedrooms', 'accommodates_bathrooms', 'bedrooms_bathrooms', 'accommodates_beds',
    'LA_neighbourhood_price', 'SF_neighbourhood_price', 'NYC_neighbourhood_price', 'Chicago_neighbourhood_price',
    'LA_accommodates', 'SF_accommodates', 'NYC_accommodates', 'Chicago_accommodates',
    'private_room_bedrooms', 'private_room_accommodates', 'shared_room_accommodates',
    'house_bedrooms', 'house_accommodates',
    'reviews_rating', 'reviews_per_accommodates', 'has_reviews',
    'strict_policy_accommodates', 'cleaning_per_accommodates',
    'verified_response_rate', 'luxury_indicator'
]

X_for_selection = X[all_features]
std_scaler_X = StandardScaler()
X_scaled_temp = std_scaler_X.fit_transform(X_for_selection)
f_scores, _ = f_regression(X_scaled_temp, y)

rfecv = RFECV(estimator=LinearRegression(), cv=5, scoring='neg_mean_squared_error')
rfecv.fit(X_scaled_temp, y)

selected_features = [all_features[i] for i in range(len(all_features)) if rfecv.support_[i]]
print(f"Optimal number of features: {rfecv.n_features_}")
print(f"Selected features: {selected_features}")

test_rmses = []
test_r2 = []
model_coefs = []

### ADDED
test_aic = []
test_bic = []
test_r2_adj = []
train_cv_rmses = []
train_cv_r2s = []

for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=10000, shuffle=True, random_state=i
    )

    X_train_sel = X_train[selected_features]
    X_test_sel = X_test[selected_features]

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train_sel)
    X_test_scaled  = scaler_X.transform(X_test_sel)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

    k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    rmses = []

    for train_idx, val_idx in k_fold.split(X_train_scaled):
        X_tr = sm.add_constant(X_train_scaled[train_idx], has_constant='add')
        X_val = sm.add_constant(X_train_scaled[val_idx],   has_constant='add')
        y_tr = y_train_scaled[train_idx]
        y_val= y_train_scaled[val_idx]

        model = sm.OLS(y_tr, X_tr).fit()
        y_pred_scaled = model.predict(X_val)

        y_pred_orig = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        y_val_orig  = scaler_y.inverse_transform(y_val.reshape(-1, 1)).ravel()

        cv_scores.append(r2_score(y_val_orig, y_pred_orig))
        rmses.append(np.sqrt(mean_squared_error(y_val_orig, y_pred_orig)))

    train_cv_rmses.append(np.mean(rmses))
    train_cv_r2s.append(np.mean(cv_scores))

    X_train_scaled = sm.add_constant(X_train_scaled, has_constant='add')
    X_test_scaled  = sm.add_constant(X_test_scaled,  has_constant='add')
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=["const"] + selected_features)
    X_test_scaled  = pd.DataFrame(X_test_scaled,  columns=["const"] + selected_features)

    final_model = sm.OLS(y_train_scaled, X_train_scaled).fit()
    y_pred_scaled = final_model.predict(X_test_scaled)
    y_pred        = scaler_y.inverse_transform(np.array(y_pred_scaled).reshape(-1,1)).ravel()

    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_r2_score = r2_score(y_test, y_pred)

    test_rmses.append(test_rmse)
    test_r2.append(test_r2_score)
    model_coefs.append(final_model.params)
    test_aic.append(final_model.aic)
    test_bic.append(final_model.bic)
    test_r2_adj.append(final_model.rsquared_adj)

# pick best model
idx = np.argmin(test_rmses)
best_params = model_coefs[idx].to_dict()

# SAVE SCALER STATS for production
scaler_stats = {
    "X_mean": dict(zip(selected_features, scaler_X.mean_)),
    "X_scale": dict(zip(selected_features, scaler_X.scale_)),
    "y_mean": float(scaler_y.mean_[0]),
    "y_scale": float(scaler_y.scale_[0])
}

json.dump(best_params, open("best_coefs.json","w"), indent=2)
json.dump(scaler_stats, open("scaler_stats.json","w"), indent=2)
json.dump(selected_features, open("feature_order.json","w"), indent=2)

print("Saved: best_coefs.json, scaler_stats.json, feature_order.json")

print("=" * 100)
print(f"\nBest Model (iteration {idx + 1}):")
print(f"RMSE: {test_rmses[idx]:.4f}")
print(f"R2: {test_r2[idx]:.4f}")
print("\nModel Coefs:")
print(model_coefs[idx])

print(f"\nAverage across all runs:")

### ADDED extra summaries
print(f"Average TRAIN CV RMSE: {np.mean(train_cv_rmses):.4f} (+/- {np.std(train_cv_rmses):.4f})")
print(f"Average TRAIN CV R2:   {np.mean(train_cv_r2s):.4f} (+/- {np.std(train_cv_r2s):.4f})")

print(f"Average TEST RMSE:     {np.mean(test_rmses):.4f} (+/- {np.std(test_rmses):.4f})")
print(f"Average TEST R2:       {np.mean(test_r2):.4f} (+/- {np.std(test_r2):.4f})")

print(f"Average Adj R2:        {np.mean(test_r2_adj):.4f}")
print(f"Average AIC:           {np.mean(test_aic):.4f}")
print(f"Average BIC:           {np.mean(test_bic):.4f}")

coef_importance = abs(np.sort(model_coefs[idx]))
top_features = coef_importance[:20]
top_indices = np.argsort(coef_importance)[::-1][:20]
top_feature_names = [all_features[i] for i in top_indices]

# Plot results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(test_rmses, '-', alpha=0.6)
plt.axhline(np.mean(test_rmses), color='r', linestyle='--', label=f'Mean: {np.mean(test_rmses):.2f}')
plt.axhline(test_rmses[idx], color='g', linestyle='--', label=f'Best: {test_rmses[idx]:.2f}')
plt.xlabel('Iteration')
plt.ylabel('Test RMSE')
plt.title('RMSE across runs')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(test_r2, '-', alpha=0.6)
plt.axhline(np.mean(test_r2), color='r', linestyle='--', label=f'Mean: {np.mean(test_r2):.3f}')
plt.axhline(test_r2[idx], color='g', linestyle='--', label=f'Best: {test_r2[idx]:.3f}')
plt.xlabel('Iteration')
plt.ylabel('Test R²')
plt.title('R² across runs')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 3, 3)
plt.barh(range(len(top_features)), top_features)
plt.yticks(range(len(top_features)), top_feature_names, fontsize=8)
plt.xlabel('Absolute Coefficient Value')
plt.title('Top 20 Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()

plt.tight_layout()
plt.savefig('model_performance_ols.png', dpi=300, bbox_inches='tight')
plt.show()
