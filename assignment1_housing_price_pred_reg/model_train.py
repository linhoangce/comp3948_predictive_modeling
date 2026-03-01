from pandas.core.dtypes.common import is_numeric_dtype
from scipy.stats import zscore
from sklearn import metrics
from sklearn.feature_selection import RFE, SelectKBest, f_regression, RFECV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import pandas as pd
import json
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 1000)

df = pd.read_csv('data/AirBNB.csv', low_memory=False)
print(df['price'].describe())
# print(df.describe())

na_df = df.isna().sum().reset_index()
na_df.columns = ["Column", "Missing"]
na_df.sort_values(['Missing'], ascending=False, inplace=True)

# print("Number of missing values for Columns")
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#     print(na_df)

# =========================================================================
# Plot outliers with boxplot
# =========================================================================
def box_plot(df, features):
    plt.subplots(nrows=1, ncols=len(features), figsize=(14,7))
    plt.xticks([])
    for i in range(len(features)):
        plt.subplot(1, len(features), i+1)
        df.boxplot(column=[features[i]])
    plt.show()

columns = ['price']
box_plot(df, columns)

# ====================================================================================
# Impute Missing Data
# ====================================================================================
def impute_nan(col_name, df, measure_type):
    # create mask for missing data column
    mask = df[col_name].isna()

    if measure_type.lower() == "mean":
        imp_val = df[col_name].mean()
    elif measure_type.lower() == "median":
        imp_val = df[col_name].median()
    elif measure_type.lower() == "mode":
        imp_val = df[col_name].mode()[0]
    else:
        print("Invalid measure type")

    df.loc[mask, col_name] = imp_val
    return df

# Impute numeric cols with mean and categorical cols with mode
missing_df = df.columns[df.isna().any()]
for col in missing_df:
    if is_numeric_dtype(df[col]):
        df = impute_nan(col, df, "mean")
    else:
        df = impute_nan(col, df, "mode")

print("="*100)

df_new = df.copy()

# ====================================================================
# Group/Bin property_type by grouping similar types together to reduce
# dimensionality instead of one-hot encoding
# =====================================================================
def group_property_types(df):
    property_mapping = {
        'Apartment': 'Apartment',
        'Condominium': 'Apartment',
        'Loft': 'Apartment',

        # Houses
        'House': 'House',
        'Townhouse': 'House',
        'Villa': 'House',
        'Bungalow': 'House',
        'In-law': 'House',
        'Cabin': 'House',
        'Chalet': 'House',

        # Guest Accommodations
        'Guesthouse': 'Guest_Accommodation',
        'Guest suite': 'Guest_Accommodation',
        'Bed & Breakfast': 'Guest_Accommodation',

        # Hotels/Hostels
        'Hostel': 'Hotel_Style',
        'Boutique hotel': 'Hotel_Style',
        'Serviced apartment': 'Hotel_Style',

        # Unique/Specialty
        'Boat': 'Unique',
        'Camper/RV': 'Unique',
        'Treehouse': 'Unique',
        'Yurt': 'Unique',
        'Castle': 'Unique',
        'Cave': 'Unique',
        'Tent': 'Unique',
        'Tipi': 'Unique',
        'Hut': 'Unique',
        'Island': 'Unique',
        'Train': 'Unique',
        'Earth House': 'Unique',

        # Other/Miscellaneous
        'Other': 'Other',
        'Dorm': 'Other',
        'Timeshare': 'Other',
        'Vacation home': 'Other',
        'Casa particular': 'Other',
        'Parking Space': 'Other'
    }

    df['property_type_grouped'] = df['property_type'].map(property_mapping)
    del df['property_type']
    return df

df_new = group_property_types(df)

# =====================================================================================
# Process DateTime data
# =====================================================================================
def convert_to_standard_datetime(df, col_name):
    df[col_name] = pd.to_datetime(df[col_name], format="mixed")
    return df

# convert since_host to years of experience and then bin them
convert_to_standard_datetime(df_new, 'host_since')

df_new['host_years'] = (df_new['host_since'].max() - df_new['host_since']).dt.days/ 365
df_new['host_exp'] = pd.cut(df_new['host_years'],
                            bins=[0, 2, 5, 7, 10],
                            labels=['new', 'mid', 'experienced', 'veteran'])

def convert_str_to_numeric(col_name, char, df):
    df[col_name] = df[col_name].astype(str).str.replace(char, "")
    df[col_name] = pd.to_numeric(df[col_name], errors='coerce').fillna(0)
    return df

df_new = convert_str_to_numeric("host_response_rate", "%", df_new)

# ===========================================================================
# Compute average price for neighbourhoods, so drop zipcode
# ===========================================================================
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#     print(df['neighbourhood'].value_counts())

# print(len(pd.unique(df['neighbourhood'])))

mean_price_per_neigh = df.groupby('neighbourhood')['price'].median()
df_new['neighbourhood_avg_price'] = df_new['neighbourhood'].map(mean_price_per_neigh)
del df_new['neighbourhood']
del df_new['zipcode']

# ============================================================================
# Dummy Variables for Categorical Columns
# ============================================================================
df_new = pd.get_dummies(df_new, drop_first=True, dtype=int,
                        columns=['property_type_grouped', 'room_type', 'bed_type',
                                 'cancellation_policy', 'city', 'host_exp'])

# Convert bool columns' values from bool to int
df_new['cleaning_fee'] = df_new['cleaning_fee'].astype(int)

bool_columns = ['host_has_profile_pic', 'host_identity_verified',
           'instant_bookable']
for col in bool_columns:
    for i in range(len(df_new)):
        if df_new.iloc[i][col] == 't':
            df_new.at[i, col] = 1
        else:
            df_new.at[i, col] = 0

datetime_cols = ['first_review', 'host_since', 'last_review']
for col in datetime_cols:
    del df_new[col]

# ============================================================================
# CREATE POLYNOMIAL AND INTERACTION FEATURES
# ============================================================================

# Polynomial features for key continuous variables
df_new['accommodates_squared'] = df_new['accommodates'] ** 2
df_new['bedrooms_squared'] = df_new['bedrooms'] ** 2
df_new['bathrooms_squared'] = df_new['bathrooms'] ** 2
df_new['beds_squared'] = df_new['beds'] ** 2

# Key interactions - size and capacity
df_new['accommodates_bedrooms'] = df_new['accommodates'] * df_new['bedrooms']
df_new['accommodates_bathrooms'] = df_new['accommodates'] * df_new['bathrooms']
df_new['bedrooms_bathrooms'] = df_new['bedrooms'] * df_new['bathrooms']
df_new['accommodates_beds'] = df_new['accommodates'] * df_new['beds']

# City-specific pricing patterns (check if these columns exist after get_dummies)
city_cols = ['city_LA', 'city_SF', 'city_NYC', 'city_Chicago']
for city_col in city_cols:
    if city_col in df_new.columns:
        city_name = city_col.replace('city_', '')
        df_new[f'{city_name}_neighbourhood_price'] = df_new[city_col] * df_new['neighbourhood_avg_price']
        df_new[f'{city_name}_accommodates'] = df_new[city_col] * df_new['accommodates']

# Room type interactions (check if these columns exist after get_dummies)
if 'room_type_Private room' in df_new.columns:
    df_new['private_room_bedrooms'] = df_new['room_type_Private room'] * df_new['bedrooms']
    df_new['private_room_accommodates'] = df_new['room_type_Private room'] * df_new['accommodates']

if 'room_type_Shared room' in df_new.columns:
    df_new['shared_room_accommodates'] = df_new['room_type_Shared room'] * df_new['accommodates']

# Property type interactions (check if these columns exist after get_dummies)
if 'property_type_grouped_House' in df_new.columns:
    df_new['house_bedrooms'] = df_new['property_type_grouped_House'] * df_new['bedrooms']
    df_new['house_accommodates'] = df_new['property_type_grouped_House'] * df_new['accommodates']

# Reviews and ratings interactions
df_new['reviews_rating'] = df_new['number_of_reviews'] * df_new['review_scores_rating']
df_new['reviews_per_accommodates'] = df_new['number_of_reviews'] / (df_new['accommodates'] + 1)
df_new['has_reviews'] = (df_new['number_of_reviews'] > 0).astype(int)

# Cancellation policy interactions (check if these columns exist)
if 'cancellation_policy_strict' in df_new.columns and 'cancellation_policy_super_strict_60' in df_new.columns:
    df_new['strict_policy_accommodates'] = (df_new['cancellation_policy_strict'] +
                                            df_new['cancellation_policy_super_strict_60']) * df_new['accommodates']

# Cleaning fee per person
df_new['cleaning_per_accommodates'] = df_new['cleaning_fee'] / (df_new['accommodates'] + 1)

# Host quality indicators
df_new['verified_response_rate'] = df_new['host_identity_verified'] * df_new['host_response_rate']

# Luxury indicators
df_new['luxury_indicator'] = ((df_new['bedrooms'] >= 3).astype(int) *
                              (df_new['bathrooms'] >= 2).astype(int) *
                              df_new['neighbourhood_avg_price'])

df_new.to_csv('data/df_new.csv', index=False)


# ==============================================================================================
# ============= FEATURE SELECTION =====================
# ==============================================================================================

def select_features_rfe(X, y, n_features, scaler_X, scaler_y):
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(np.array(y).reshape(-1, 1))

    rfe = RFE(LinearRegression(), n_features_to_select=n_features)
    rfe.fit(X_scaled, y_scaled)
    selected_features = X.keys()[rfe.support_ == True]
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

df_new = pd.read_csv('data/df_new.csv')

# Remove irrelevant features identified by several rounds of running
# feature selection algorithms to improve running time
df_new = df_new.drop(columns=df_new.loc[:, '24_hour_check_in':'ev_charger'].columns)
df_new = df_new.drop(columns=df_new.loc[:, 'elevator_in_building':'stove'].columns)
df_new = df_new.drop(columns=df_new.loc[:, 'tv':'translation_missing:_en_hosting_amenity_49'].columns)
mm_scaler_X = MinMaxScaler()
mm_scaler_y = MinMaxScaler()
std_scaler_X = StandardScaler()
std_scaler_y = StandardScaler()
rb_scaler_X = RobustScaler()
rb_scaler_y = RobustScaler()

X = df_new.copy()
del X['price']
y = df_new['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=10000, shuffle=True
)


def plot_feature_selection_result(rmse_bfs, features_rfe, params_rfe,
                                  rmse_ffs, features_ffs, params_ffs, scaler_name):
    # get min rmse index
    idx_bfs = np.argmin(rmse_bfs)

    # Plot results
    plt.figure(figsize=(24, 16))

    plt.subplot(2, 2, 1)
    plt.plot(rmse_bfs, '-', alpha=0.6)
    plt.axhline(np.mean(rmse_bfs), color='r', linestyle='--', label=f'Mean: {np.mean(rmse_bfs):.2f}')
    plt.axhline(rmse_bfs[idx_bfs], color='g', linestyle='--', label=f'Best: {rmse_bfs[idx_bfs]:.2f}')
    plt.xlabel('Number of Features')
    plt.ylabel('RMSE')
    plt.title('RMSE - BFS')
    plt.legend()
    plt.grid(alpha=0.3)

    idx_ffs = np.argmin(rmse_ffs)

    plt.subplot(2, 2, 2)
    plt.plot(rmse_ffs, '-', alpha=0.6)
    plt.axhline(np.mean(rmse_ffs), color='r', linestyle='--', label=f'Mean: {np.mean(rmse_ffs):.3f}')
    plt.axhline(rmse_ffs[idx_ffs], color='g', linestyle='--', label=f'Best: {rmse_ffs[idx_ffs]:.3f}')
    plt.xlabel('Number of Features')
    plt.ylabel('RMSE')
    plt.title('RMSE - FFS')
    plt.legend()
    plt.grid(alpha=0.3)

    coef_importance = abs(np.sort(params_rfe[idx_bfs]))

    plt.subplot(2, 2, 3)
    plt.barh(range(len(coef_importance)), coef_importance)
    plt.yticks(range(len(features_rfe)), features_rfe, fontsize=8)
    plt.xlabel('Absolute Coefficient Value')
    plt.title(f'Top {len(coef_importance)} Feature Importance - BFS')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    coef_importance = abs(np.sort(params_ffs[idx_ffs]))

    plt.subplot(2, 2, 4)
    plt.barh(range(len(coef_importance)), coef_importance)
    plt.yticks(range(len(features_ffs)), features_ffs, fontsize=8)
    plt.xlabel('Absolute Coefficient Value')
    plt.title(f'Top {len(coef_importance)}  Feature Importance - FFS')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    plt.tight_layout()
    plt.show()


# ======================================================================================
# Feature Selection - Scaling
# ======================================================================================
rmse_bfs_mm = []
r2_bfs_mm = []
best_bfs_mm = []
rmse_ffs_mm = []
r2_ffs_mm = []
best_ffs_mm = []
ffs_params = []
bfs_params = []
print("\n*************** MM Scaler *************")
for i in range(6, 30):
        features_rfe = select_features_rfe(X, y, i, mm_scaler_X, std_scaler_y)
        result = build_evaluate_predictor(features_rfe, X, y, mm_scaler_X, std_scaler_y)
        rmse_bfs_mm.append(result['rmse'])
        r2_bfs_mm.append(result['r2'])
        best_bfs_mm.append(features_rfe)
        bfs_params.append(result['model'])

        features_ffs = select_features_ffs(X, y, i, mm_scaler_X)
        result = build_evaluate_predictor(features_ffs, X, y, mm_scaler_X, std_scaler_y)
        rmse_ffs_mm.append(result['rmse'])
        r2_ffs_mm.append(result['r2'])
        best_ffs_mm.append(features_rfe)
        ffs_params.append(result['model'])


plot_feature_selection_result(rmse_bfs_mm, features_rfe, bfs_params,
                              rmse_ffs_mm, features_ffs, ffs_params, "MinMax Scaling")

# ================================================================================================================

print("\n*************** STD Scaler *************")
rmse_bfs_std = []
r2_bfs_std = []
best_bfs_std = []
rmse_ffs_std = []
r2_ffs_std = []
best_ffs_std = []
ffs_params = []
bfs_params = []

for i in range(6, 30):
        features_rfe = select_features_rfe(X, y, i, std_scaler_X, std_scaler_y)
        result = build_evaluate_predictor(features_rfe, X, y, std_scaler_X, std_scaler_y)
        rmse_bfs_std.append(result['rmse'])
        r2_bfs_std.append(result['r2'])
        best_bfs_std.append(features_rfe)
        bfs_params.append(result['model'])

        features_ffs = select_features_ffs(X, y, i, std_scaler_X)
        result = build_evaluate_predictor(features_ffs, X, y, std_scaler_X, std_scaler_y)
        rmse_ffs_std.append(result['rmse'])
        r2_ffs_std.append(result['r2'])
        best_ffs_std.append(features_rfe)
        ffs_params.append(result['model'])

plot_feature_selection_result(rmse_bfs_std, features_rfe, bfs_params,
                              rmse_ffs_std, features_ffs, ffs_params, "Standard Scaling")

# =============================================================================================================

print("\n*************** Robust Scaler *************")
rmse_bfs_rb = []
r2_bfs_rb = []
best_bfs_rb = []
rmse_ffs_rb = []
r2_ffs_rb = []
best_ffs_rb = []
ffs_params = []
bfs_params = []
for i in range(6, 30):
        features_rfe = select_features_rfe(X, y, i, rb_scaler_X, rb_scaler_X)
        result = build_evaluate_predictor(features_rfe, X, y, rb_scaler_X, rb_scaler_X)
        rmse_bfs_rb.append(result['rmse'])
        r2_bfs_rb.append(result['r2'])
        best_bfs_rb.append(features_rfe)
        bfs_params.append(result['model'])

        features_ffs = select_features_ffs(X, y, i, rb_scaler_X)
        result = build_evaluate_predictor(features_ffs, X, y, rb_scaler_X, rb_scaler_X)
        rmse_ffs_rb.append(result['rmse'])
        r2_ffs_rb.append(result['r2'])
        best_ffs_rb.append(features_rfe)
        ffs_params.append(result['model'])


plot_feature_selection_result(rmse_bfs_rb, features_rfe, bfs_params,
                              rmse_ffs_rb, features_ffs, ffs_params, "Robust Scaling")



# ======================================================================================
# Training with selected best features
# ======================================================================================

# ======================================================================================
# Model A - Training with Features from BFS
# ======================================================================================

### Droping insignificant columns to improve computation time
print('='*100)
print("******************** Model A TRAINING ***********************")
print('='*100)
df_copy = df_new.copy()

# hold out a subset for testing
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

for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=10000, shuffle=True
    )

    selected_features = ['accommodates', 'bathrooms', 'cleaning_fee', 'number_of_reviews',
       'review_scores_rating', 'bedrooms', 'beds', 'elevator',
       'suitable_for_events', 'neighbourhood_avg_price',
       'room_type_Private room', 'room_type_Shared room',
       'cancellation_policy_super_strict_60', 'city_Chicago', 'city_LA'
    ]

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
        y_val_orig = scaler_y.inverse_transform(np.array(y_val).reshape(-1, 1)).ravel()
        cv_scores.append(r2_score(y_val_orig, y_pred_orig))
        rmses.append(np.sqrt(mean_squared_error(y_val_orig, y_pred_orig)))

    # store avg train cv
    train_cv_r2s.append(np.mean(cv_scores))
    train_cv_rmses.append(np.mean(rmses))

    # Final model on FULL training set
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=selected_features)  # <-- add names
    X_test_scaled  = pd.DataFrame(X_test_scaled,  columns=selected_features)  # <-- add names

    X_train_scaled = sm.add_constant(X_train_scaled, has_constant='add')
    X_test_scaled = sm.add_constant(X_test_scaled, has_constant='add')

    final_model = sm.OLS(y_train_scaled, X_train_scaled).fit()   # FIX scaled y
    y_pred_scaled = final_model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(np.array(y_pred_scaled).reshape(-1,1)).ravel()

    test_rmses.append(np.sqrt(mean_squared_error(y_test, y_pred)))
    test_r2.append(r2_score(y_test, y_pred))
    model_coefs.append(final_model.params)
    test_aic.append(final_model.aic)
    test_bic.append(final_model.bic)
    test_r2_adj.append(final_model.rsquared_adj)

# ===== FINAL SUMMARY =====
best_idx = np.argmin(test_rmses)
print(best_idx)
print(test_rmses[best_idx])
print(test_r2[best_idx])
print(model_coefs[best_idx])   # <-- EXACT FORMAT you wanted
print(f"AVG RMSE: {np.mean(test_rmses)}")
print(f"AVG TRAIN CV RMSE: {np.mean(train_cv_rmses):.4f} (+/-{np.std(train_cv_rmses):.4f})")
print(f"AVG TRAIN CV R2  : {np.mean(train_cv_r2s):.4f} (+/-{np.std(train_cv_r2s):.4f})")
print(f"AVG TEST RMSE: {np.mean(test_rmses):.4f} (+/-{np.std(test_rmses):.4f})")
print(f"AVG TEST R2  : {np.mean(test_r2):.4f} (+/-{np.std(test_r2):.4f})")
print(f"AVG Adj Test R2: {np.mean(test_r2_adj):.4f}")
print(f"AVG AIC:        {np.mean(test_aic):.4f}")
print(f"AVG BIC:        {np.mean(test_bic):.4f}")

plt.plot(test_rmses, '-')
plt.title("Test RMSE per iteration")
plt.show()


# ===================================================================================
# Model B – BFS with refined Dataset
# ===================================================================================
### Droping insignificant columns to improve computation time
print('='*100)
print("******************** Model A TRAINING ***********************")
print('='*100)
df_new = pd.read_csv('data/df_new.csv')
df_new = df_new.drop(columns=df_new.loc[:, '24_hour_check_in':'ev_charger'].columns)
df_new = df_new.drop(columns=df_new.loc[:, 'elevator_in_building':'stove'].columns)
df_new = df_new.drop(columns=df_new.loc[:, 'tv':'translation_missing:_en_hosting_amenity_49'].columns)
X = df_new.copy()
y = df_new['price']
del X['price']

test_rmses = []
test_r2 = []
model_coefs = []

test_aic = []
test_bic = []
test_r2_adj = []

train_cv_rmses = []
train_cv_r2s = []

for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=10000, shuffle=True
    )

    selected_features = [
        'accommodates', 'bathrooms', 'cleaning_fee', 'host_identity_verified',
        'host_response_rate', 'number_of_reviews', 'review_scores_rating',
        'bedrooms', 'beds', 'neighbourhood_avg_price',
        'property_type_grouped_House',
        'property_type_grouped_Other',
        'room_type_Private room', 'room_type_Shared room', 'bed_type_Real Bed',
        'cancellation_policy_moderate', 'cancellation_policy_strict',
        'cancellation_policy_super_strict_60',
        'city_Chicago', 'city_LA', 'city_NYC', 'city_SF'
    ]

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
        y_val_orig = scaler_y.inverse_transform(np.array(y_val).reshape(-1, 1)).ravel()
        cv_scores.append(r2_score(y_val_orig, y_pred_orig))
        rmses.append(np.sqrt(mean_squared_error(y_val_orig, y_pred_orig)))

    # store avg train cv
    train_cv_r2s.append(np.mean(cv_scores))
    train_cv_rmses.append(np.mean(rmses))

    # Final model on FULL training set
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=selected_features)  # <-- add names
    X_test_scaled  = pd.DataFrame(X_test_scaled,  columns=selected_features)  # <-- add names

    X_train_scaled = sm.add_constant(X_train_scaled, has_constant='add')
    X_test_scaled = sm.add_constant(X_test_scaled, has_constant='add')

    final_model = sm.OLS(y_train_scaled, X_train_scaled).fit()   # FIX scaled y
    y_pred_scaled = final_model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(np.array(y_pred_scaled).reshape(-1,1)).ravel()

    test_rmses.append(np.sqrt(mean_squared_error(y_test, y_pred)))
    test_r2.append(r2_score(y_test, y_pred))
    model_coefs.append(final_model.params)
    test_aic.append(final_model.aic)
    test_bic.append(final_model.bic)
    test_r2_adj.append(final_model.rsquared_adj)

# ===== FINAL SUMMARY =====
best_idx = np.argmin(test_rmses)
print(best_idx)
print(test_rmses[best_idx])
print(test_r2[best_idx])
print(model_coefs[best_idx])   # <-- EXACT FORMAT you wanted
print(f"AVG RMSE: {np.mean(test_rmses)}")
print(f"AVG TRAIN CV RMSE: {np.mean(train_cv_rmses):.4f} (+/-{np.std(train_cv_rmses):.4f})")
print(f"AVG TRAIN CV R2  : {np.mean(train_cv_r2s):.4f} (+/-{np.std(train_cv_r2s):.4f})")
print(f"AVG TEST RMSE: {np.mean(test_rmses):.4f} (+/-{np.std(test_rmses):.4f})")
print(f"AVG TEST R2  : {np.mean(test_r2):.4f} (+/-{np.std(test_r2):.4f})")
print(f"AVG Adj Test R2: {np.mean(test_r2_adj):.4f}")
print(f"AVG AIC:        {np.mean(test_aic):.4f}")
print(f"AVG BIC:        {np.mean(test_bic):.4f}")

plt.plot(test_rmses, '-')
plt.title("Test RMSE per iteration")
plt.show()


# ===================================================================================
# Outliers
# ===================================================================================
print("="*100)
print("************* Running Feature Selection on Outlier CLipping Dataset")
print("="*100 +"\n\n")
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

print("\n*************** STD Scaler *************")
rmse_bfs_std = []
r2_bfs_std = []
best_bfs_std = []
rmse_ffs_std = []

for i in range(6, 30):
    k_fold = KFold(n_splits=5, shuffle=True, random_state=0)
    cv_rmse_list = []

    for train_idx, val_idx in k_fold.split(X_train):
        X_tr = X_train.iloc[train_idx]
        y_tr = y_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        y_val = y_train.iloc[val_idx]

        # <-- feature selection happens ONLY on X_tr / y_tr
        features_rfe = select_features_rfe(X_tr, y_tr, n_features=i, scaler_X=std_scaler_X, scaler_y=std_scaler_y)

        # transform X_tr and X_val using ONLY those features
        X_tr2 = X_tr[features_rfe]
        X_val2 = X_val[features_rfe]

        X_tr2 = std_scaler_X.fit_transform(X_tr2)
        X_val2 = std_scaler_X.transform(X_val2)
        y_tr2 = std_scaler_y.fit_transform(y_tr.values.reshape(-1, 1))

        X_tr2 = sm.add_constant(X_tr2, has_constant='add')
        X_val2 = sm.add_constant(X_val2, has_constant='add')

        # train model on fold
        model = sm.OLS(y_tr2, X_tr2).fit()
        y_pred = model.predict(X_val2)
        y_pred = std_scaler_y.inverse_transform(y_pred.reshape(-1, 1))

        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        cv_rmse_list.append(rmse)
        bfs_params.append(model.params)
        best_bfs_std.append(features_rfe)

    # average CV rmse for this k
    rmse_k = np.mean(cv_rmse_list)
    rmse_bfs_std.append(rmse_k)
    print(f"k={i} average rmse = {rmse_k}")

rmses = []
r2s= []
model_coefs = []

best_features = ['accommodates', 'bathrooms', 'cleaning_fee', 'review_scores_rating',
       'bedrooms', 'beds', 'elevator', 'neighbourhood_avg_price',
       'room_type_Private room', 'room_type_Shared room']

print(f"SELECTED FEATURES: {best_features}\n\n")

# ======================================================================================
# Model C Training - Outlier Clipping
# ======================================================================================
print("="*100)
print("\t\t******************* RUNNING TRAINING ON 100 ITERATIONS ********************")
print("="*100 + '\n\n')

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


# =====================================================================================================
# Model D Training - BFS with Feature interaction and polynomial
# =====================================================================================================
print("\n\n" + '='*100 + '\n\n')
print("************************ MODEL D TRAINING ************************")
print("\n\n" + '='*100 + '\n\n')
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
