from sklearn.feature_selection import RFE, f_regression, RFECV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 1000)

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

# ====================================================================================
# Impute Missing Data
# ====================================================================================
def impute_nan_columns(X_train, X_test):
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns
    num_cols = X_train.select_dtypes(include=[np.number]).columns

    for col in cat_cols:
        X_train[col] = X_train[col].astype(str)
        X_test[col] = X_test[col].astype(str)

    cat_imputer = SimpleImputer(strategy='most_frequent')
    num_imputer = SimpleImputer(strategy='median')

    X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
    X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])

    X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
    X_test[num_cols] = num_imputer.transform(X_test[num_cols])

    return X_train, X_test, cat_imputer, num_imputer

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
    return df, property_mapping

def compute_and_bin_host_experience(df):
    df['host_since'] = pd.to_datetime(df['host_since'], errors='coerce')
    df['host_years'] = (
            (df['host_since'].max() - df['host_since']).dt.days / 365
    )
    df['host_exp'] = pd.cut(df['host_years'],
                                bins=[0, 2, 5, 7, 50],
                                labels=['new', 'mid', 'experienced', 'veteran'])
    del df['host_since']
    return df

def convert_host_response_rate_to_int(df):
    df['host_response_rate'] = df['host_response_rate'].astype(str).str.replace('%', "")
    df['host_response_rate'] = pd.to_numeric(df['host_response_rate'], errors='coerce').fillna(0)
    return df


# ===========================================================================
# Compute average price for neighbourhoods, so drop zipcode
# ===========================================================================
def compute_median_price_per_neighbourhood(df):
    median_price_per_neigh = df.groupby('neighbourhood')['price'].median()
    df['neighbourhood_median_price'] = df['neighbourhood'].map(median_price_per_neigh)
    del df['neighbourhood']
    del df['zipcode']
    return df, median_price_per_neigh

# Convert bool columns' values from bool to int
def convert_col_bool_to_int(df):
    df['cleaning_fee'] = df['cleaning_fee'].astype(int)

    bool_columns = ['host_has_profile_pic', 'host_identity_verified',
               'instant_bookable']
    for col in bool_columns:
        df[col] = (df[col] == 't').astype(int)

    df = df.drop(columns=['first_review', 'last_review'], axis=1)
    return df

def create_dummies(X_train, X_test):
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns
    num_cols = X_train.select_dtypes(include=[np.number]).columns

    encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')

    X_train_cat = encoder.fit_transform(X_train[cat_cols])
    X_test_cat = encoder.transform(X_test[cat_cols])

    cat_feature_names = encoder.get_feature_names_out(cat_cols)

    X_train_cat = pd.DataFrame(X_train_cat, columns=cat_feature_names, index=X_train.index)
    X_test_cat = pd.DataFrame(X_test_cat, columns=cat_feature_names, index=X_test.index)

    X_train_final = pd.concat([X_train[num_cols], X_train_cat], axis=1)
    X_test_final = pd.concat([X_test[num_cols], X_test_cat], axis=1)

    return X_train_final, X_test_final, encoder, cat_cols, num_cols


# ============================================================================
# CREATE POLYNOMIAL AND INTERACTION FEATURES
# ============================================================================
def create_feature_interactions(df):
    # Polynomial features for key continuous variables
    df['accommodates_squared'] = df['accommodates'] ** 2
    df['bedrooms_squared'] = df['bedrooms'] ** 2
    df['bathrooms_squared'] = df['bathrooms'] ** 2
    df['beds_squared'] = df['beds'] ** 2

    # Key interactions - size and capacity
    df['accommodates_bedrooms'] = df['accommodates'] * df['bedrooms']
    df['accommodates_bathrooms'] = df['accommodates'] * df['bathrooms']
    df['bedrooms_bathrooms'] = df['bedrooms'] * df['bathrooms']
    df['accommodates_beds'] = df['accommodates'] * df['beds']

    # City-specific pricing patterns
    df['LA_neighbourhood_price'] = df['city_LA'] * df['neighbourhood_median_price']
    df['SF_neighbourhood_price'] = df['city_SF'] * df['neighbourhood_median_price']
    df['NYC_neighbourhood_price'] = df['city_NYC'] * df['neighbourhood_median_price']
    df['Chicago_neighbourhood_price'] = df['city_Chicago'] * df['neighbourhood_median_price']

    # City-specific capacity pricing
    df['LA_accommodates'] = df['city_LA'] * df['accommodates']
    df['SF_accommodates'] = df['city_SF'] * df['accommodates']
    df['NYC_accommodates'] = df['city_NYC'] * df['accommodates']
    df['Chicago_accommodates'] = df['city_Chicago'] * df['accommodates']

    # Room type interactions
    df['private_room_bedrooms'] = df['room_type_Private room'] * df['bedrooms']
    df['private_room_accommodates'] = df['room_type_Private room'] * df['accommodates']
    df['shared_room_accommodates'] = df['room_type_Shared room'] * df['accommodates']

    # Property type interactions
    df['house_bedrooms'] = df['property_type_grouped_House'] * df['bedrooms']
    df['house_accommodates'] = df['property_type_grouped_House'] * df['accommodates']

    # Reviews and ratings interactions
    df['reviews_rating'] = df['number_of_reviews'] * df['review_scores_rating']
    df['reviews_per_accommodates'] = df['number_of_reviews'] / (df['accommodates'] + 1)
    df['has_reviews'] = (df['number_of_reviews'] > 0).astype(int)

    # Cancellation policy interactions
    df['strict_policy_accommodates'] = (df['cancellation_policy_strict'] +
                                       df['cancellation_policy_super_strict_60']) * df['accommodates']

    # Cleaning fee per person
    df['cleaning_per_accommodates'] = df['cleaning_fee'] / (df['accommodates'] + 1)

    # Host quality indicators
    df['verified_response_rate'] = df['host_identity_verified'] * df['host_response_rate']

    # Luxury indicators
    df['luxury_indicator'] = (df['bedrooms'] >= 3).astype(int) * (df['bathrooms'] >= 2).astype(int) * df[
        'neighbourhood_median_price']

    df.to_csv('data/feature_interactions.csv', index=False)
    return df


def preprocess_dataset(df, target_col="price"):
    df, property_mapping = group_property_types(df)
    df = compute_and_bin_host_experience(df)
    df = convert_host_response_rate_to_int(df)
    df = convert_col_bool_to_int(df)
    df, median_price_per_neigh = compute_median_price_per_neighbourhood(df)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2
    )

    X_train, X_test, cat_imputer, num_imputer = impute_nan_columns(X_train, X_test)
    X_train, X_test, encoder, cat_cols, num_cols = create_dummies(X_train, X_test)

    X_train = create_feature_interactions(X_train)
    X_test = create_feature_interactions(X_test)

    X_train.to_csv('data/train_features.csv', index=False)
    X_test.to_csv('data/test_features.csv', index=False)
    y_train.to_csv('data/train_label.csv', index=False)
    y_test.to_csv('data/test_label.csv', index=False)

    return (X_train, X_test, y_train, y_test,
            cat_imputer, num_imputer, encoder, property_mapping, median_price_per_neigh,
            cat_cols, num_cols)

# ==============================================================================================
# ============= FEATURE SELECTION =====================
# ==============================================================================================
def select_features_rfe(X, y, n_features):
    rfe = RFE(LinearRegression(), n_features_to_select=n_features)
    rfe.fit(X, y)
    return X.columns[rfe.support_]

def select_features_ffs(X, y, n_features):
    f_vals, _ = f_regression(X, y)
    scores = pd.Series(f_vals, index=X.columns)
    return scores.sort_values(ascending=False).head(n_features).index

def crossfold_evaluate_fs(X, y_log, method, k_features, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    rmses_price, r2s = [], []
    y_log_max = float(np.log1p(2000))  # or np.max(y_log)

    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y_log.iloc[train_idx], y_log.iloc[val_idx]

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        # ---- feature selection INSIDE fold ----
        if method == "rfe":
            feats = select_features_rfe(X_tr, y_tr, k_features)
        elif method == "ffs":
            feats = select_features_ffs(X_tr, y_tr, k_features)
        else:
            raise ValueError("method must be 'rfe' or 'ffs'")

        X_tr_scaled = scaler_X.fit_transform(X_tr[feats])
        X_val_scaled = scaler_X.transform(X_val[feats])
        y_tr_scaled = scaler_y.fit_transform(np.array(y_tr).reshape(-1, 1))

        X_tr_f = sm.add_constant(X_tr_scaled, has_constant='add')
        X_val_f = sm.add_constant(X_val_scaled, has_constant='add')

        model = sm.OLS(y_tr_scaled, X_tr_f).fit()
        y_pred = model.predict(X_val_f)

        y_pred_orig = scaler_y.inverse_transform(np.array(y_pred).reshape(-1, 1)).ravel()
        rmses_price.append(np.sqrt(mean_squared_error(y_val, y_pred_orig)))
        r2s.append(r2_score(y_val, y_pred_orig))

    return float(np.mean(rmses_price)), float(np.mean(r2s))


def plot_feature_selection_results(results_df):
    plt.figure(figsize=(14, 6))

    # ---------- RMSE ----------
    plt.subplot(1, 2, 1)
    plt.plot(results_df['n_features'], results_df['rmse_rfe'],
             marker='o', label='RFE')
    plt.plot(results_df['n_features'], results_df['rmse_ffs'],
             marker='s', label='FFS')
    plt.xlabel('Number of Features')
    plt.ylabel('RMSE')
    plt.title('RMSE vs Number of Features')
    plt.legend()
    plt.grid(alpha=0.3)

    # ---------- R² ----------
    plt.subplot(1, 2, 2)
    plt.plot(results_df['n_features'], results_df['r2_rfe'],
             marker='o', label='RFE')
    plt.plot(results_df['n_features'], results_df['r2_ffs'],
             marker='s', label='FFS')
    plt.xlabel('Number of Features')
    plt.ylabel('R²')
    plt.title('R² vs Number of Features')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def evaluate_feature_set(X_train, y_train, X_test, y_test, features, name):
    """
    Evaluate a feature set using cross-validation and test set
    Returns dict with all metrics
    """
    X_train_sel = X_train[features]
    X_test_sel = X_test[features]

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train_sel)
    X_test_scaled = scaler_X.transform(X_test_sel)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

    # Cross-validation
    k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_rmses = []
    cv_r2s = []

    for train_idx, val_idx in k_fold.split(X_train_scaled):
        X_tr = sm.add_constant(X_train_scaled[train_idx], has_constant='add')
        X_val = sm.add_constant(X_train_scaled[val_idx], has_constant='add')
        y_tr = y_train_scaled[train_idx]
        y_val = y_train_scaled[val_idx]

        model = sm.OLS(y_tr, X_tr).fit()
        y_pred_scaled = model.predict(X_val)

        y_pred_orig = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        y_val_orig = scaler_y.inverse_transform(y_val.reshape(-1, 1)).ravel()

        cv_r2s.append(r2_score(y_val_orig, y_pred_orig))
        cv_rmses.append(np.sqrt(mean_squared_error(y_val_orig, y_pred_orig)))

    # Final model on full training set
    X_train_scaled = sm.add_constant(X_train_scaled, has_constant='add')
    X_test_scaled = sm.add_constant(X_test_scaled, has_constant='add')

    final_model = sm.OLS(y_train_scaled, X_train_scaled).fit()
    y_pred_scaled = final_model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_r2 = r2_score(y_test, y_pred)

    return {
        'name': name,
        'features': features,
        'n_features': len(features),
        'cv_rmse_mean': np.mean(cv_rmses),
        'cv_rmse_std': np.std(cv_rmses),
        'cv_r2_mean': np.mean(cv_r2s),
        'cv_r2_std': np.std(cv_r2s),
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'aic': final_model.aic,
        'bic': final_model.bic,
        'adj_r2': final_model.rsquared_adj,
        'model': final_model,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y
    }


if __name__ == "__main__":
    df = pd.read_csv("data/AirBNB.csv")

    X = df.drop(columns=['price'], axis=1)
    y = df['price']

    (X_train, X_test, y_train, y_test,
     cat_imputer, num_imputer, encoder,
     property_mapping, median_price_per_neigh,
     cat_cols, num_cols) = preprocess_dataset(df)

    final_selected_features = [
      "accommodates",
      "bathrooms",
      "cleaning_fee",
      "host_identity_verified",
      "number_of_reviews",
      "review_scores_rating",
      "bedrooms",
      "beds",
      "neighbourhood_median_price",
      "property_type_grouped_Other",
      "room_type_Private room",
      "room_type_Shared room",
      "cancellation_policy_moderate",
      "cancellation_policy_super_strict_60",
      "city_LA",
      "city_NYC",
      "city_SF",
      "accommodates_squared",
      "bedrooms_squared",
      "bathrooms_squared",
      "beds_squared",
      "accommodates_bedrooms",
      "accommodates_bathrooms",
      "bedrooms_bathrooms",
      "accommodates_beds",
      "LA_neighbourhood_price",
      "SF_neighbourhood_price",
      "NYC_neighbourhood_price",
      "LA_accommodates",
      "SF_accommodates",
      "NYC_accommodates",
      "Chicago_accommodates",
      "private_room_bedrooms",
      "shared_room_accommodates",
      "house_bedrooms",
      "reviews_rating",
      "reviews_per_accommodates",
      "has_reviews",
      "cleaning_per_accommodates",
      "verified_response_rate",
      "luxury_indicator"
    ]

    result = evaluate_feature_set(X_train, y_train, X_test, y_test,
                                  final_selected_features, f"RFE (n=)")
    results = []
    results.append(result)
    print(
        f"RFE n=: CV RMSE={result['cv_rmse_mean']:.2f}, "
        f"Test RMSE={result['test_rmse']:.2f}, R2={result['test_r2']:.4f}")

    X_final = pd.concat([X_train, X_test], axis=0)
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_final[final_selected_features])

    X_scaled = sm.add_constant(X_scaled)

    model = sm.OLS(y, X_scaled).fit()

    with open("model_resources.pkl", "wb") as f:
        pickle.dump({
            'model_params': model.params,
            'scaler_X': scaler_X,
            'cat_imputer': cat_imputer,
            'num_imputer': num_imputer,
            'encoder': encoder,
            'property_mapping': property_mapping,
            'median_price_per_neigh': median_price_per_neigh,
            'selected_features': final_selected_features,
            'cat_cols': cat_cols,
            'num_cols': num_cols
        }, f)


    # ============================================================================================================
    # RUN THIS CODE FOR ANALYSIS
    # ============================================================================================================

    # base_features = ['accommodates', 'bathrooms', 'cleaning_fee', 'host_identity_verified',
    #                  'host_response_rate', 'number_of_reviews', 'review_scores_rating',
    #                  'bedrooms', 'beds', 'neighbourhood_median_price',
    #                  'property_type_grouped_House', 'property_type_grouped_Other',
    #                  'room_type_Private room', 'room_type_Shared room', 'bed_type_Real Bed',
    #                  'cancellation_policy_moderate', 'cancellation_policy_strict',
    #                  'cancellation_policy_super_strict_60', 'city_Chicago', 'city_LA',
    #                  'city_NYC', 'city_SF']
    #
    # all_features = base_features + [
    #     'accommodates_squared', 'bedrooms_squared', 'bathrooms_squared', 'beds_squared',
    #     'accommodates_bedrooms', 'accommodates_bathrooms', 'bedrooms_bathrooms', 'accommodates_beds',
    #     'LA_neighbourhood_price', 'SF_neighbourhood_price', 'NYC_neighbourhood_price', 'Chicago_neighbourhood_price',
    #     'LA_accommodates', 'SF_accommodates', 'NYC_accommodates', 'Chicago_accommodates',
    #     'private_room_bedrooms', 'private_room_accommodates', 'shared_room_accommodates',
    #     'house_bedrooms', 'house_accommodates',
    #     'reviews_rating', 'reviews_per_accommodates', 'has_reviews',
    #     'strict_policy_accommodates', 'cleaning_per_accommodates',
    #     'verified_response_rate', 'luxury_indicator'
    # ]
    #
    # print("=" * 100)
    # print("FEATURE SELECTION COMPARISON")
    # print("=" * 100)
    #
    # # Prepare data for feature selection
    # X_for_selection = X_train[all_features]
    # std_scaler_X = StandardScaler()
    # X_scaled_temp = std_scaler_X.fit_transform(X_for_selection)
    #
    # # ============================================================================
    # # 1. F-REGRESSION FEATURE SELECTION (Top K features)
    # # ============================================================================
    # print("\n1. Running F-Regression Feature Selection...")
    # f_scores, p_values = f_regression(X_scaled_temp, y_train)
    #
    # f_reg_df = pd.DataFrame({
    #     'feature': all_features,
    #     'f_score': f_scores,
    #     'p_value': p_values
    # }).sort_values('f_score', ascending=False)
    #
    # print("\nTop 20 features by F-score:")
    # print(f_reg_df.head(20))
    #
    # # Test different numbers of features
    # f_reg_results = []
    # for n in [10, 15, 20, 25, 30, 35, 40]:
    #     features = f_reg_df.head(n)['feature'].tolist()
    #     result = evaluate_feature_set(X_train, y_train, X_test, y_test,
    #                                   features, f"F-Regression (top {n})")
    #     f_reg_results.append(result)
    #     print(
    #         f"F-Regression top {n}: CV RMSE={result['cv_rmse_mean']:.2f}, Test RMSE={result['test_rmse']:.2f}, R²={result['test_r2']:.4f}")
    #
    # # ============================================================================
    # # 2. RFE FEATURE SELECTION
    # # ============================================================================
    # print("\n2. Running RFE Feature Selection...")
    #
    # # Test different numbers of features
    # rfe_results = []
    # for n in [10, 15, 20, 25, 30, 35, 40]:
    #     rfe = RFE(estimator=LinearRegression(), n_features_to_select=n)
    #     rfe.fit(X_scaled_temp, y_train)
    #     features = [all_features[i] for i in range(len(all_features)) if rfe.support_[i]]
    #
    #     result = evaluate_feature_set(X_train, y_train, X_test, y_test,
    #                                   features, f"RFE (n={n})")
    #     rfe_results.append(result)
    #     print(
    #         f"RFE n={n}: CV RMSE={result['cv_rmse_mean']:.2f}, Test RMSE={result['test_rmse']:.2f}, R²={result['test_r2']:.4f}")
    #
    # # ============================================================================
    # # 3. RFECV (AUTOMATIC SELECTION)
    # # ============================================================================
    # print("\n3. Running RFECV (Automatic Selection)...")
    # rfecv = RFECV(estimator=LinearRegression(), cv=5, scoring='neg_mean_squared_error',
    #               min_features_to_select=10)
    # rfecv.fit(X_scaled_temp, y_train)
    #
    # rfecv_features = [all_features[i] for i in range(len(all_features)) if rfecv.support_[i]]
    # rfecv_result = evaluate_feature_set(X_train, y_train, X_test, y_test,
    #                                     rfecv_features, f"RFECV (optimal={len(rfecv_features)})")
    # print(f"RFECV optimal features: {len(rfecv_features)}")
    # print(
    #     f"CV RMSE={rfecv_result['cv_rmse_mean']:.2f}, Test RMSE={rfecv_result['test_rmse']:.2f}, R²={rfecv_result['test_r2']:.4f}")
    #
    # # ============================================================================
    # # 4. COMBINED APPROACH (Intersection of F-Reg and RFE)
    # # ============================================================================
    # print("\n4. Testing Combined Feature Sets (Intersection & Union)...")
    #
    # combined_results = []
    # for n in [20, 25, 30, 35]:
    #     # Get top features from both methods
    #     f_reg_top = set(f_reg_df.head(n)['feature'].tolist())
    #
    #     rfe = RFE(estimator=LinearRegression(), n_features_to_select=n)
    #     rfe.fit(X_scaled_temp, y_train)
    #     rfe_top = set([all_features[i] for i in range(len(all_features)) if rfe.support_[i]])
    #
    #     # Intersection (features selected by both)
    #     intersection = list(f_reg_top & rfe_top)
    #     if len(intersection) >= 10:
    #         result = evaluate_feature_set(X_train, y_train, X_test, y_test,
    #                                       intersection, f"Intersection (n={len(intersection)})")
    #         combined_results.append(result)
    #         print(
    #             f"Intersection ({len(intersection)} features): CV RMSE={result['cv_rmse_mean']:.2f}, Test RMSE={result['test_rmse']:.2f}, R²={result['test_r2']:.4f}")
    #
    #     # Union (features selected by either)
    #     union = list(f_reg_top | rfe_top)
    #     if len(union) <= 45:
    #         result = evaluate_feature_set(X_train, y_train, X_test, y_test,
    #                                       union, f"Union (n={len(union)})")
    #         combined_results.append(result)
    #         print(
    #             f"Union ({len(union)} features): CV RMSE={result['cv_rmse_mean']:.2f}, Test RMSE={result['test_rmse']:.2f}, R²={result['test_r2']:.4f}")
    #
    # # ============================================================================
    # # 5. COMPARE ALL METHODS
    # # ============================================================================
    # print("\n" + "=" * 100)
    # print("COMPARISON SUMMARY")
    # print("=" * 100)
    #
    # all_results = f_reg_results + rfe_results + [rfecv_result] + combined_results
    #
    # comparison_df = pd.DataFrame([{
    #     'Method': r['name'],
    #     'N Features': r['n_features'],
    #     'CV RMSE': r['cv_rmse_mean'],
    #     'CV R²': r['cv_r2_mean'],
    #     'Test RMSE': r['test_rmse'],
    #     'Test R²': r['test_r2'],
    #     'Adj R²': r['adj_r2'],
    #     'AIC': r['aic'],
    #     'BIC': r['bic']
    # } for r in all_results])
    #
    # comparison_df = comparison_df.sort_values('Test RMSE')
    # print("\nAll Methods Ranked by Test RMSE:")
    # print(comparison_df.to_string(index=False))
    #
    #
    # # ============================================================================
    # # 6. SAVE BEST MODEL
    # # ============================================================================
    # best_result = all_results[comparison_df.index[0]]
    #
    # print("\n" + "=" * 100)
    # print(f"BEST MODEL: {best_result['name']}")
    # print("=" * 100)
    # print(f"Number of features: {best_result['n_features']}")
    # print(f"CV RMSE: {best_result['cv_rmse_mean']:.4f} (+/- {best_result['cv_rmse_std']:.4f})")
    # print(f"CV R²: {best_result['cv_r2_mean']:.4f} (+/- {best_result['cv_r2_std']:.4f})")
    # print(f"Test RMSE: {best_result['test_rmse']:.4f}")
    # print(f"Test R²: {best_result['test_r2']:.4f}")
    # print(f"Adjusted R²: {best_result['adj_r2']:.4f}")
    # print(f"AIC: {best_result['aic']:.4f}")
    # print(f"BIC: {best_result['bic']:.4f}")
    # print(f"\nSelected features: {best_result['features']}")
    #
    #
    #
    # # ============================================================================
    # # 7. VISUALIZATION
    # # ============================================================================
    # fig = plt.figure(figsize=(20, 12))
    #
    # # Plot 1: Test RMSE comparison
    # ax1 = plt.subplot(2, 3, 1)
    # methods = comparison_df['Method']
    # x_pos = np.arange(len(methods))
    # colors = ['red' if 'F-Regression' in m else 'blue' if 'RFE' in m else 'green' for m in methods]
    # ax1.barh(x_pos, comparison_df['Test RMSE'], color=colors, alpha=0.7)
    # ax1.set_yticks(x_pos)
    # ax1.set_yticklabels(methods, fontsize=8)
    # ax1.set_xlabel('Test RMSE', fontsize=10)
    # ax1.set_title('Test RMSE by Method', fontsize=12)
    # ax1.invert_yaxis()
    # ax1.grid(alpha=0.3, axis='x')
    #
    # # Plot 2: Test R² comparison
    # ax2 = plt.subplot(2, 3, 2)
    # ax2.barh(x_pos, comparison_df['Test R²'], color=colors, alpha=0.7)
    # ax2.set_yticks(x_pos)
    # ax2.set_yticklabels(methods, fontsize=8)
    # ax2.set_xlabel('Test R²', fontsize=10)
    # ax2.set_title('Test R² by Method', fontsize=12)
    # ax2.invert_yaxis()
    # ax2.grid(alpha=0.3, axis='x')
    #
    # # Plot 3: Number of features vs RMSE
    # ax3 = plt.subplot(2, 3, 3)
    # f_reg_df_plot = comparison_df[comparison_df['Method'].str.contains('F-Regression')]
    # rfe_df_plot = comparison_df[
    #     comparison_df['Method'].str.contains('RFE') & ~comparison_df['Method'].str.contains('RFECV')]
    # combined_df_plot = comparison_df[comparison_df['Method'].str.contains('Intersection|Union')]
    #
    # ax3.scatter(f_reg_df_plot['N Features'], f_reg_df_plot['Test RMSE'],
    #             c='red', s=100, alpha=0.6, label='F-Regression', marker='o')
    # ax3.scatter(rfe_df_plot['N Features'], rfe_df_plot['Test RMSE'],
    #             c='blue', s=100, alpha=0.6, label='RFE', marker='s')
    # ax3.scatter(combined_df_plot['N Features'], combined_df_plot['Test RMSE'],
    #             c='green', s=100, alpha=0.6, label='Combined', marker='^')
    # ax3.set_xlabel('Number of Features', fontsize=10)
    # ax3.set_ylabel('Test RMSE', fontsize=10)
    # ax3.set_title('Features vs RMSE', fontsize=12)
    # ax3.legend()
    # ax3.grid(alpha=0.3)
    #
    # # Plot 4: CV RMSE vs Test RMSE
    # ax4 = plt.subplot(2, 3, 4)
    # ax4.scatter(comparison_df['CV RMSE'], comparison_df['Test RMSE'],
    #             c=colors, s=100, alpha=0.7)
    # lims = [min(comparison_df['CV RMSE'].min(), comparison_df['Test RMSE'].min()),
    #         max(comparison_df['CV RMSE'].max(), comparison_df['Test RMSE'].max())]
    # ax4.plot(lims, lims, 'k--', alpha=0.5, label='Perfect Agreement')
    # ax4.set_xlabel('CV RMSE', fontsize=10)
    # ax4.set_ylabel('Test RMSE', fontsize=10)
    # ax4.set_title('CV vs Test RMSE', fontsize=12)
    # ax4.legend()
    # ax4.grid(alpha=0.3)
    #
    # # Plot 5: AIC/BIC comparison
    # ax5 = plt.subplot(2, 3, 5)
    # x = np.arange(len(methods))
    # width = 0.35
    # ax5.bar(x - width / 2, comparison_df['AIC'], width, label='AIC', alpha=0.7)
    # ax5.bar(x + width / 2, comparison_df['BIC'], width, label='BIC', alpha=0.7)
    # ax5.set_xticks(x)
    # ax5.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
    # ax5.set_ylabel('Information Criterion', fontsize=10)
    # ax5.set_title('AIC/BIC Comparison', fontsize=12)
    # ax5.legend()
    # ax5.grid(alpha=0.3, axis='y')
    #
    # # Plot 6: Feature importance for best model
    # ax6 = plt.subplot(2, 3, 6)
    # coefs_array = best_result['model'].params[1:]  # Skip first element (const)
    # coef_names = best_result['features']
    # coefs = pd.Series(coefs_array, index=coef_names)
    #
    # top_15 = abs(coefs).nlargest(15)
    # colors_imp = ['green' if coefs[f] > 0 else 'red' for f in top_15.index]
    # ax6.barh(range(len(top_15)), [coefs[f] for f in top_15.index], color=colors_imp, alpha=0.7)
    # ax6.set_yticks(range(len(top_15)))
    # ax6.set_yticklabels(top_15.index, fontsize=8)
    # ax6.set_xlabel('Coefficient Value', fontsize=10)
    # ax6.set_title(f'Top 15 Features - {best_result["name"]}', fontsize=12)
    # ax6.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    # ax6.invert_yaxis()
    # ax6.grid(alpha=0.3, axis='x')
    #
    # plt.tight_layout()
    # plt.show()


