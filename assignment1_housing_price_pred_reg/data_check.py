import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFE, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import matplotlib.pyplot as plt



# ======================================================
# 1. FEATURE ENGINEERING
# ======================================================
def group_property_types(df):
    df = df.copy()
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

def feature_engineering(df):
    df = df.copy()

    # -----------------------------
    # Target transformation
    # -----------------------------
    df['price_log'] = np.log1p(df['price'])

    # -----------------------------
    # Capacity features (log + caps)
    # -----------------------------
    df['log_accommodates'] = np.log1p(df['accommodates'])
    df['log_beds'] = np.log1p(df['beds'])
    df['log_bedrooms'] = np.log1p(df['bedrooms'])
    df['log_bathrooms'] = np.log1p(df['bathrooms'])

    df['bedrooms_capped'] = np.minimum(df['bedrooms'], 5)
    df['bathrooms_capped'] = np.minimum(df['bathrooms'], 4)

    # -----------------------------
    # Ratio / efficiency features
    # -----------------------------
    # df['price_per_guest'] = df['price'] / (df['accommodates'] + 1)
    df['beds_per_guest'] = df['beds'] / (df['accommodates'] + 1)
    df['bathrooms_per_bed'] = df['bathrooms'] / (df['beds'] + 1)

    # -----------------------------
    # Review features
    # -----------------------------
    df['number_of_reviews'] = df['number_of_reviews'].fillna(0)
    df['review_scores_rating'] = df['review_scores_rating'].fillna(0)

    df['log_reviews'] = np.log1p(df['number_of_reviews'])
    df['rating_centered'] = df['review_scores_rating'] - 90
    df['rating_strength'] = df['review_scores_rating'] * df['log_reviews']
    df['is_highly_rated'] = (df['review_scores_rating'] >= 95).astype(int)

    # -----------------------------
    # Host experience
    # -----------------------------
    df['host_since'] = pd.to_datetime(df['host_since'], errors='coerce')
    df['host_years'] = (
        (df['host_since'].max() - df['host_since']).dt.days / 365
    )

    df['log_host_years'] = np.log1p(df['host_years'])

    df['host_exp'] = pd.cut(
        df['host_years'],
        bins=[-np.inf, 1, 3, 6, 10, np.inf],
        labels=['new', 'early', 'mid', 'experienced', 'veteran']
    ).astype(str)

    df['host_response_rate'] = df['host_response_rate'].astype(str).str.replace('%', "")
    df['host_response_rate'] = pd.to_numeric(df['host_response_rate'], errors='coerce').fillna(0)

    # -----------------------------
    # Boolean cleanup
    # -----------------------------
    bool_cols = ['host_has_profile_pic', 'host_identity_verified', 'instant_bookable']
    for col in bool_cols:
        df[col] = (df[col] == 't').astype(int)

    # -----------------------------
    # Drop unused raw columns
    # -----------------------------
    drop_cols = [
        'price',
        'host_since',
        'number_of_reviews',
        'review_scores_rating',
        'neighbourhood',
        'zipcode',
        'first_review',
        'last_review'
    ]

    df.drop(columns=drop_cols, inplace=True, errors='ignore')

    return df


# ======================================================
# 2. TRAIN / TEST PREPROCESSING
# ======================================================

def preprocess_dataset(df, target_col='price_log'):
    df = group_property_types(df)
    df = feature_engineering(df)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -----------------------------
    # Identify column types
    # -----------------------------
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns
    num_cols = X_train.select_dtypes(include=[np.number]).columns

    for col in cat_cols:
        X_train[col] = X_train[col].astype(str)
        X_test[col] = X_test[col].astype(str)

    # -----------------------------
    # Imputation
    # -----------------------------
    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')

    X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
    X_test[num_cols] = num_imputer.transform(X_test[num_cols])

    X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
    X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])

    # -----------------------------
    # Scaling
    # -----------------------------
    # scaler = RobustScaler()
    # X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    # X_test[num_cols] = scaler.transform(X_test[num_cols])

    # -----------------------------
    # One-hot encoding
    # -----------------------------
    encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')

    X_train_cat = encoder.fit_transform(X_train[cat_cols])
    X_test_cat = encoder.transform(X_test[cat_cols])

    cat_feature_names = encoder.get_feature_names_out(cat_cols)

    X_train_cat = pd.DataFrame(X_train_cat, columns=cat_feature_names, index=X_train.index)
    X_test_cat = pd.DataFrame(X_test_cat, columns=cat_feature_names, index=X_test.index)

    X_train_final = pd.concat([X_train[num_cols], X_train_cat], axis=1)
    X_test_final = pd.concat([X_test[num_cols], X_test_cat], axis=1)

    X_train_final.to_csv('data/train_features.csv', index=False)
    X_test_final.to_csv('data/test_features.csv', index=False)
    y_train.to_csv('data/train_label.csv', index=False)
    y_test.to_csv('data/test_label.csv', index=False)

    return X_train_final, X_test_final, y_train, y_test


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

        # ---- feature selection INSIDE fold ----
        if method == "rfe":
            feats = select_features_rfe(X_tr, y_tr, k_features)
        elif method == "ffs":
            feats = select_features_ffs(X_tr, y_tr, k_features)
        else:
            raise ValueError("method must be 'rfe' or 'ffs'")

        X_tr_f = sm.add_constant(X_tr[feats], has_constant='add')
        X_val_f = sm.add_constant(X_val[feats], has_constant='add')

        model = sm.OLS(y_tr, X_tr_f).fit()
        y_pred_log = model.predict(X_val_f)

        # ---- keep predictions in a reasonable log range to avoid exp explosion ----
        y_pred_log = np.clip(y_pred_log, 0, y_log_max)
        print("max y_pred_log:", float(np.max(y_pred_log)), "=> price:", float(np.expm1(np.max(y_pred_log))))

        # RMSE in price space
        y_pred_price = np.expm1(y_pred_log)
        y_val_price = np.expm1(y_val)
        rmses_price.append(np.sqrt(mean_squared_error(y_val_price, y_pred_price)))

        # R² must be in log space
        r2s.append(r2_score(y_val, y_pred_log))

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




# ======================================================
# 3. RUN
# ======================================================

if __name__ == "__main__":
    df = pd.read_csv("data/AirBNB.csv")

    X_train, X_test, y_train, y_test = preprocess_dataset(df)

    # ===============================
    # Feature Selection Evaluation
    # ===============================


    results = []
    for k in range(6, 30):
        rmse_rfe, r2_rfe = crossfold_evaluate_fs(X_train, y_train, "rfe", k)
        rmse_ffs, r2_ffs = crossfold_evaluate_fs(X_train, y_train, "ffs", k)

        results.append({
            "n_features": k,
            "rmse_rfe": rmse_rfe, "r2_rfe": r2_rfe,
            "rmse_ffs": rmse_ffs, "r2_ffs": r2_ffs
        })

    results_df = pd.DataFrame(results)
    plot_feature_selection_results(results_df)
    print(results_df.sort_values('rmse_rfe'))
