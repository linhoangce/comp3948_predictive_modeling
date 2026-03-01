import json
import numpy as np
import pandas as pd

COEFS = json.load(open("best_coefs.json"))
SCALER = json.load(open("scaler_stats.json"))
FEATURE_ORDER = json.load(open("feature_order.json"))

def predict_price(features: dict) -> float:
    """
    predict using trained coeffs and scaler stats
    """
    pred_std = COEFS["const"]

    for name in FEATURE_ORDER:
        beta = COEFS.get(name, 0)
        x     = features.get(name, 0)
        mu    = SCALER["X_mean"].get(name, 0)
        scale = SCALER["X_scale"].get(name, 1)
        x_std = (x - mu) / scale
        pred_std += beta * x_std

    pred = pred_std * SCALER["y_scale"] + SCALER["y_mean"]
    return float(pred)


def process_data(file_path):
    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 1000)

    df = pd.read_csv(file_path, low_memory=False)

    na_df = df.isna().sum().reset_index()
    na_df.columns = ["Column", "Missing"]
    na_df.sort_values(['Missing'], ascending=False, inplace=True)

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
        if col.isnumeric():
            df = impute_nan(col, df, "mean")
        else:
            df = impute_nan(col, df, "mode")

    print("=" * 100)

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

    df_new['host_years'] = (df_new['host_since'].max() - df_new['host_since']).dt.days / 365
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
    mean_price_per_neigh = df.groupby('neighbourhood')['price'].mean()
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

    return df_new


def predict_and_save(file_path, output_path='predictions.csv'):
    """
    Process data, make predictions, and save results to CSV
    """
    df_processed = process_data(file_path)

    # Make predictions for each row
    predictions = []
    for idx, row in df_processed.iterrows():
        features_dict = row.to_dict()
        pred = predict_price(features_dict)
        predictions.append(pred)

    pd.DataFrame({'Price': predictions}).to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

    return df_processed


df_with_predictions = predict_and_save('data/AirBNB1.csv', 'predictions.csv')