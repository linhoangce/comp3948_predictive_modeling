import json
import numpy as np
import pandas as pd
import pickle

def process_data(df, cat_imputer, num_imputer, encoder,
                     property_mapping, median_price_per_neigh,
                 cat_cols, num_cols):
    # =============== IMPUTING =======================================
    # cat_cols = df.select_dtypes(include=['object', 'category']).columns
    # num_cols = df.select_dtypes(include=[np.number]).columns
    # for col in cat_cols:
    #     df[col] = df[col].astype(str)

    # ================ MAPPING ==========================================
    df['property_type_grouped'] = df['property_type'].map(property_mapping)
    df['neighbourhood_median_price'] = df['neighbourhood'].map(median_price_per_neigh)

    df['host_since'] = pd.to_datetime(df['host_since'], errors='coerce')
    df['host_years'] = (
            (df['host_since'].max() - df['host_since']).dt.days / 365
    )
    df['host_exp'] = pd.cut(df['host_years'],
                            bins=[0, 2, 5, 7, 50],
                            labels=['new', 'mid', 'experienced', 'veteran'])

    df['host_response_rate'] = df['host_response_rate'].astype(str).str.replace('%', "")
    df['host_response_rate'] = pd.to_numeric(df['host_response_rate'], errors='coerce').fillna(0)


    # =============== CONVERTING BOOL ====================================
    df['cleaning_fee'] = df['cleaning_fee'].astype(int)
    bool_columns = ['host_has_profile_pic', 'host_identity_verified',
                    'instant_bookable']
    for col in bool_columns:
        df[col] = (df[col] == 't').astype(int)

    df[cat_cols] = cat_imputer.transform(df[cat_cols])
    df[num_cols] = num_imputer.transform(df[num_cols])

    # =============== DUMMY ===============================================
    df_encoded = encoder.transform(df[cat_cols])
    cat_feature_names = encoder.get_feature_names_out(cat_cols)

    df_cat = pd.DataFrame(df_encoded, columns=cat_feature_names, index=df.index)
    df = pd.concat([df[num_cols], df_cat], axis=1)

    # ================ FEATURE INTERACTIONS ===============================
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

    df.to_csv('data_processed.csv', index=False)
    return df

def predictions(model_params, df, scaler_X, features):
    X = df[features]
    X = df.reindex(columns=selected_features, fill_value=0)
    X_scaled = scaler_X.transform(X)

    X_scaled = np.column_stack([np.ones(len(X_scaled)), X_scaled])
    params = np.array(model_params)
    y_pred = X_scaled @ params

    return y_pred


if __name__ == "__main__":
    df = pd.read_csv('data/AirBNB1.csv')

    with open('model_resources.pkl', 'rb') as f:
        model_resources = pickle.load(f)

    model_params = model_resources['model_params']
    scaler_X = model_resources['scaler_X']
    cat_imputer = model_resources['cat_imputer']
    num_imputer = model_resources['num_imputer']
    encoder = model_resources['encoder']
    property_mapping = model_resources['property_mapping']
    median_price_per_neigh = model_resources['median_price_per_neigh']
    selected_features = model_resources['selected_features']
    cat_cols = model_resources['cat_cols']
    num_cols = model_resources['num_cols']

    df = process_data(df, cat_imputer, num_imputer, encoder,
                     property_mapping, median_price_per_neigh,
                      cat_cols, num_cols)

    preds = predictions(model_params, df, scaler_X, selected_features)

    preds = pd.DataFrame(preds, columns=['price'])
    preds.to_csv('predictions.csv', index=False)



