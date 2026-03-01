import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

def impute_nan_columns(X_train, X_test):
   cat_cols = X_train.select_dtypes(include=['object', 'category']).columns
   for col in cat_cols:
       mode_val = X_train[col].mode(dropna=True)
       if len(mode_val) == 0:
           continue
       mode_val = mode_val[0]
       X_train[col] = X_train[col].fillna(mode_val)

       if col in X_test.columns:
           X_test[col] = X_test[col].fillna(mode_val)

       numeric_cols = X_train.select_dtypes(include=[np.number]).columns
       if len(numeric_cols) > 0:
           imputer = KNNImputer(n_neighbors=10)
           X_train_num = imputer.fit_transform(X_train[numeric_cols])
           X_test_num = imputer.transform(X_test[numeric_cols])

           X_train[numeric_cols] = X_train_num
           X_test[numeric_cols] = X_test_num
   return X_train, X_test, imputer

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

def convert_to_standard_datetime(df, col_name):
    df = df.copy()
    df[col_name] = pd.to_datetime(df[col_name], format="mixed")
    return df

def convert_host_experience(df):
    df = df.copy()
    df['host_years'] = (df['host_since'].max() - df['host_since']).dt.days/ 365
    df['host_exp'] = pd.cut(df['host_years'],
                            bins=[0, 2, 5, 7, 10],
                            labels=['new', 'mid', 'experienced', 'veteran'])
    df['host_exp'] = df['host_exp'].astype(str)
    return df

def convert_str_to_numeric(df):
    df = df.copy()
    df["host_response_rate"] = df["host_response_rate"].astype(str).str.replace("%", "")
    df["host_response_rate"] = pd.to_numeric(df["host_response_rate"], errors='coerce').fillna(0)

    df['cleaning_fee'] = pd.to_numeric(df['cleaning_fee'], errors='coerce').astype(int)

    bool_columns = ['host_has_profile_pic', 'host_identity_verified',
                    'instant_bookable']
    for col in bool_columns:
        df[col] = (df[col] == 't').astype(int)

    # datetime_cols = ['first_review', 'host_since', 'last_review']
    # for col in datetime_cols:
    #     del df[col]
    return df

def create_dummies(X_train, X_test):
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns
    print(cat_cols)
    for col in cat_cols:
        X_train[col] = X_train[col].astype(str)
        X_test[col] = X_test[col].astype(str)

    encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    encoder.fit(X_train[cat_cols])

    train_cat_encoded = encoder.transform(X_train[cat_cols])
    test_cat_encoded = encoder.transform(X_test[cat_cols])
    encoded_cols = encoder.get_feature_names_out(cat_cols)

    df_train_cat = pd.DataFrame(train_cat_encoded, columns=encoded_cols)
    df_test_cat = pd.DataFrame(test_cat_encoded, columns=encoded_cols)

    num_cols = [col for col in X_train.columns if col not in cat_cols]

    X_train_final = pd.concat(
        [X_train[num_cols].reset_index(drop=True), df_train_cat.reset_index(drop=True)],
        axis=1
    )
    X_test_final = pd.concat(
        [X_test[num_cols].reset_index(drop=True), df_test_cat.reset_index(drop=True)],
        axis=1
    )

    X_train_final, X_test_final = X_train_final.align(
        X_test_final, join='left', axis=1, fill_value=0
    )

    return X_train_final, X_test_final, encoder

def preprocess_dataset(df, target_col: str='price'):
    df = df.copy()

    df = convert_str_to_numeric(df)
    df = group_property_types(df)
    df = convert_to_standard_datetime(df, 'host_since')
    df = convert_host_experience(df)

    X = df.drop(columns=[target_col, 'neighbourhood', 'zipcode']).copy()
    y = df[target_col].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=10000, shuffle=True
    )

    X_train, X_test, imputer = impute_nan_columns(X_train, X_test)
    X_train, X_test, encoder = create_dummies(X_train, X_test)

    X_train.to_csv('data/train_features.csv', index=False)
    X_test.to_csv('data/test_features.csv', index=False)
    y_train.to_csv('data/train_label.csv', index=False)
    y_test.to_csv('data/test_label.csv', index=False)
    return X_train, X_test, y_train, y_test, imputer, encoder



if __name__ == "__main__":
    df = pd.read_csv('data/AirBNB.csv')
    X_train, X_test, y_train, y_test, imputer, encoder = preprocess_dataset(
        df
    )