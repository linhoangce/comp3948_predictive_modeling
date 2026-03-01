import numpy as np
import pandas as pd
import pickle

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

def preprocess_dataset(df, encoder, imputer, employment_map, mode_val):
    """
    1. Impute cat (mode) + numeric (KNN) on train, apply to test.
    2. Encode ordinal 'employment' with fixed mapping.
    3. Add fixed age bins -> dummy.
    4. Create dummy variables for all other categoricals.
    """
    X = df.copy()

    # ==================================================================
    # Encode ordinal "employment"
    # ==================================================================
    if 'employment' not in X.columns:
        X['employment'] = '<1'

    X['employment'] = X['employment'].map(employment_map)
    X['employment'] = pd.to_numeric(X['employment'])

    # ==================================================================
    # Impute NaN columns
    # ==================================================================
    # Categorical imputation (mode)
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        X[col] = X[col].fillna(mode_val)

    # Numeric imputation (KNN on all numeric columns)
    num_cols = X.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        X_num = imputer.transform(X[num_cols])
        X[num_cols] = X_num

    # ==================================================================
    # Bin "age"
    # ==================================================================
    if "age" in X.columns:
        bins = [18, 25, 35, 50, 65, 100]
        labels = ["18-25", "25-35", "35-50", "50-65", "65+"]

        age_bin = pd.cut(
            X['age'],
            bins=bins,
            labels=labels,
            include_lowest=True,
            right=True
        )
        age_dummies = pd.get_dummies(
            age_bin, prefix='age_bin', drop_first=True, dtype=int
        )

        X = X.drop(columns=['age'])

        X = pd.concat([X.reset_index(drop=True), age_dummies.reset_index(drop=True)], axis=1)

    else:
        raise Exception(f'Missing required column name: "age"!\n'
                        f'Cannot proceed with processing the data!')

    # ==================================================================
    # Create dummies
    # ==================================================================
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    encoder.fit(X[cat_cols])

    cat_encoded = encoder.transform(X[cat_cols])
    encoded_cols = encoder.get_feature_names_out(cat_cols)

    # create DataFrame with encoded features
    df_cat = pd.DataFrame(cat_encoded, columns=encoded_cols)

    num_cols = [col for col in X.columns if col not in cat_cols]

    X_final = pd.concat(
        [X[num_cols].reset_index(drop=True), df_cat.reset_index(drop=True)],
        axis=1
    )

    return X_final


def make_predictions(model, X, features, scaler):
    X_selected = X[features]
    X_scaled = scaler.transform(X_selected)

    y_pred = model.predict(X_scaled)

    return y_pred


def main():
    df = pd.read_csv('data/Credit_Mystery.csv')

    with open('model_training_resources.pkl', 'rb') as f:
        training_resources = pickle.load(f)

    encoder = training_resources['encoder']
    imputer = training_resources['imputer']
    employment_map = training_resources['employment_map']
    mode_val = training_resources['mode_val']
    scaler = training_resources['scaler']
    features = training_resources['features']
    model = training_resources['model']
    all_columns = training_resources['full_columns']
    all_columns_df = pd.DataFrame(columns=all_columns)

    X = preprocess_dataset(df, encoder, imputer, employment_map, mode_val)

    # Align all training features on production dataset to avoid KeyError
    X, _ = X.align(all_columns_df, join='right', axis=1, fill_value=0)

    y_pred = make_predictions(model, X, features, scaler)
    pred_df = pd.DataFrame(y_pred, columns=['class'])

    print(f'Predictions:\n{y_pred}\n')
    pred_df.to_csv('Predictions.csv', index=False)
    print('Saved predictions!')


if __name__ == '__main__':
    main()