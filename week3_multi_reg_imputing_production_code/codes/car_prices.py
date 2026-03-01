import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose


PATH = r"C:\Users\linho\Desktop\CST\term3\pred_analytics\data\carPrice.csv"

df = pd.read_csv(PATH)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

def view_quick_stats():
    print("\n*** Show contents of the file")
    print(df.head())

    print("\n*** Description of all columns")
    print(df.info())

    print("\n*** Describe numeric values")
    print(df.describe())

    print("\n*** Showing frequencies")

    # Show frequencies
    print(df["model"].value_counts())
    print("")
    print(df['transmission'].value_counts())
    print("")
    print(df['fuel type'].value_counts())
    print("")
    print(df['engine size'].value_counts())
    print("")
    print(df['fuel type2'].value_counts())
    print("")
    print(df['year'].value_counts)
    print("")

### fix price column
for i in range(0, len(df)):
    priceStr = str(df.iloc[i]['price'])
    priceStr = priceStr.replace("£", "")
    priceStr = priceStr.replace("-", "")
    priceStr = priceStr.replace(",", "")
    df.at[i, "price"] = priceStr

df['price'] = pd.to_numeric(df['price'])

### impute missing values for year column
avg_year = df['year'].mean()
df.loc[df['year'].isna(), 'year'] = avg_year.astype(int)

### impute missing values for engine size2 column
df.loc[df['engine size2'].isna(), 'engine size2'] = 0

df['engine size2'] = pd.to_numeric(df['engine size2'])
df['mileage2'].value_counts()

view_quick_stats()

### Fix mileage column
for i in range(0, len(df)):
    mileageStr = str(df.loc[i]["mileage"])
    mileageStr = mileageStr.replace(",", "")
    df.at[i, "mileage"] = mileageStr

    try:
        if not mileageStr.isnumeric():
            df.at[i, "mileage"] = "0"
    except Exception as e:
        error = str(e)
        print(error)

df["mileage"] = pd.to_numeric(df["mileage"])
view_quick_stats()

### Exercise 1
# tempDf = df[['transmission', 'fuel type2']]
# dummyDf = pd.get_dummies(tempDf, columns=['transmission', 'fuel type2'], dtype=int)
# df = pd.concat([df, dummyDf], axis=1)

# get the unique values from each column
transmission = pd.unique(df['transmission'])
fuel_type2 = pd.unique(df['fuel type2'])

# create mapping dict
transmissions = {val: i for i, val in enumerate(transmission)}
fuel_type2 = {val: i for i, val in enumerate(fuel_type2)}

# map each matching key to a value of int in the columns
transmission_mapped = df['transmission'].map(transmissions)
fuel_type2_mapped = df['fuel type2'].map(fuel_type2)
# create two new columns for dummy variables
df['dummy_transmission'] = transmission_mapped
df['dummy_fuel_type2'] = fuel_type2_mapped
# print(df.head(20))

#
# print(df.head())
# #
# # #############################################################3
# # ### compute correlation matrix
# # corr = df.corr()
# # # plot heatmap
# # sns.heatmap(corr, xticklabels=corr.columns,
# #             yticklabels=corr.columns)
# # plt.show()

X = df[['engine size2', 'year']].values
X = sm.add_constant(X)
y = df['price'].values

# split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

model = sm.OLS(y_train, X_train).fit()
y_pred = model.predict(X_test)

print(model.summary())

print(f'RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}')

X = df[['engine size2', 'year', 'dummy_transmission', 'dummy_fuel_type2']].values
X = sm.add_constant(X)
y = df['price'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2,
)

model = sm.OLS(y_train, X_train).fit()
y_pred = model.predict(X_test)

print("\n\n==========================================================================================")
print(model.summary())
print(f'RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}')

### Exercise 3
print("\n\n========================================================================================")

df['year_binned'] = df['year'].where(df['year'] > 2013, '2013_before')
dummy_bins = pd.get_dummies(df['year_binned'], prefix="year", dtype=int)
df = pd.concat([df, dummy_bins], axis=1)
print(df.head(50))

X = df[['engine size2', 'dummy_transmission',
        'dummy_fuel_type2', 'year_2020.0', 'year_2019.0',
        'year_2018.0', 'year_2017.0', 'year_2016.0', 'year_2015.0',
        'year_2014.0', 'year_2013_before']].values
X = sm.add_constant(X)
y = df['price'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = sm.OLS(y_train, X_train).fit()
y_pred = model.predict(X_test)

print(model.summary())
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")

print("\n\n========================================================")

def predictCarPrice(true_price, transmission, fuel_type2, year_2020=0,
                    year_2019=0, year_2018=0, year_2017=0, year_2015=0,
                    year_2014=0, year_2013_before=0):
    pred_price = (2.171e+04 -1161.9829*transmission -1662.8866*fuel_type2
         + 1.851e+04*year_2020 + 1.081e+04*year_2019 + 5645.5256*year_2018 +
         3200.5298*year_2017 -2768.2490*year_2015 -3834.0026*year_2014
         -1.037e+04*year_2013_before)

    print(f'Actual price: {true_price} | Predicted price: {pred_price}')

print(predictCarPrice(30495, 0, 0, year_2020=1))
print(predictCarPrice(24500, 0, 2, year_2017=1))
print(predictCarPrice(14495, 2, 1, year_2015=1))