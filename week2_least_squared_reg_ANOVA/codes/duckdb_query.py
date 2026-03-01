import pandas as pd
import duckdb

path = r"C:\Users\linho\Desktop\CST\term3\pred_analytics\data\retailerDB.csv"
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

mydf = pd.read_csv(path)

SQL = ("SELECT vendor, (price*0.85*quantity) AS TotalRevenue FROM mydf "
       "GROUP BY (vendor, price, quantity) "
       "HAVING vendor IN ('Silverware Inc.', 'Waterford Corp.')")
query_df = duckdb.sql(SQL).df()
print(query_df)