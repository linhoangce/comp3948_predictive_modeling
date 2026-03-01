import pandas as pd
from sqlalchemy import create_engine

path = r'C:\Users\linho\Desktop\CST\term3\pred_analytics\data'
csv_data = r'\retailerDB.csv'

df = pd.read_csv(path + csv_data)

def show_query_result(sql):
    # Create an in-memory table called 'Inventory'
    engine = create_engine('sqlite://', echo=False)
    connection = engine.connect()
    df.to_sql(name='RetailInventory', con=connection,
              if_exists='replace', index=False)

    # execute query
    query_result = pd.read_sql(sql, connection)
    return query_result

SQL = ("SELECT vendor, (price*0.85*quantity) as TotalRevenue FROM "
       "RetailInventory GROUP BY vendor "
       "HAVING vendor IN ('Silverware Inc.', 'Waterford Corp.')")
results = show_query_result(SQL)
print(results)