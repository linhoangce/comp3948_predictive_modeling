import pandas as pd
import matplotlib as plt

path = r"C:\Users\linho\Desktop\CST\term3\pred_analytics\data\wnba.csv"

df = pd.read_csv(path, sep=',')
print(df.head())
print(df.describe())

df_gp = df[['GP']]
df_pts = df[['PTS']]
df_gp_clipped = df_gp.clip(upper=36)
df_pts_clipped = df_pts.clip(upper=860)

df['GP_Clipped'] = df_gp_clipped
df['PTS_CLipped'] = df_pts_clipped

df_non_outliers = df[(df['GP'] < 36) & (df['PTS'] < 860)]
print("\n\n")
print(df_non_outliers)

