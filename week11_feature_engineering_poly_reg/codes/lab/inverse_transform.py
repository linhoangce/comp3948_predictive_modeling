import pandas as pd
from sklearn.model_selection import train_test_split

from lab.plot import show_X_y

x = [0.2, 0.5, 0.7, 0.9,1,2,3,4,5,6]
y = [5.0, 2.0, 1.44, 1.12, 1.1, 0.53, 0.34, 0.25, 0.2, 0.17]

df_X = pd.DataFrame({'x': x})
df_y = pd.DataFrame({'y': y})

X_train, X_test, y_train, y_test = train_test_split(
    df_X, df_y, test_size=0.3, random_state=42
)

X_train['xt'] = 1/X_train['x']
X_test['xt'] = 1/X_test['x']

show_X_y(x, y, 'x', 'Before')
show_X_y(X_train['xt'], y_train, 'xt', 'y=1/x')