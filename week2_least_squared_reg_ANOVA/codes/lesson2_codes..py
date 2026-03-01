import pandas as pd
from sklearn.model_selection import train_test_split
from statsmodels.formula.api import ols
from sklearn import metrics
import math

def perform_simple_regression():
    data = [[0.19, 0.13],
            [0.28, 0.12],
            [0.35, 0.35],
            [0.37, 0.34],
            [0.20, 0.36],
            [0.16, 0.90]]

    df = pd.DataFrame(data, columns=['X', 'target'])

    # split datasets using 60% for training
    # X_train, X_test, y_train, y_test = train_test_split(df['X'],
    #                                                     df['target'],
    #                                                     test_size=0.2)
    #
    # # create train and test dataframes
    # train_data = {'X': X_train, 'target': y_train}
    # train_df = pd.DataFrame(train_data)
    #
    # # initialize model
    # model = ols('target~X', data=train_df).fit()
    # y_pred = model.predict(X_test)
    #
    # # present X_test, y_test, y_pred and error sum of squares
    # data = {'X_test': X_test, 'y_test': y_test, 'y_pred': y_pred}
    # df_result = pd.DataFrame(data)
    # df_result['y_test - y_pred'] = (df_result['y_test'] - df_result['y_pred'])
    # df_result['(y_test - y_pred)^2'] = (df_result['y_test'] - df_result['y_pred'])**2
    #
    # print(df_result)
    #
    # # manually calculate the deviation between actual and predicted vals
    # rmse = math.sqrt(df_result['(y_test - y_pred)^2'].sum() / len(df_result))
    # print(f'RMSE is average deviation between actual and predicted values: \n'
    #       f'{str(rmse)}')
    #
    # # show faster way to calculate deviation between actual and predicted values
    # rmse2 = math.sqrt(metrics.mean_squared_error(y_test, y_pred))
    # print(f'The automated root mean square error calculation: {rmse2}')

    model = ols('target~X', data=df).fit()

    # show model summary
    print(model.summary())

    print('\nMake prediction')
    x_sample = df['X']
    print('X=' + str(x_sample))
    preds = model.predict(x_sample)
    print('y_pred=' + str(preds))

perform_simple_regression()