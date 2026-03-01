import pandas               as pd
import statsmodels.api      as sm
import numpy                as np
import matplotlib.pyplot    as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

#------------------------------------------------
# Shows plot of actual vs. predicted and RMSE.
#------------------------------------------------
def showResidualPlotAndRMSE(x, y, predictions):
    xmax      = max(x)
    xmin      = min(x)
    residuals = y - predictions

    plt.figure(figsize=(8, 3))
    plt.title('x and y')
    plt.plot([xmin,xmax],[0,0],'--',color='black')
    plt.title("Residuals")
    plt.scatter(x,residuals,color='red')
    plt.show()

    # Calculate RMSE
    mse = mean_squared_error(y,predictions)
    rmse = np.sqrt(mse)
    print("RMSE: " + str(rmse))

#------------------------------------------------
# Shows plot of x vs. y.
#------------------------------------------------
def showXandYplot(x,y, xtitle, title):
    plt.figure(figsize=(8, 4))
    plt.plot(x,y,color='blue')
    plt.title(title)
    plt.xlabel(xtitle)
    plt.ylabel('y')
    plt.show()

PATH   = r"C:\Users\linho\Desktop\CST\term3\pred_analytics\data\abs.csv"
df     = pd.read_csv(PATH)

x = df[['abs(450nm)']]  # absorbance
y = df[['ug/L']]        # protein concentration
showXandYplot(x,y, 'absorbance x', 'Protein Concentration(y) and Absorbance(x)')

# Show raw x and y relationship
x = sm.add_constant(x)

# Show model.
model       = sm.OLS(y, x).fit()
predictions = model.predict(x)
print(model.summary())

# Show RMSE.
preddf      = pd.DataFrame({"predictions":predictions})
residuals   = y['ug/L']-preddf['predictions']
resSq       = [i**2 for i in residuals]
rmse        = np.sqrt(np.sum(resSq)/len(resSq))
print("RMSE: " + str(rmse))

# Show the residual plot
plt.scatter(x['abs(450nm)'],residuals)
plt.show()


# grid search
dfX = pd.DataFrame({"x": df['abs(450nm)']})
dfY = pd.DataFrame({"y": df['ug/L']})
dfX = sm.add_constant(dfX)

# split into train and test so that transforms are chosen on training data
X_train, X_test, y_train, y_test = train_test_split(dfX, dfY, test_size=0.3,
                                                   )


transform_function = {
    'sqrt': lambda x: np.sqrt(x),
    'inv': lambda x: 1 / x,
    'neg_inv': lambda x: -1 / x,
    'sqr': lambda x: x * x,
    'log': lambda x: np.log(x),
    'neg_log': lambda x: -np.log(x),
    'exp': lambda x: np.exp(x),
    'neg_exp': lambda x: np.exp(-x),
}

# Pass a tuple of strings to test for trans. Refer to transform_function for the
# keywords.
def grid_search(dfX, y, trans):

    dfTransformations = pd.DataFrame()
    for tran in trans:
        # Transform x
        dfX['xt'] = transform_function[tran](dfX['x'])
        model_t = sm.OLS(y, dfX[['const', 'xt']]).fit()
        predictions_t = model_t.predict(dfX[['const', 'xt']])
        # Calculate RMSE
        mse = mean_squared_error(y, predictions_t)
        rmse = np.sqrt(mse)
        dfTransformations = dfTransformations._append({
            "tran":tran, "rmse":rmse}, ignore_index=True)
    dfTransformations = dfTransformations.sort_values(by=['rmse'])
    print(dfTransformations)
    bestTransform = dfTransformations.iloc[0]['tran']
    return bestTransform

print("Grod searching results:")
bestTransform = grid_search(X_train, y_train, ('sqrt', 'neg_inv', 'log', 'exp', 'neg_exp'))


# exp transformation

# X_train, X_test, y_train, y_test = train_test_split(
#     dfX, dfY, test_size=0.3, random_state=42
# )

# Section B: Transform and plot graph with transformed x.
# Use the correct feature column 'x'
X_train['xt'] = np.exp(X_train['x'])
X_test['xt'] = np.exp(X_test['x'])

showXandYplot(X_train['xt'] ,y_train, 'exp(x)', 'Protein Concentration(y) and exp(x)')

# Build model with transformed x.
# We only need the 'const' and the new 'xt' columns for the model
model_t       = sm.OLS(y_train, X_train[['const', 'xt']]).fit()
predictions_t = model_t.predict(X_test[['const', 'xt']])
print(model_t.summary())

# The showResidualPlotAndRMSE function expects a specific format
showResidualPlotAndRMSE(X_test['xt'], y_test['y'], predictions_t)