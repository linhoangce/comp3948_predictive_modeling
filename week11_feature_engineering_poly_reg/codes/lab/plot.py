import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def show_X_y(X, y, title, xlab):
    plt.plot(X, y, color='blue')
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel('y')

def show_residual_rmse(X, y, y_pred):
    residuals = y - y_pred
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    plt.plot([min(X), max(X)], [0, 0], '--', color='black')
    plt.title(f'Residuals - $R^2$={r2:.4f}, RMSE={rmse:.4f}')
    plt.scatter(X, residuals, color='red')
