from sklearn.preprocessing import StandardScaler
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris.data
features = ['sepal length (cm)', 'sepal width (cm)',
            'petal length (cm)', 'petal width (cm)']
y = iris.target
X_std = StandardScaler().fit_transform(X)

# Generate covariance matrix to show bivariate relationships
cov_mat = np.cov(np.transpose(X_std))
print(f'\nCovariance matrix:\n {cov_mat}')

# When data is standardized, the covariance matrix is the same as
# the correlation matrix
cor_mat = np.corrcoef(np.transpose(X_std))
print(f"\nCorrelation matrix:\n {cor_mat}")

# Perform Eigen decomposition on the covariance matrix
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print(f"\nEigenvectors:\n {eig_vecs}")
print(f"\nEigenvalues:\n {eig_vals}")

# Show scree plot
plt.plot([1, 2, 3, 4], eig_vals, 'ro-', linewidth=2)
plt.title("Scree plot")
plt.xlabel("Principal Component")
plt.ylabel('Eigenvalues')
plt.show()

# Calculate cumulative values
sum_eigenvalues = eig_vals.sum()
# cum_values = []
# cum_sum = 0

# for i in range(len(eig_vals) + 1):
#     cum_values.append(cum_sum)
#     if i < len(eig_vals):
#         cum_sum += eig_vals[i] / sum_eigenvalues

# OPTIMIZED CODES ========================================================
from itertools import accumulate
cum_values = [0] + list(accumulate(v/sum_eigenvalues for v in eig_vals))
# ========================================================================

plt.plot([0, 1, 2, 3, 4], cum_values, 'ro-', linewidth=2)
plt.title('Variance Explained by Principal components')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
plt.show()

