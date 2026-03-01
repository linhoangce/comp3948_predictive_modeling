import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition      import PCA
from sklearn.preprocessing      import StandardScaler
from sklearn.metrics            import classification_report
import pandas                   as pd
from sklearn.model_selection    import train_test_split
from sklearn.linear_model import LogisticRegression

PATH     = r"C:\Users\linho\Desktop\CST\term3\pred_analytics\data\milk.csv" # Set your folder path here.
df       = pd.read_csv(PATH)

# Split the data at the start to hold back a test set.
train, test = train_test_split(df, test_size=0.2)

X_train = train.copy()
X_test  = test.copy()
del X_train['labels']
del X_test['labels']
del X_train['dates']
del X_test['dates']

y_train = train['labels']
y_test  = test['labels']

# Scale X values.
xscaler         = StandardScaler()
Xtrain_scaled   = xscaler.fit_transform(X_train)
Xtest_scaled    = xscaler.transform(X_test)

# print(Xtrain_scaled.shape)

# Generate PCA components.
pca = PCA(0.8)

# Always fit PCA with train data. Then transform the train data.
X_reduced_train = pca.fit_transform(Xtrain_scaled)

# Transform test data with PCA
X_reduced_test  = pca.transform(Xtest_scaled)

print("\nPrincipal Components")
print(pca.components_)

print("\nExplained variance: ")
print(pca.explained_variance_)

# Train regression model on training data
model = LogisticRegression(solver='liblinear')
model.fit(X_reduced_train, y_train)

# Predict with test data.
preds = model.predict(X_reduced_test)

report = classification_report(y_test, preds)
print(report)

cov_mat = np.cov(np.transpose(Xtrain_scaled))
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

eig_vals_sorted = np.sort(eig_vals)[::-1]
N = 10

plt.plot(range(1, N+1), eig_vals_sorted[:N], 'ro-', linewidth=2)
plt.title('Scree plot of Top 10 Features')
plt.xticks(range(1, N+1))
plt.xlabel('Principal Components')
plt.ylabel('Eigen values')
plt.show()

eig_vals_sum = eig_vals.sum()

from itertools import accumulate
cum_values = [0] + list(accumulate(e / eig_vals_sum for e in eig_vals))

plt.plot(range(1, N+1), cum_values[:N], 'ro-', linewidth=2)
plt.title('Variance Explained by PC (Top 10)')
plt.xticks(range(1, N+1))
plt.xlabel('Principal Components')
plt.ylabel('Cumulative Eigen value')
plt.show()