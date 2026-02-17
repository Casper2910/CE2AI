import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1:
iris = pd.read_csv('iris.csv')

# features (d dimensions) and target
x = iris.drop(columns=['variety'])
y = iris['variety']

# split: 70% train, 30% test
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=1, stratify=y
)

# standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2:
import numpy as np

def covariance_matrix(matrix):
    """
    matrix: numpy array of shape (n_samples, n_features)
    returns: covariance matrix of shape (n_features, n_features)
    https://www.youtube.com/watch?v=152tSYtiQbw
    """
    # count features:
    n_samples = matrix.shape[0]
    
    # calculate mean of each feature
    means = np.mean(matrix, axis=0)
    
    # center the data
    centered_matrix = matrix - means
    
    # compute covariance
    cov = centered_matrix.T.dot(centered_matrix)
    
    #average the feature-wise deviation products using the sample covariance formula.
    cov /= (n_samples - 1)
    
    return cov

# Compute covariance matrix
cov_mat = covariance_matrix(X_train_scaled)
print(cov_mat, type(cov_mat))

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_mat)

print(eigenvalues)
print(eigenvectors)

# 3:
# sum of eigenvalues
tot = sum(eigenvalues)

# sort for largest eigenvalues
var_exp = [(i / tot) for i in sorted(eigenvalues, reverse=True)]

# cumulative sum of the elements
cum_var_exp = np.cumsum(var_exp)

n_components = len(eigenvalues) # 4 features

#plot:
import matplotlib.pyplot as plt
plt.bar(range(1, n_components + 1), var_exp, alpha=0.5, align='center', label='Individual explained variance')
plt.step(range(1, n_components + 1), cum_var_exp, where='mid', label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# 4:
k = 3
# Make a list of (eigenvalue, eigenvector) tuples
eigen_pairs = [(np.abs(eigenvalues[i]), eigenvectors[:, i]) for i in range(len(eigenvalues))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eigen_pairs.sort(key=lambda k: k[0], reverse=True)  

print(eigen_pairs)

w = np.hstack([
    eigen_pairs[i][1][:, np.newaxis]
    for i in range(k)
])
print('Matrix W:\n', w)

# 5:
# project onto the PCA subspace
X_train_proj = X_train_scaled @ w

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for label in y_train.unique():
    idx = y_train == label
    ax.scatter(
        X_train_proj[idx, 0],
        X_train_proj[idx, 1],
        X_train_proj[idx, 2],
        label=label,
        s=30
    )

ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('PC 3')
ax.set_title('Iris samples projected onto PCA subspace')
ax.legend()

plt.show()
