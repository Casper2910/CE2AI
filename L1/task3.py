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
    """
    n_samples = matrix.shape[0]
    mean = np.mean(matrix, axis=0)
    X_centered = matrix - mean
    cov = (X_centered.T @ X_centered) / (n_samples - 1)
    return cov

# Compute covariance matrix
cov_mat = covariance_matrix(X_train_scaled)

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_mat)
