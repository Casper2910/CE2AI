import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# import
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
X_train_std = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# count features:
d = X_train_std.shape[1]

np.set_printoptions(precision=4)

# mean vector per class
mean_vecs = []
unique_labels = np.unique(y_train)  # use actual labels present in training data
for label in unique_labels:
    mean_vecs.append(np.mean(X_train_std[y_train == label], axis=0))
    print('MV %s: %s\n' % (label, mean_vecs[-1]))

# within-class scatter matrix
S_W = np.zeros((d, d))
for label, mv in zip(unique_labels, mean_vecs):
    class_scatter = np.zeros((d, d))
    for row in X_train_std[y_train == label]:
        row, mv = row.reshape(d, 1), mv.reshape(d, 1)
        class_scatter += (row - mv).dot((row - mv).T)
    S_W += class_scatter

print('Within-class scatter matrix: %sx%s' % (S_W.shape[0], S_W.shape[1]))

# between-class scatter matrix
mean_overall = np.mean(X_train_std, axis=0)
S_B = np.zeros((d, d))
for i, (label, mean_vec) in enumerate(zip(unique_labels, mean_vecs)):
    n            = X_train_std[y_train == label, :].shape[0]
    mean_vec     = mean_vec.reshape(d, 1)      # make column vector
    mean_overall = mean_overall.reshape(d, 1)
    S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)

print('Between-class scatter matrix: %sx%s' % (S_B.shape[0], S_B.shape[1]))

print('Between-class scatter matrix: %sx%s' % (S_B.shape[0], S_B.shape[1]))

# eigendecomposition of S_W^-1 * S_B (pinv handles singular/near-singular S_W)
eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.pinv(S_W).dot(S_B))

eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]

# sorting eigenvalues
eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)

print('Eigenvalues in descending order:\n')
for eigen_val in eigen_pairs:
    print(eigen_val[0])
    
import matplotlib.pyplot as plt

# discriminability plot
tot = sum(eigen_vals.real)
discr = [(i / tot) for i in sorted(eigen_vals.real, reverse=True)]
cum_discr = np.cumsum(discr)
plt.bar(range(1, d + 1), discr, alpha=0.5, align='center', label='Individual "discriminability"')
plt.step(range(1, d + 1), cum_discr, where='mid', label='Cumulative "discriminability"')
plt.ylabel('"Discriminability" ratio')
plt.xlabel('Linear Discriminants')
plt.ylim([-0.1, 1.1])
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# transformation matrix:
w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real, eigen_pairs[1][1][:, np.newaxis].real))
print('Matrix W:\n', w)

# project onto linear discriminants and plot
X_train_lda = X_train_std.dot(w)
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_lda[y_train==l, 0],
    X_train_lda[y_train==l, 1] * (-1),
    c=c, label=l, marker=m)
    
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()