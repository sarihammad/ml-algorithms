import numpy as np
from pca import PCA

# generate some synthetic 3D data
np.random.seed(42)
mean = [0, 0, 0]
cov = [[3, 2, 1], [2, 2, 1], [1, 1, 1]]  # correlated features
X = np.random.multivariate_normal(mean, cov, size=100)

# fit PCA to reduce to 2 dimensions
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# print results
print("Original shape:", X.shape)
print("--------------")
print("Reduced shape:", X_reduced.shape)
print("--------------")
print("Principal Components (each row is a PC):")
print(pca.components_)
