import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None

    def fit(self, X):
        # center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # covariance matrix
        cov = np.cov(X_centered, rowvar=False)

        # eigenvalues and eigenvectors
        eigvals, eigvecs = np.linalg.eigh(cov)  # eigh is for symmetric matrices

        # Sort eigenvectors by descending eigenvalues
        sorted_idx = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, sorted_idx]

        # select the top n_components
        self.components_ = eigvecs[:, :self.n_components].T  # shape: (n_components, n_features)

    def transform(self, X):
        X_centered = X - self.mean_
        return X_centered @ self.components_.T  # project onto principal components

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
