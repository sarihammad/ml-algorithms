import numpy as np

class KMeans:
    def __init__(self, k=3, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol  # tolerance for convergence
        self.centroids = None
        self.labels = None

    def fit(self, X):
        n_samples, _ = X.shape
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_indices]

        for _ in range(self.max_iters):
            # assignment step
            distances = self._compute_distances(X)
            labels = np.argmin(distances, axis=1)

            # recompute centroids
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.k)])

            # check for convergence
            diff = np.linalg.norm(self.centroids - new_centroids)
            if diff < self.tol:
                break

            self.centroids = new_centroids

        self.labels = labels

    def predict(self, X):
        distances = self._compute_distances(X)
        return np.argmin(distances, axis=1)

    def _compute_distances(self, X):
        return np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
