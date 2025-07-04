import numpy as np
from collections import Counter

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        # distances to all training points
        distances = np.linalg.norm(self.X_train - x, axis=1)

        # indices of k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]

        # labels of the k neighbors
        k_nearest_labels = self.y_train[k_indices]

        # majority vote
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def score(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)
