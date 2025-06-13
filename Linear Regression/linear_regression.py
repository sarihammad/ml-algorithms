import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # assuming X is a 2D array with shape (n_samples, n_features)
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features) # init weights with 0 for each feature
        self.bias = 0 # init bias term with 0

        for _ in range(self.n_iterations):
            # predictions
            y_pred = X @ self.weights + self.bias

            # gradients
            dw = (2 / n_samples) * X.T @ (y_pred - y)
            db = (2 / n_samples) * np.sum((y_pred - y))

            # update params
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return X @ self.weights + self.bias

    def mean_squared_error(self, y, y_pred):
        return np.mean((y - y_pred) ** 2)
