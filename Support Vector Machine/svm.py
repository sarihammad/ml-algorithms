import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param  # regularization strength
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y == -1, -1, 1)

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            for idx in range(n_samples):
                xi = X[idx]
                yi = y_[idx]
                condition = yi * (np.dot(xi, self.weights) + self.bias) >= 1
                if condition:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights - yi * xi)
                    self.bias -= self.learning_rate * yi

    def predict(self, X):
        approx = np.dot(X, self.weights) + self.bias
        return np.sign(approx)

    def accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)
