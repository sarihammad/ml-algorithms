import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations

        # initialize weights
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, a):
        return a * (1 - a)

    def fit(self, X, y):
        for _ in range(self.n_iterations):
            # forward pass
            z1 = X @ self.W1 + self.b1
            a1 = self.sigmoid(z1)

            z2 = a1 @ self.W2 + self.b2
            a2 = self.sigmoid(z2)

            # backward pass
            error = a2 - y
            dW2 = a1.T @ (error * self.sigmoid_derivative(a2))
            db2 = np.sum(error * self.sigmoid_derivative(a2), axis=0, keepdims=True)

            d_hidden = (error * self.sigmoid_derivative(a2)) @ self.W2.T * self.sigmoid_derivative(a1)
            dW1 = X.T @ d_hidden
            db1 = np.sum(d_hidden, axis=0, keepdims=True)

            # update weights
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2

    def predict(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = self.sigmoid(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = self.sigmoid(z2)
        return (a2 >= 0.5).astype(int)

    def accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)
