import numpy as np
from linear_regression import LinearRegression

# y = 2x + 1
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([3, 5, 7, 9, 11])

model = LinearRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X, y)
preds = model.predict(X)

print(f"Predictions: {preds}")
print("---------------------------")
print(f"MSE: {model.mean_squared_error(y, preds)}")

