import numpy as np
from logistic_regression import LogisticRegression

# binary classification (y = 1 if x >= 4 and 0 otherwise)
X = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([0, 0, 0, 1, 1, 1])

model = LogisticRegression()
model.fit(X, y)

probs = model.predict_proba(X)
preds = model.predict(X)

print(f"Probabilities: {probs}")
print("---------------------")
print(f"Predictions: {preds}")
print("---------------------")
print(f"Actual: {y}")
