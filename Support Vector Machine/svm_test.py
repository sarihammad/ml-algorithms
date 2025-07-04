import numpy as np
from svm import SVM

# Simple binary classification example
X = np.array([[1, 2], [2, 3], [3, 3], [5, 5], [1, 0], [2, 1]])
y = np.array([0, 0, 0, 1, 1, 1])

# Convert to -1 and 1
y = np.where(y == 0, -1, 1)

model = SVM()
model.fit(X, y)
predictions = model.predict(X)

print("Predictions:", predictions)
print("Accuracy:", model.accuracy(y, predictions))
