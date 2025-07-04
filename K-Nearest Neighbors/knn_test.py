import numpy as np
from knn import KNNClassifier

X_train = np.array([[1], [2], [3], [10], [11], [12]])
y_train = np.array([0, 0, 0, 1, 1, 1])

X_test = np.array([[4], [9]])
y_test = np.array([0, 1])

model = KNNClassifier(k=3)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print("Predictions:", preds)
print("Accuracy:", model.score(X_test, y_test))
