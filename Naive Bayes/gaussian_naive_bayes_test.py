import numpy as np
from gaussian_naive_bayes import GaussianNaiveBayes

# 
X = np.array([[1], [2], [3], [8], [9], [10]])
y = np.array([0, 0, 0, 1, 1, 1])

model = GaussianNaiveBayes()
model.fit(X, y)

preds = model.predict(X)

print(f"Actual: {y}")
print("--------------------")
print(f"Predictions: {preds}")
print("--------------------")
print(f"Accuracy: {np.mean(preds == y)}")
