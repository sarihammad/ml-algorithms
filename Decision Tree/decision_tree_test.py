import numpy as np
from decision_tree import DecisionTree
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# synthetic data
X, y = make_classification(n_samples=200, n_features=2, n_classes=2, n_redundant=0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

tree = DecisionTree(max_depth=10)
tree.fit(X_train, y_train)
predictions = tree.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
