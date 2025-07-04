import numpy as np

class GaussianNaiveBayes:
    def __init__(self):
        self.classes = None
        self.class_priors = {}
        self.class_feature_means = {}
        self.class_feature_vars = {}

    def fit(self, X, y):
        self.classes = np.unique(y)

        for c in self.classes:
            X_c = X[y == c]
            self.class_priors[c] = X_c.shape[0] / X.shape[0]
            self.class_feature_means[c] = X_c.mean(axis=0)
            self.class_feature_vars[c] = X_c.var(axis=0) + 1e-9 # epsilon for stability

    def predict(self, X):
        predictions = [self._predict_sample(x) for x in X]
        return np.array(predictions)

    def _predict_sample(self, x):
        posteriors = []

        for c in self.classes:
            prior = np.log(self.class_priors[c])
            class_mean = self.class_feature_means[c]
            class_var = self.class_feature_vars[c]
            likelihood = -0.5 * np.sum(np.log(2 * np.pi * class_var))
            likelihood -= 0.5 * np.sum(((x - class_mean) ** 2) / class_var)
            posteriors.append(prior + likelihood)

        return self.classes[np.argmax(posteriors)]
