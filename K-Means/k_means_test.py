import numpy as np
import matplotlib.pyplot as plt
from k_means import KMeans

# generate synthetic clustered data
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)

# fit the KMeans model
kmeans = KMeans(k=3)
kmeans.fit(X)
labels = kmeans.labels
centroids = kmeans.centroids

# plot the clustered data
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30, label='Points')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, marker='X', label='Centroids')

plt.title("K-Means Clustering Result")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("k_means_demo.png")
plt.show()
