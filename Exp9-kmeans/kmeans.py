from random import uniform
import numpy as np


def euclidean_dist(point, data):
    return np.sqrt(np.sum((point-data)**2, axis=1))


class KMeans:
    def __init__(self, n_clusters=8, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X_train):
        min_ = np.min(X_train, axis=0)
        max_ = np.max(X_train, axis=0)
        self.centroids = [uniform(min_, max_) for _ in range(self.n_clusters)]

        iteration = 0
        prev_centroids = None
        while np.not_equal(self.centroids, prev_centroids).any() and iteration < self.max_iter:
            sorted_points = [[] for _ in range(self.n_clusters)]
            for x in X_train:
                dists = euclidean_dist(x, self.centroids)
                centroid_idx = np.argmin(dists)
                sorted_points[centroid_idx].append(x)

                prev_centroids = self.centroids
                self.centroids = [np.mean(cluster, axis=0)
                                  for cluster in sorted_points]

                for i, centroid in enumerate(self.centroids):
                    if np.isnan(centroid).any():
                        self.centroids[i] = prev_centroids[i]
            iteration += 1

    def predict(self, X):
        centroids = []
        centroid_ids = []
        for sample in X:
            dists = euclidean_dist(sample, self.centroids)
            centroid_idx = np.argmin(dists)
            centroids.append(self.centroids[centroid_idx])
            centroid_ids.append(centroid_idx)
        return centroids, centroid_ids
