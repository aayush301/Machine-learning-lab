import numpy as np


class KNN:
    def __init__(self, k=5):
        self.X = np.array
        self.y = np.array
        self.k = k

    def _convert_str_to_float(self, X):
        for row in range(len(X)):
            for col in range(len(X[0])):
                X[row][col] = float(X[row][col])
        return X

    def _euclidean_distance(self, arr1, arr2):
        dist = 0.0
        for i in range(len(arr1)):
            dist += (arr1[i] - arr2[i]) ** 2
        return np.sqrt(dist)

    def _get_k_neighbor_classes(self, sample):
        distances = np.array(
            [self._euclidean_distance(row, sample) for row in self.X])
        index_dists = distances.argsort()
        return list(self.y[index_dists[:self.k]])

    def _predict_class(self, sample):
        neighbor_classes = self._get_k_neighbor_classes(sample)
        return max(neighbor_classes, key=neighbor_classes.count)

    def fit(self, X, y):
        self.X = self._convert_str_to_float(X)
        self.y = y

    def predict(self, X):
        y_pred = [self._predict_class(sample)
                  for sample in self._convert_str_to_float(X)]
        return y_pred
