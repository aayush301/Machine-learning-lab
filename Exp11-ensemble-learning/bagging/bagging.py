import numpy as np
from sklearn.tree import DecisionTreeClassifier


class Bagger:
    def fit(self, X_train, y_train, B, seed=None):
        self.X_train = X_train
        self.N, self.D = X_train.shape
        self.y_train = y_train
        self.B = B
        self.seed = seed
        self.trees = []

        np.random.seed(seed)

        for b in range(self.B):
            sample = np.random.choice(
                np.arange(self.N), size=self.N, replace=True)
            X_train_b = X_train[sample]
            y_train_b = y_train[sample]

            tree = DecisionTreeClassifier()
            tree.fit(X_train_b, y_train_b)
            self.trees.append(tree)

    def predict(self, X_test):
        y_test_hats = np.empty((len(self.trees), len(X_test)))
        for i, tree in enumerate(self.trees):
            y_test_hats[i] = tree.predict(X_test)

        return y_test_hats.mean(0)
