from asyncio import threads
import numpy as np


class DecisionStump:
    def __init__(self):
        self.polarity = 1
        self.threshold = None
        self.feature_idx = None
        self.alpha = None

    def predict(self, X):
        n_samples = X.shape[0]
        X_c = X[:, self.feature_idx]
        preds = np.ones(n_samples)
        if self.polarity == 1:
            preds[X_c < self.threshold] = -1
        else:
            preds[X_c > self.threshold] = -1

        return preds


class AdaBoost:
    def __init__(self, n_clf=5):
        self.n_clf = n_clf

    def fit(self, X, y):
        n_samples, n_features = X.shape
        w = np.full(n_samples, (1/n_samples))

        self.clfs = []
        for _ in range(self.n_clf):
            clf = DecisionStump()
            min_error = float('inf')
            for feature in range(n_features):
                X_c = X[:, feature]
                thresholds = np.unique(X_c)
                for threshold in thresholds:
                    p = 1
                    preds = np.ones(n_samples)
                    preds[X_c < threshold] = -1

                    misclassified = w[y != preds]
                    error = sum(misclassified)

                    if error > 0.5:
                        p = -1
                        error = 1 - error

                    if error < min_error:
                        min_error = error
                        clf.threshold = threshold
                        clf.feature_idx = feature
                        clf.polarity = p

            EPS = 1e-10
            clf.alpha = 0.5 * \
                np.log((1.0 - min_error + EPS) / (min_error + EPS))
            preds = clf.predict(X)
            w *= np.exp(-clf.alpha * y * preds)
            w /= np.sum(w)
            self.clfs.append(clf)

    def predict(self, X):
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.clfs]
        y_pred = np.mean(clf_preds, axis=0)
        y_pred = np.sign(y_pred)
        return y_pred
