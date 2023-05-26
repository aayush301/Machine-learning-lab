import numpy as np


class LinearRegression:
    def fit(self, X, y):
        n = np.size(X)
        mean_x = np.mean(X)
        mean_y = np.mean(y)

        num = np.sum(y*X) - n*mean_y*mean_x
        deno = np.sum(X*X) - n*mean_x*mean_x

        b1 = num / deno
        b0 = mean_y - b1*mean_x

        self.intercept_ = b0
        self.coef_ = b1

    def predict(self, X):
        return X*self.coef_ + self.intercept_
