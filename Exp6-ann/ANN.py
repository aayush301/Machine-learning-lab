import numpy as np


class ANN:
    def __init__(self):
        self.wts = []
        self.bias = 0

    def sigmoid(self, X):
        return 1/(1+np.exp(-X))

    def fit(self, X, Y, epochs, loss_threshold):
        self.wts, self.bias = self.gradient_descent(
            X, Y, epochs, loss_threshold)

    def gradient_descent(self, X, y_true, epochs, loss_threshold):
        wts = np.zeros(X.shape[1])
        bias = 0
        rate = 0.5  # learning rate

        for i in range(epochs):
            weighted_sum = np.dot(X, wts) + bias
            y_predicted = self.sigmoid(weighted_sum)
            y_diff = y_predicted - y_true

            wts_d = 1/(len(X)) * (np.dot(np.transpose(X), y_diff))
            bias_d = np.mean(y_diff)

            wts = wts - rate * wts_d
            bias = bias - rate * bias_d

            loss = self.log_loss(y_true, y_predicted)
            print(f"Epoch: {i}, loss: {loss}")
            if(loss <= loss_threshold):
                break

        return wts, bias

    def log_loss(self, y_true, y_predicted):
        epsilon = 1e-15
        y_predicted_new = [max(x, epsilon) for x in y_predicted]
        y_predicted_new = [min(x, 1-epsilon) for x in y_predicted_new]
        y_predicted_new = np.array(y_predicted_new)
        return -np.mean(y_true * np.log(y_predicted_new) + (1-y_true)*np.log(1-y_predicted_new))

    def predict(self, X_test):
        weighted_sum = np.dot(X_test, self.wts) + self.bias
        return self.sigmoid(weighted_sum)

    def get_weights(self):
        return (self.wts, self.bias)
