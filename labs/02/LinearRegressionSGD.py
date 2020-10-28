from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class LinearRegressionSGD(BaseEstimator):
    def get_params(self, deep=True):
        return super().get_params(deep)

    def set_params(self, **params):
        return super().set_params(**params)

    def __init__(self, epochs=1000, batch_size=50, alpha=1, learning_rate=0.01, random_state=None):
        self.random_state = random_state
        self.epochs = epochs
        self.batch_size = batch_size
        self.alpha = alpha
        self.learning_rate = learning_rate

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        generator = np.random.RandomState(self.random_state)
        self.weights_ = generator.uniform(size=X.shape[1])
        for _ in range(self.epochs):
            permutation = generator.permutation(X.shape[0])

            for i in range(int(np.ceil(X.shape[0] / self.batch_size))):
                batch_data = X[permutation[i * self.batch_size:(i + 1) * self.batch_size]]
                batch_target = y[permutation[i * self.batch_size:(i + 1) * self.batch_size]]
                batch_prediction = batch_data @ self.weights_
                gradient = (batch_data.T @ (batch_prediction - batch_target) + self.alpha * self.weights_) \
                           / batch_data.shape[0]
                self.weights_ = self.weights_ - self.learning_rate * gradient
                if mean_squared_error(batch_target, batch_prediction, squared=False) > 10000:
                    return self
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return X @ self.weights_

    def score(self, X, y):
        return -mean_squared_error(y, self.predict(X), squared=False)
