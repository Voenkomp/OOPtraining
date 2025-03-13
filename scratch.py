import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42
)
X, y = pd.DataFrame(X), pd.Series(y)


class MySVM:
    def __init__(self, n_iter: int = 10, learning_rate: float = 0.001):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None
        self.b = None

    def __repr__(self):
        return f"MySVM class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose=False):
        y = y.where(
            y == 1, -1
        )  # Если True, то элемент не меняется, в противном случае -1
        self.weights = np.ones(X.shape[1])
        self.b = 1

        # answer = (np.dot(self.weights, X.iloc[1]) + self.b) * y.iloc[1]
        # print(f"Формула дает такой ответ: {answer}")

        for i in range(self.n_iter):
            for idx in range(X.shape[0]):
                X_i, y_i = X.iloc[idx], y.iloc[idx]
                if y_i * (self.weights @ X_i + self.b) >= 1:
                    gradient_w = 2 * self.weights
                    gradient_b = 0
                else:
                    gradient_w = 2 * self.weights - y_i * X_i
                    gradient_b = -y_i

                self.weights -= self.learning_rate * gradient_w
                self.b -= self.learning_rate * gradient_b
            # LOSS_F = np.sum(self.weights**2) + np.sum(
            #     max([0, 1 - y.iloc[i] * (self.weights @ X.iloc[i] + self.b)])
            # ) / len(y)
            # if verbose and i % verbose == 0:
            #     f'Добавить лог'

    def get_coef(self):
        return self.weights, self.b

    def predict(self, X: pd.DataFrame):
        y_pred = np.sign(X @ self.weights + self.b)
        y_pred = y_pred.where(y_pred == 1, 0).astype(int)
        return y_pred


obj1 = MySVM(n_iter=10, learning_rate=0.05)

obj1.fit(X, y)

print(obj1.predict(X))
