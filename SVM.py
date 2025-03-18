import pandas as pd
import numpy as np
import random

from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42
)
X, y = pd.DataFrame(X), pd.Series(y)


class MySVM:
    def __init__(
        self,
        n_iter: int = 10,
        learning_rate: float = 0.001,
        C: int = 1,
        sgd_sample=None,
        random_state: int = 42,
    ):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None
        self.b = None
        self.C = C
        self.sgd_sample = sgd_sample
        self.random_state = random_state

    def __repr__(self):
        return f"MySVM class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose=False):
        # Если True, то элемент не меняется, в противном случае -1
        y = y.where(y == 1, -1)
        self.weights = np.ones(X.shape[1])
        self.b = 1

        random.seed(self.random_state)

        if isinstance(self.sgd_sample, float) and 0 < self.sgd_sample <= 1:
            self.sgd_sample = int(round(self.sgd_sample * X.shape[0]))

        for i in range(self.n_iter):

            """Calculate LOSS FUNCTION"""
            if verbose and ((i + 1) % verbose == 0 or i == 0):
                hinge_loss = 0
                for idx in range(X.shape[0]):
                    X_i, y_i = X.iloc[idx], y.iloc[idx]
                    hinge_loss += max(0, 1 - y_i * (self.weights @ X_i + self.b))
                LOSS_F = np.sum(self.weights**2) + (self.C * hinge_loss) / len(y)
                print(f"{i + 1} | loss: {LOSS_F}")
            """End calculate"""

            if self.sgd_sample:
                sample_rows_idx = random.sample(range(X.shape[0]), self.sgd_sample)
            else:
                sample_rows_idx = range(X.shape[0])

            for idx in sample_rows_idx:
                X_i, y_i = X.iloc[idx], y.iloc[idx]

                if y_i * (self.weights @ X_i + self.b) >= 1:
                    gradient_w = 2 * self.weights
                    gradient_b = 0
                else:
                    gradient_w = 2 * self.weights - self.C * y_i * X_i
                    gradient_b = -(self.C * y_i)

                self.weights -= self.learning_rate * gradient_w
                self.b -= self.learning_rate * gradient_b

    def get_coef(self):
        return self.weights, self.b

    def predict(self, X: pd.DataFrame):
        y_pred = np.sign(X @ self.weights + self.b)
        return y_pred.where(y_pred == 1, 0).astype(int)


obj1 = MySVM(n_iter=10, learning_rate=0.05)
obj1.fit(X, y)
print(obj1.get_coef())
