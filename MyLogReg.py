import pandas as pd
import numpy as np
from sklearn.datasets import make_regression

X, y = make_regression(
    n_samples=1000, n_features=14, n_informative=10, noise=15, random_state=42
)
X = pd.DataFrame(X)
y = pd.Series(y)

X.columns = [f"col_{col}" for col in X.columns]


class MyLogReg:
    def __init__(self, n_iter=10, learning_rate=0.1):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None

    def __str__(self):
        return (
            f"MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"
        )

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose=False):

        X.insert(
            loc=0, column="ones", value=1
        )  # добавление в матрицу фичей единичного столбца
        W = np.ones(X.shape[1])

        for i in range(1, self.n_iter + 1):
            y_hat = 1 / (1 + np.e ** (np.dot(X, W)))  # предсказываю значения y
            Logloss = (-1 / len(y)) * np.sum(
                y * np.log10(y_hat + 10**-15) + (1 - y) * np.log10(1 - y_hat + 10**-15)
            )  # расчет функции потерь
            grad = 1 / len(y) * np.dot((y_hat - y), X)
            W -= self.learning_rate * grad
            self.weights = W

    def get_coef(self):
        return np.array(self.weights)


object1 = MyLogReg(50, 0.1)

object1.fit(X, y)
print(np.mean(object1.get_coef()))
