import pandas as pd
import numpy as np
from sklearn.datasets import make_regression

"""Этот код является имплементацией алгоритма машинного обучения логистической регрессии"""

X, y = make_regression(
    n_samples=400, n_features=14, n_informative=5, noise=15, random_state=42
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
        self.weights = np.ones(X.shape[1])  # заполнение вектора весов единицами
        eps = 1e-15

        for i in range(1, self.n_iter + 1):

            y_hat = 1 / (
                1 + np.exp(-(X @ self.weights))
            )  # расчет предсказанных значений (y_hat)

            Logloss = -np.mean(
                y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps)
            )

            if verbose and (i == 1 or i % verbose) == 0:
                print(f"{i} | loss: {Logloss}")

            grad = 1 / len(y) * np.dot((y_hat - y), X)  # расчет градиента
            self.weights -= self.learning_rate * grad  # обновление весов

    def get_coef(self):
        return self.weights[1:]

    """возращает вероятности"""

    def predict_proba(self, X: pd.DataFrame):
        if X.columns[0] != "ones":
            X.insert(loc=0, column="ones", value=1)
        return 1 / (1 + np.exp(-(X @ self.weights)))

    """переводит вероятности в бинарные классы"""

    def predict(self, X: pd.DataFrame):
        return np.where(self.predict_proba(X) > 0.5, 1, 0)


object1 = MyLogReg(n_iter=50, learning_rate=0.1)

object1.fit(X, y)

# print(np.mean(object1.get_coef()))
# print(object1.predict_proba(X))
print(np.mean(object1.predict_proba(X)))
print(sum(object1.predict(X)))