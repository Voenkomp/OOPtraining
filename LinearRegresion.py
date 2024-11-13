import pandas as pd
import numpy as np
from sklearn.datasets import make_regression

X, y = make_regression(
    n_samples=1000, n_features=14, n_informative=10, noise=15, random_state=42
)
X = pd.DataFrame(X)
y = pd.Series(y)

X.columns = [f"col_{col}" for col in X.columns]


class MyLineReg:
    def __init__(
        self, n_iter, learning_rate, metric: str = None, reg=None, l1_coef=0, l2_coef=0
    ):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None
        self.metric = metric
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef

    def __str__(self):
        return (
            f"MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"
        )

    def get_metric(self, metric, y, y_hat):
        metrics = {
            "mae": np.sum(abs(y - y_hat)) / len(y),
            "mse": np.sum((y_hat - y) ** 2) / len(y),
            "rmse": np.sqrt(np.sum((y_hat - y) ** 2) / len(y)),
            "mape": np.sum(np.abs((y - y_hat) / y)) * (100 / len(y)),
            "r2": 1 - (np.sum(((y - y_hat) ** 2)) / np.sum(((y - np.mean(y)) ** 2))),
        }
        return metrics[metric]

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: int = False):
        # X.insert(0, 'ones', 1) #дополнение вектора фичей единичным столбцом
        W = np.ones(X.shape[1])  # инициализация вектора весов соответствующей длины

        for i in range(self.n_iter):
            y_hat = np.dot(X, W)  # вычисление предсказаний

            MSE = self.get_metric(
                "mse", y, y_hat
            )  # вычисление метрики MSE (в данном случае MSE)

            if verbose:
                if i == 0:
                    if self.metric:
                        print(
                            f"start | loss: {MSE} | {self.metric}: {self.get_metric(self.metric, y, y_hat)}"
                        )
                    else:
                        print(f"start | loss: {MSE}")
                elif (i + 1) % verbose == 0:
                    if self.metric:
                        print(
                            f"{i} | loss: {MSE} | {self.metric}: {self.get_metric(self.metric, y, y_hat)}"
                        )
                    else:
                        print(f"{i} | loss: {MSE}")

            grad = 2 / len(y) * np.dot((y_hat - y), X)
            W -= self.learning_rate * grad
            self.weights = W

    def get_coef(self):
        return self.weights[1:]

    def predict(self, X: pd.DataFrame):
        # X.insert(0, 'ones', 1)
        return np.dot(X, self.weights)

    def get_best_score(self):
        last_metric = self.get_metric(self.metric, y, self.predict(X))
        return last_metric


object = MyLineReg(50, 0.1, metric="mai")
object.fit(X, y, 10)
object.get_best_score()
