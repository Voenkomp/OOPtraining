import pandas as pd
import numpy as np
import random
from sklearn.datasets import make_regression

X, y = make_regression(
    n_samples=1000, n_features=14, n_informative=10, noise=15, random_state=42
)
X = pd.DataFrame(X)
y = pd.Series(y)

X.columns = [f"col_{col}" for col in X.columns]


class MyLineReg:
    def __init__(
        self,
        n_iter,
        learning_rate,
        metric: str = None,
        reg=None,
        l1_coef: float = 0,
        l2_coef: float = 0,
        sgd_sample=None,
        random_state=42,
    ):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None
        self.metric = metric
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state

    def __str__(self):
        return (
            f"MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"
        )

    def get_metric(self, y, y_hat):
        metrics = {
            "mae": np.sum(abs(y - y_hat)) / len(y),
            "mse": np.sum((y_hat - y) ** 2) / len(y),
            "rmse": np.sqrt(np.sum((y_hat - y) ** 2) / len(y)),
            "mape": np.sum(np.abs((y - y_hat) / y)) * (100 / len(y)),
            "r2": 1 - (np.sum(((y - y_hat) ** 2)) / np.sum(((y - np.mean(y)) ** 2))),
        }
        return metrics[self.metric] if self.metric else metrics["mse"]

    def get_reg(self, W):
        if self.reg:
            dict_reg = {
                "l1": (self.l1_coef * np.sum(np.abs(W)), self.l1_coef * np.sign(W)),
                "l2": (self.l2_coef * np.sum(W**2), self.l2_coef * 2 * W),
                "elasticnet": (
                    self.l1_coef * np.sum(np.abs(W)) + self.l2_coef * np.sum(W**2),
                    self.l1_coef * np.sign(W) + self.l2_coef * 2 * W,
                ),
            }
            return dict_reg[self.reg]
        else:
            return 0, 0

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: int = False):

        random.seed(
            self.random_state
        )  # фиксируем рандом, чтобы выборка была одинаковая

        X.insert(0, "ones", 1)  # дополнение вектора фичей единичным столбцом
        W = np.ones(X.shape[1])  # инициализация вектора весов соответствующей длины

        if (
            isinstance(self.sgd_sample, float) and 0 < self.sgd_sample <= 1
        ):  # если введена дробь, то рассчитывается кол-во строк от этой доли
            self.sgd_sample = int(round(self.sgd_sample * X.shape[0]))
            # print(self.sgd_sample)

        for i in range(1, self.n_iter + 1):

            y_hat = np.dot(X, W)  # вычисление предсказаний
            reg_part_loss, reg_part_grad = self.get_reg(W)

            LOSS_F = self.get_metric(y, y_hat) + reg_part_loss  # расчет функции потерь

            if verbose:
                if i == 1:
                    if self.metric:
                        print(
                            f"start | loss: {LOSS_F} | {self.metric}: {self.get_metric(y, y_hat)}"
                        )
                    else:
                        print(f"start | loss: {LOSS_F}")
                elif i % verbose == 0:
                    if self.metric:
                        print(
                            f"{i} | loss: {LOSS_F} | {self.metric}: {self.get_metric(y, y_hat)}"
                        )
                    else:
                        print(f"{i} | loss: {LOSS_F}")

            if self.sgd_sample:
                sample_rows_idx = random.sample(range(X.shape[0]), self.sgd_sample)

                grad = (
                    2
                    / len(sample_rows_idx)
                    * np.dot(
                        (y_hat[sample_rows_idx] - y.iloc[sample_rows_idx]),
                        X.iloc[sample_rows_idx],
                    )
                    + reg_part_grad
                )
            else:
                grad = 2 / len(y) * np.dot((y_hat - y), X) + reg_part_grad

            if callable(self.learning_rate):
                W -= self.learning_rate(i) * grad
            else:
                W -= self.learning_rate * grad

            self.weights = W

    def get_coef(self):
        return self.weights[1:]

    def predict(self, X: pd.DataFrame):
        # X.insert(0, 'ones', 1)
        return np.dot(X, self.weights)

    def get_best_score(self):
        last_metric = self.get_metric(y, self.predict(X))
        return last_metric


object = MyLineReg(100, 0.01, sgd_sample=0.1)
object.fit(X, y, 10)
print(np.sum(object.get_coef()))
