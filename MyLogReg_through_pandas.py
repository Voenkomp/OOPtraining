import pandas as pd
import numpy as np
import random
from sklearn.datasets import make_regression

"""Этот код является имплементацией алгоритма машинного обучения логистической регрессии"""

X, y = make_regression(
    n_samples=400, n_features=14, n_informative=5, noise=15, random_state=42
)
X = pd.DataFrame(X)
y = pd.Series(y)

X.columns = [f"col_{col}" for col in X.columns]


class MyLogReg:
    def __init__(
        self,
        n_iter: int = 10,
        learning_rate: float = 0.1,
        metric: str = None,
        reg=None,
        l1_coef: float = 0,
        l2_coef: float = 0,
        sgd_sample=None,
        random_sate=42,
    ):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None
        self.metric = metric
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_sate

    def __str__(self):
        return (
            f"MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"
        )

    @staticmethod
    def accuracy(y: pd.Series, y_predict: pd.DataFrame):
        return (y == y_predict).mean()

    @staticmethod
    def precision(y: pd.Series, y_predict: pd.DataFrame):
        tp = ((y == 1) & (y_predict == 1)).sum()
        fp = ((y == 0) & (y_predict == 1)).sum()
        return tp / (tp + fp)

    @staticmethod
    def recall(y: pd.Series, y_predict: pd.DataFrame):
        tp = ((y == 1) & (y_predict == 1)).sum()
        fn = ((y == 1) & (y_predict == 0)).sum()
        return tp / (tp + fn)

    @staticmethod
    def f1(y: pd.Series, y_predict: pd.DataFrame):
        precision = MyLogReg.precision(y, y_predict)
        recall = MyLogReg.recall(y, y_predict)
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def roc_auc(y: pd.Series, score: pd.Series):
        score = score.round(10)  # округляем, чтобы пройти валидатор на степике
        score_y = pd.concat([score, y], axis=1)
        score_y = score_y.sort_values(by=0, ascending=False)
        positive = score_y[score_y[1] == 1]
        negative = score_y[score_y[1] == 0]

        total = 0
        for cur_score in negative[0]:
            score_higher = (positive[0] > cur_score).sum()
            score_equal = (positive[0] == cur_score).sum()
            total += score_higher + 0.5 * score_equal

        return total / (positive.shape[0] * negative.shape[0])

    def get_reg(self, weights):
        if self.reg:
            dict_reg = {
                "l1": (
                    self.l1_coef * np.sum(np.abs(weights)),
                    self.l1_coef * np.sign(weights),
                ),
                "l2": (self.l2_coef * np.sum(weights**2), self.l2_coef * 2 * weights),
                "elasticnet": (
                    self.l1_coef * np.sum(np.abs(weights))
                    + self.l2_coef * np.sum(weights**2),
                    self.l1_coef * np.sign(weights) + self.l2_coef * 2 * weights,
                ),
            }
            return dict_reg[self.reg]
        else:
            return 0, 0

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: bool = False):

        random.seed(self.random_state)

        if isinstance(self.sgd_sample, float) and 0 < self.sgd_sample <= 1:
            self.sgd_sample = int(round(self.sgd_sample * X.shape[0]))

        X.insert(
            loc=0, column="ones", value=1
        )  # добавление в матрицу фичей единичного столбца
        self.weights = np.ones(X.shape[1])  # заполнение вектора весов единицами
        eps = 1e-15

        for i in range(1, self.n_iter + 1):

            y_hat = 1 / (
                1 + np.exp(-(X @ self.weights))
            )  # расчет предсказанных значений (y_hat)

            reg_loss, reg_grad = self.get_reg(self.weights)

            Logloss = (
                -np.mean(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))
                + reg_loss
            )

            if verbose and (i == 1 or i % verbose == 0):
                if self.metric:
                    print(f"{i} | loss: {Logloss} | {self.metric}: ")
                print(f"{i} | loss: {Logloss}")

            if self.sgd_sample:
                sample_rows_idx = random.sample(range(X.shape[0]), self.sgd_sample)
                print(max(sample_rows_idx), min(sample_rows_idx))
                grad = (
                    1
                    / len(sample_rows_idx)
                    * np.dot(
                        (y_hat[sample_rows_idx] - y.iloc[sample_rows_idx]),
                        X.iloc[sample_rows_idx],
                    )
                    + reg_grad
                )
            else:
                grad = (
                    1 / len(y) * np.dot((y_hat - y), X) + reg_grad
                )  # расчет градиента

            if callable(self.learning_rate):
                self.weights -= self.learning_rate(i) * grad
            else:
                self.weights -= self.learning_rate * grad  # обновление весов

    def get_coef(self):
        return self.weights[1:]

    def get_best_score(self):
        if self.metric:
            if self.metric == "roc_auc":
                last_metric = getattr(self, self.metric)(y, self.predict_proba(X))
            else:
                last_metric = getattr(self, self.metric)(y, self.predict(X))
            return last_metric
        return "Вы вызвали метрику не указав ее в параметре 'metric'"

    """возращает вероятности"""

    def predict_proba(self, X: pd.DataFrame):
        if X.columns[0] != "ones":
            X.insert(loc=0, column="ones", value=1)
        return 1 / (1 + np.exp(-(X @ self.weights)))

    """переводит вероятности в бинарные классы"""

    def predict(self, X: pd.DataFrame):
        return np.where(self.predict_proba(X) > 0.5, 1, 0)


object1 = MyLogReg(n_iter=50, learning_rate=0.1, sgd_sample=0.1)

object1.fit(X, y)

print(np.mean(object1.get_coef()))
# print(object1.predict_proba(X))
# print(np.mean(object1.predict_proba(X)))
# print(sum(object1.predict(X)))
# print(object1.get_best_score())
