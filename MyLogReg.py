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
    def __init__(
        self, n_iter: int = 10, learning_rate: float = 0.1, metric: str = None
    ):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None
        self.metric = metric

    def __str__(self):
        return (
            f"MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"
        )

    @staticmethod
    def accuracy(y: pd.Series, y_hat: pd.Series):
        return np.sum(np.where(y == y_hat, 1, 0)) / len(y)

    @staticmethod
    def precision(y: pd.Series, y_hat: pd.Series):
        last_metric = np.sum(np.where((y == 1) & (y_hat == 1), 1, 0)) / (
            np.sum(
                np.where(((y == 1) & (y_hat == 1)) | ((y == 0) & (y_hat == 1)), 1, 0)
            )
        )
        return last_metric

    @staticmethod
    def recall(y: pd.Series, y_hat: pd.Series):
        last_metric = np.sum(np.where((y == 1) & (y_hat == 1), 1, 0)) / (
            np.sum(
                np.where(((y == 1) & (y_hat == 1)) | ((y == 1) & (y_hat == 0)), 1, 0)
            )
        )
        return last_metric

    @staticmethod
    def f1(y: pd.Series, y_hat: pd.Series):
        precision = MyLogReg.precision(y, y_hat)
        recall = MyLogReg.recall(y, y_hat)
        return 2 * ((precision * recall) / (precision + recall))

    @staticmethod
    def roc_auc(y: pd.Series, score: np):
        positive = np.sum(y)
        negative = len(y) - positive
        print(f"P: {positive}, N: {negative}")
        round_score = np.round(score, 10)
        round_score_y = np.column_stack((round_score, y))
        sort_round_score_y = round_score_y[round_score_y[:, 0].argsort()][::-1]
        print(sort_round_score_y)
        print("Начало отсчета")
        total = 0
        for i in range(len(sort_round_score_y)):
            if sort_round_score_y[i, 1] == 0:
                upper_positive = np.sum(
                    sort_round_score_y[:i, 1][
                        sort_round_score_y[:i, 0] != sort_round_score_y[i, 0]
                    ]
                )
                print(f"Положительные классы, выше по скору {upper_positive}")
                equal_positive = (
                    np.sum(
                        sort_round_score_y[:, 1][
                            sort_round_score_y[:, 0] == sort_round_score_y[i, 0]
                        ]
                    )
                    / 2
                )
                print(
                    f"Сумма классов с таким же скором деленная на два {equal_positive}"
                )
                total += upper_positive + equal_positive
        return total / (positive * negative)

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: bool = False):

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
                if self.metric:
                    print(f"{i} | loss: {Logloss} | {self.metric}: ")
                print(f"{i} | loss: {Logloss}")

            grad = 1 / len(y) * np.dot((y_hat - y), X)  # расчет градиента
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


# object1 = MyLogReg(n_iter=50, learning_rate=0.1, metric="precision")

# object1.fit(X, y)

# print(np.mean(object1.get_coef()))
# print(object1.predict_proba(X))
# print(np.mean(object1.predict_proba(X)))
# print(sum(object1.predict(X)))
# print(object1.get_best_score())
