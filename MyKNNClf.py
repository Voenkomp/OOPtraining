import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=50, n_features=5, n_informative=2, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f"col_{col}" for col in X.columns]


class MyKNNClf:
    def __init__(self, k: int = 3, metric: str = "euclidean", weight: str = "uniform"):
        self.k = k
        self.train_size = None
        self.X = None
        self.y = None
        self.metric = metric
        self.weight = weight

    def __repr__(self):
        return f"MyKNNClf class: k={self.k}"

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Здесь хранится обучающая\тренировочная выборка"""
        self.X_train = X
        self.y_train = y
        self.train_size = self.X_train.shape

    def _euclidean_distance(self, row):
        return np.sqrt(((row - self.X_train) ** 2).sum(axis=1))

    def _chebyshev_distance(self, row):
        return (np.abs(row - self.X_train)).max(axis=1)

    def _manhattan_distance(self, row):
        return np.abs(row - self.X_train).sum(axis=1)

    def _cosine_distance(self, row):
        numerator = (row * self.X_train).sum(axis=1)
        denominator = np.sqrt(np.sum((row**2))) * np.sqrt((self.X_train**2).sum(axis=1))
        return 1 - (numerator / denominator)

    def get_metrics_predict(self, row: pd.Series) -> int:
        class_index = self.get_metrics(row)
        if self.weight != "uniform":
            return 1 if class_index >= 0.5 else 0
        return 1 if class_index.mean() >= 0.5 else 0

    def predict(self, X_test: pd.DataFrame):
        return X_test.apply(self.get_metrics_predict, axis=1)

    def get_metrics(self, row: pd.Series) -> pd.Series:
        dist_min = (
            getattr(self, "_" + self.metric + "_distance")(row)
            .sort_values()
            .head(self.k)
        )
        if self.weight == "rank":
            ind_class_sort = self.y_train[dist_min.index].reset_index(drop=True)
            ind_class_sort.index += 1
            q_class1 = (
                1 / (pd.Series(ind_class_sort[ind_class_sort == 1].index + 1))
            ).sum()
            denominator = (1 / pd.Series(range(1, ind_class_sort.shape[0] + 1))).sum()
            return q_class1 / denominator
        elif self.weight == "distance":
            q_class1 = (1 / dist_min[self.y_train[dist_min.index] == 1]).sum()
            denominator = (1 / dist_min).sum()
            return q_class1 / denominator
        return self.y_train[dist_min.index]

    def predict_proba(self, X_test: pd.DataFrame):

        prediction = X_test.apply(self.get_metrics, axis=1)
        if self.weight != "uniform":
            return prediction
        return prediction.mean(axis=1)


obj1 = MyKNNClf(1)
obj1.fit(X, y)

print(obj1.train_size)
