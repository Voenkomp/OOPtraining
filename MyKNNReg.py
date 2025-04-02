import pandas as pd
import numpy as np


class MyKNNReg:
    def __init__(self, k: int = 3, metric: str = "euclidean", weight: str = "uniform"):
        self.k = k
        self.train_size = None
        self.metric = metric
        self.weight = weight

    def __repr__(self):
        return f"MyKNNReg class: k={self.k}"

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.X_train = X
        self.y_train = y
        self.train_size = X.shape

    def _euclidean(self, row: pd.Series) -> pd.Series:
        return np.sqrt(((row - self.X_train) ** 2).sum(axis=1))

    def _chebyshev(self, row: pd.Series) -> pd.Series:
        return (np.abs(row - self.X_train)).max(axis=1)

    def _manhattan(self, row: pd.Series) -> pd.Series:
        return np.abs(row - self.X_train).sum(axis=1)

    def _cosine(self, row: pd.Series) -> pd.Series:
        numerator = (row * self.X_train).sum(axis=1)
        denominator = np.sqrt(np.sum((row**2))) * np.sqrt((self.X_train**2).sum(axis=1))
        return 1 - (numerator / denominator)

    def get_metrics(self, row):
        dist_min = getattr(self, "_" + self.metric)(row).sort_values().head(self.k)
        if self.weight == "rank":
            rank = pd.Series(range(1, self.k + 1))
            weights = (1 / rank) / (1 / rank).sum()
            return np.dot(weights, self.y_train[dist_min.index])
        elif self.weight == "distance":
            weights = (1 / dist_min) / (1 / dist_min).sum()
            return np.dot(weights, self.y_train[dist_min.index])
        return self.y_train[dist_min.index].mean()

    def predict(self, X: pd.DataFrame):
        return X.apply(self.get_metrics, axis=1)
