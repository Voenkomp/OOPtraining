import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=50, n_features=5, n_informative=2, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f"col_{col}" for col in X.columns]


class MyKNNClf:
    def __init__(self, k: int = 3, metric: str = "euclidean"):
        self.k = k
        self.train_size = None
        self.X = None
        self.y = None
        self.metric = metric

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

    def euclid_mean_predict(self, row: pd.Series) -> int:
        mean = self.euclid_mean(row)
        return 1 if mean >= 0.5 else 0

    def predict(self, X_test: pd.DataFrame):
        return X_test.apply(self.euclid_mean_predict, axis=1)

    def euclid_mean(self, row: pd.Series) -> float:
        dist_min_ind = (
            getattr(self, "_" + self.metric + "_distance")(row)
            .sort_values()
            .head(self.k)
            .index
        )
        return self.y_train[dist_min_ind].mean()

    def predict_proba(self, X_test: pd.DataFrame):
        return X_test.apply(self.euclid_mean, axis=1)


obj1 = MyKNNClf(1)
obj1.fit(X, y)

print(obj1.train_size)
