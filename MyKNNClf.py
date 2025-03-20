import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=50, n_features=5, n_informative=2, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f"col_{col}" for col in X.columns]


class MyKNNClf:
    def __init__(self, k: int = 3):
        self.k = k
        self.train_size = None
        self.X = None
        self.y = None

    def __repr__(self):
        return f"MyKNNClf class: k={self.k}"

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Здесь хранится обучающая\тренировочная выборка"""
        self.X_train = X
        self.y_train = y
        self.train_size = self.X_train.shape

    def euclid_mean_predict(self, row: pd.Series) -> int:
        mean = self.euclid_mean(row)
        return 1 if mean >= 0.5 else 0

    def predict(self, X_test: pd.DataFrame):
        return X_test.apply(self.euclid_mean_predict, axis=1)

    def euclid_mean(self, row: pd.Series) -> float:
        dist_min_ind = (
            np.sqrt(((row - self.X_train) ** 2).sum(axis=1))
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
