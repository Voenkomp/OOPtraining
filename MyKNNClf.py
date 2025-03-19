import pandas as pd
import numpy as np


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
        self.X = X
        self.y = y
        self.train_size = self.X.shape

    def predict(self, X: pd.DataFrame):
        vector = []
        for idx in range(X.shape[0]):
            distance = self.X.apply(
                lambda X: np.sqrt(np.sum((X - X.iloc[idx]) ** 2)), axis=0
            )
            min_distance = distance.nsmallest(self.k).index
            class_y = self.y.iloc[min_distance]
            count1 = sum(class_y)
            count0 = len(class_y) - count1
            vector.append(1) if count1 >= count0 else vector.append(0)
        return pd.Series(vector)

    def predict_proba(self, X: pd.DataFrame):
        return pd.Series([0, 1, 2])


obj1 = MyKNNClf(5)

print(obj1)
