import pandas as pd


class MyKNNClf:
    def __init__(self, k: int = 3):
        self.k = k
        self.train_size = None
        self.X = None
        self.y = None

    def __repr__(self):
        return f"MyKNNClf class: k={self.k}"

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y
        self.train_size = self.X.shape

    def predict(self):

    

obj1 = MyKNNClf(5)

print(obj1)
