import pandas as pd
import numpy as np

X = pd.DataFrame({"column1": [1, 2, 3, 4, 5], "column2": [6, 7, 8, 9, 10]})
y = pd.Series([1, 0, 0, 1, 1])


class MySVM:
    def __init__(self, n_iter: int = 10, learning_rate: float = 0.001):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weight = None
        self.b = None

    def __repr__(self):
        return f"MySVM class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose=False):
        y = y.mask(y == 0, -1)
        return y
        # if verbose and i % verbose == 0:
        #     f'Добавить лог'


object1 = MySVM()
print(object1.fit(X, y))
print(object1)
str(object1)
