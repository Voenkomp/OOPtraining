import pandas as pd
from sklearn.datasets import make_regression

X, y = make_regression(
    n_samples=1000, n_features=14, n_informative=10, noise=15, random_state=42
)
X = pd.DataFrame(X)
y = pd.Series(y)

X.columns = [f"col_{col}" for col in X.columns]
print(X.columns)


class MyLineReg:
    def __init__(self, n_iter, learning_rate):
        self.n_iter = n_iter
        self.learning_rate = learning_rate

    def __str__(self):
        return (
            f"MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"
        )


object = MyLineReg(100, 0.5)

print(object)
