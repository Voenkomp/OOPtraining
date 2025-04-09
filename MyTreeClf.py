import pandas as pd
import numpy as np


X = pd.DataFrame([8, 9, 15, 2, 38, 1, 8, 4, 15])
y = pd.Series([1, 2, 3])


class MyTreeClf:
    def __init__(
        self, max_depth: int = 5, min_samples_split: int = 2, max_leafs: int = 20
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs

    def __repr__(self):
        return f"MyTreeClf class: max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}"

    def get_entropy(self, group):
        eps = 1e-15
        return -(
            (group.sum() / group.shape[0])
            * np.log2((group.sum() + eps) / group.shape[0])
            + (
                ((group.shape[0] - group.sum()) / group.shape[0])
                * np.log2(((group.shape[0] - group.sum()) + eps) / group.shape[0])
            )
        )

    def get_sum_entropy(self, elem, X, y, col):
        left_group, right_group = y[X[col] <= elem], y[X[col] > elem]
        left_entropy = (left_group.shape[0] / X.shape[0]) * self.get_entropy(left_group)
        right_entropy = (right_group.shape[0] / X.shape[0]) * self.get_entropy(
            right_group
        )
        return left_entropy + right_entropy

    def get_best_split(self, X: pd.DataFrame, y: pd.Series):
        best_split = ("", 0, 0)
        for col in X.columns:
            unique_splits = (
                X[col].drop_duplicates().sort_values().rolling(window=2).mean()[1:]
            )
            S0 = -(
                (y.sum() / y.shape[0]) * np.log2(y.sum() / y.shape[0])
                + ((y.shape[0] - y.sum()) / y.shape[0])
                * np.log2((y.shape[0] - y.sum()) / y.shape[0])
            )
            best_split_col = pd.DataFrame(
                {
                    "un_spl": unique_splits,
                    "ig": S0 - unique_splits.apply(self.get_sum_entropy, X, y, col),
                }
            )
            max_split_ind = best_split_col["ig"].max().index

            if best_split_col["ig"].iloc[max_split_ind] > best_split[2]:
                best_split = (col, best_split_col.iloc[max_split_ind].to_list())
                best_split = (
                    col,
                    best_split_col[best_split_col["ig"] == best_split_col["ig"].max()][
                        "un_spl"
                    ],
                    best_split_col["ig"].max(),
                )
        return best_split


obj1 = MyTreeClf()
obj1.get_best_split(X, y)

print(obj1)
