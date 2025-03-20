import pandas as pd
import numpy as np

X = pd.DataFrame({"col1": [1, 2, 1, 4, 5], "col2": [6, 7, 8, 6, 10]})
X_test = pd.DataFrame({"col1": [1, 7, 3, 4, 5], "col2": [6, 5, 8, 9, 10]})

y = pd.Series([1, 1, 0, 0, 1])
k = 4
result = []
for idx in range(X_test.shape[0]):
    distance = np.sqrt(
        np.sum(X.apply(lambda X: (X - X_test.iloc[idx]) ** 2, axis=1), axis=1)
    )

    print(distance)

    min_ind = np.argpartition(distance, k)[:k]
    print(min_ind)
    class1 = np.sum(y[min_ind])
    class0 = len(y) - class1
    print(f"class1: {class1}")
    print(f"class0: {class0}")
    result.append(1) if class1 >= class0 else result.append(0)

print(result)
