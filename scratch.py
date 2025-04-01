import pandas as pd
import numpy as np

x = pd.Series([1, 2, 3, 4, 5])
y = pd.Series([6, 7, 8, 9, 10])

print((x * y).sum())
print(np.dot(x, y))
