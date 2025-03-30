import pandas as pd
import numpy as np

y = pd.Series([1, 2, 3, 4, 5])

y_test = y[2 < y]
y_test.reset_index(drop=True)
y_test.index += 1

print(y_test)
print(y_test.shape[0])
