import pandas as pd
import numpy as np

a = np.array([0.34, 0.27, 0.83, 0.24, 0.5])
b = pd.Series([1, 1, 1, 0, 0])

new_arr = np.column_stack((np.arange(1, len(a) + 1), np.sort(a), b))

print(new_arr)
print(new_arr[:, 1])
