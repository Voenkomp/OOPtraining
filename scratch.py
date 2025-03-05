import pandas as pd
import numpy as np

a = np.array([0.34, 0.27, 0.24, 0.24, 0.5])
b = pd.Series([1, 1, 1, 0, 0])

new_arr = np.column_stack((a, b))
sort_new_arr = new_arr[new_arr[:, 1].argsort()][::-1]

print(new_arr)
print(sort_new_arr)
print(sort_new_arr[:4][sort_new_arr[:4, 0] != sort_new_arr[4, 0]])
# for i in range(len(sort_new_arr)):
#     print(sort_new_arr[: i + 1, 0])
