import pandas as pd
import numpy as np

x = pd.DataFrame({"col1": [1, 2, 3, 4, 5], "col2": [6, 7, 8, 9, 10]})
y = pd.Series([6, 7, 8, 9, 10])

print(x.idxmax(axis="index"))
