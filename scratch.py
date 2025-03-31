import pandas as pd
import numpy as np

y = pd.Series([1, 2, 3, 4, 5])
distance = [1.27823921, 2.83950234, 3.81237458234]


y_test = y[2 < y]
y_test.index = distance

print(y_test)
print(y_test.shape[0])
