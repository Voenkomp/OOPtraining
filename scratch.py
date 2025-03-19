import pandas as pd

check = pd.Series([1, 2, 3, 4, 5, 6])

check_ind = check.nsmallest(3).index
result = check[1]

print(check_ind)
print(result)
