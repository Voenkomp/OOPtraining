import pandas as pd
import numpy as np
from MyLogReg_through_pandas import MyLogReg

a = np.array([0.34, 0.27, 0.24, 0.24, 0.5])
b = pd.Series([1, 1, 1, 0, 0])

new_arr = np.column_stack((a, b))
sort_new_arr = new_arr[new_arr[:, 1].argsort()][::-1]

# print(new_arr)
# print(sort_new_arr)
# print(sort_new_arr[:4][sort_new_arr[:4, 0] != sort_new_arr[4, 0]])
# for i in range(len(sort_new_arr)):
#     print(sort_new_arr[: i + 1, 0])

y = pd.Series(
    [1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
)
y_hat = pd.Series([1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1])
score = pd.DataFrame(
    {"score": [0.91, 0.86, 0.78, 0.6, 0.6, 0.55, 0.51, 0.46, 0.45, 0.45, 0.42]}
)

y_score = pd.concat([y, score], axis=1)
y_score_sort = y_score.sort_values(by=0, ascending=False)

# print(MyLogReg.accuracy(y, y_hat))
# print(((y == 1) & (y_hat == 1)).sum())
print(y_score)
# print(y_score_sort)


class Test:
    def summa(a, b):
        return a + b


a, b = 2, 7

ex = Test()
# print(Test.summa(a, b))

# print(MyLogReg.roc_auc(y, score))

# pd_series = pd.Series([1, 2, 3, 4, 5])
# pd_series.index = [i for i in range(2, 7)]
# print(pd_series)
# print(type(pd_series))
# print(pd_series.values)
# print(pd_series.index)
# print(pd_series[2])


# pd_dataframe = pd.DataFrame(
#     {
#         "country": ["Russia", "Kazahstan", "Belarus", "Japan"],
#         "population": [142.04, 25.9, 10, 100],
#         "square": [17.1, 6, 4, 3],
#     },
#     index=["RU", "KZ", "BY", "JP"],
# )
# print(pd_dataframe)
# print("Строка выведена через loc:\n", pd_dataframe.loc["RU"])
# print("Строка выведена через iloc:\n", pd_dataframe.iloc[0])
# print()
# print(pd_dataframe.loc["RU":"BY", "population"])
# print()
# print("Фильтрация с помощью булевых массивов: ", pd_dataframe[pd_dataframe.square >= 6])
