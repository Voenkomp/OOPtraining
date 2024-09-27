import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, method="simple"):
        self.method = method
        self.a = None

    def fit(self, X, y):
        if self.method == "simple":
            self.a = sum(y / X) / len(X)
        elif self.method == "MNK":
            self.a = sum(y * X) / sum(X**2)

    def predict(self, X):
        if self.a == None:
            print("Model not trained yet")
        else:
            print(self.a)


class ModelEvaluation:
    @staticmethod
    def mean_square_error(model, y, X):
        pass


# Генерирование данных
X = np.random.rand(100, 1)

a = 1  # коэффициент для создания тестовых данных

noise = np.random.rand(100, 1) - 0.5
r = 0.1  # амплитуда шума

y = np.dot(X, a) + np.dot(noise, r)

# Тренировка моделей
model_simple = LinearRegression("simple")
model_simple.fit(X, y)
print(model_simple.a)
model_MNK = LinearRegression("MNK")
model_MNK.fit(X, y)
print(model_MNK.a)


# Создаем данные для линии 1
x1 = np.linspace(0, 1, 100)
y1 = model_simple.predict(x1)

# Создаем данные для линии 2
x2 = np.linspace(0, 1, 100)
y1 = model_MNK.predict(x2)