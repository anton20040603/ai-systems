# LR_3_task_4.py
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# Завантаження даних
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

# Розбивка
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.5, random_state=0)

# Модель
regr = linear_model.LinearRegression()
regr.fit(Xtrain, ytrain)
ypred = regr.predict(Xtest)

# Метрики
print("Коефіцієнти регресії:", regr.coef_)
print("Вільний член (intercept):", round(regr.intercept_, 2))
print("R2 score:", round(r2_score(ytest, ypred), 2))
print("Mean absolute error (MAE):", round(mean_absolute_error(ytest, ypred), 2))
print("Mean squared error (MSE):", round(mean_squared_error(ytest, ypred), 2))

# Графік
fig, ax = plt.subplots()
ax.scatter(ytest, ypred, edgecolors=(0, 0, 0), alpha=0.7)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Виміряно')
ax.set_ylabel('Передбачено')
ax.set_title('Лінійна регресія — набір даних діабет')
plt.tight_layout()
plt.show()
