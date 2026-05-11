# LR_3_task_6.py

import numpy as np

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Варіант 6
np.random.seed(42)

m = 100
X = np.linspace(-3, 3, m)
y = 2 * np.sin(X) + np.random.uniform(-0.6, 0.6, m)

# Перетворюємо X у 2D-масив
X = X.reshape(-1, 1)

# Функція побудови кривих навчання
def plot_learning_curves(model, X, y, title=''):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_errors = []
    val_errors = []

    for m_size in range(2, len(X_train)):
        model.fit(X_train[:m_size], y_train[:m_size])

        y_train_predict = model.predict(X_train[:m_size])
        y_val_predict = model.predict(X_val)

        train_errors.append(
            mean_squared_error(y_train[:m_size], y_train_predict)
        )

        val_errors.append(
            mean_squared_error(y_val, y_val_predict)
        )

    plt.plot(np.sqrt(train_errors), 'r-+', linewidth=2, label='train')
    plt.plot(np.sqrt(val_errors), 'b-', linewidth=3, label='val')

    plt.xlabel('Розмір навчального набору')
    plt.ylabel('RMSE')
    plt.title(title)
    plt.legend()
    plt.grid(True)


# Лінійна регресія
plt.figure(figsize=(14, 4))

plt.subplot(1, 3, 1)
lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X, y, 'Лінійна регресія')

# Поліноміальна degree=10
plt.subplot(1, 3, 2)
poly10 = Pipeline([
    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
    ("lin_reg", LinearRegression()),
])
plot_learning_curves(poly10, X, y, 'Поліноміальна degree=10')

# Поліноміальна degree=2
plt.subplot(1, 3, 3)
poly2 = Pipeline([
    ("poly_features", PolynomialFeatures(degree=2, include_bias=False)),
    ("lin_reg", LinearRegression()),
])
plot_learning_curves(poly2, X, y, 'Поліноміальна degree=2')

plt.tight_layout()
plt.show()