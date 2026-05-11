# LR_3_task_5.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('TkAgg')

# Варіант 6
np.random.seed(42)
m = 100
X = np.linspace(-3, 3, m)
y = 2 * np.sin(X) + np.random.uniform(-0.6, 0.6, m)

X = X.reshape(-1, 1)

# Лінійна регресія
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_lin_pred = lin_reg.predict(X)

# Поліноміальна регресія (degree=2)
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

print("X[0]:", X[0])
print("X_poly[0]:", X_poly[0])

poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_poly_pred = poly_reg.predict(X_poly)

print("\nМодель варіанту 6:")
print("y = 2 * sin(X) + рівномірний шум[-0.6, 0.6]")
print("\nКоефіцієнти поліноміальної регресії:")
print(f"intercept: {poly_reg.intercept_:.4f}")
print(f"coef: {poly_reg.coef_}")
print(f"\nR2 лінійна: {r2_score(y, y_lin_pred):.4f}")
print(f"R2 поліноміальна (degree=2): {r2_score(y, y_poly_pred):.4f}")

# Графік
plt.figure(figsize=(10, 5))
plt.scatter(X, y, color='blue', s=20, label='Дані (варіант 6)')
plt.plot(X, y_lin_pred, color='red', linewidth=2, label='Лінійна регресія')
plt.plot(X, y_poly_pred, color='green', linewidth=2, label='Поліноміальна (degree=2)')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Варіант 6: y = 2 * sin(X) + шум')
plt.legend()
plt.grid(True)
plt.show()
