# LR_7_tasks.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib
matplotlib.use('TkAgg')

# ============================================================
# ЗАВДАННЯ 2. Метод найменших квадратів — варіант 6
# X: 3.33  1  63  0.87  0.42  0.27
# Y: 0.48  1.03  2.02  4.25  7.16  11.5
# ============================================================

X_data = np.array([3.33, 1, 63, 0.87, 0.42, 0.27], dtype=float)
Y_data = np.array([0.48, 1.03, 2.02, 4.25, 7.16, 11.5], dtype=float)

# Метод найменших квадратів вручну
n = len(X_data)
sum_x  = np.sum(X_data)
sum_y  = np.sum(Y_data)
sum_x2 = np.sum(X_data**2)
sum_xy = np.sum(X_data * Y_data)

# Нормальні рівняння: β0*n + β1*sum_x = sum_y
#                    β0*sum_x + β1*sum_x2 = sum_xy
A = np.array([[n, sum_x],
              [sum_x, sum_x2]])
b = np.array([sum_y, sum_xy])
beta = np.linalg.solve(A, b)
beta0, beta1 = beta[0], beta[1]

print("=" * 50)
print("ЗАВДАННЯ 2. Метод найменших квадратів")
print("=" * 50)
print(f"β0 (вільний член) = {beta0:.4f}")
print(f"β1 (нахил)        = {beta1:.4f}")
print(f"Рівняння: y = {beta0:.4f} + {beta1:.4f}*x")

# Прогноз
X_line = np.linspace(0, 14, 100)
Y_line = beta0 + beta1 * X_line
Y_pred = beta0 + beta1 * X_data

# Якість
r2  = r2_score(Y_data, Y_pred)
mae = mean_absolute_error(Y_data, Y_pred)
mse = mean_squared_error(Y_data, Y_pred)
print(f"\nR2  = {r2:.4f}")
print(f"MAE = {mae:.4f}")
print(f"MSE = {mse:.4f}")

# Графік
plt.figure(figsize=(8, 5))
plt.scatter(X_data, Y_data, color='blue', s=80, zorder=5, label='Експериментальні точки')
plt.plot(X_line, Y_line, color='red', linewidth=2, label=f'y = {beta0:.2f} + {beta1:.2f}·x')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Завдання 2. МНК — варіант 6')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('task2_mnk.png', dpi=150)
plt.show()

# ============================================================
# ЗАВДАННЯ 3. Інтерполяція
# x = [0.1, 0.3, 0.4, 0.6, 0.7]
# y = [3.2, 3.0, 1.0, 1.8, 1.9]
# ============================================================

x_nodes = np.array([0.1, 0.3, 0.4, 0.6, 0.7])
y_nodes = np.array([3.2, 3.0, 1.0, 1.8, 1.9])

# Заповнення матриці Вандермонда (поліном степеню 4)
n_nodes = len(x_nodes)
X_vander = np.vander(x_nodes, n_nodes, increasing=True)

print("\n" + "=" * 50)
print("ЗАВДАННЯ 3. Інтерполяція")
print("=" * 50)
print("Матриця X (Вандермонда):")
print(np.round(X_vander, 4))

# Коефіцієнти полінома
coeffs = np.linalg.solve(X_vander, y_nodes)
print(f"\nКоефіцієнти інтерполяційного полінома:")
for i, c in enumerate(coeffs):
    print(f"  a{i} = {c:.4f}")

# Функція полінома
def poly(x):
    result = 0
    for i, c in enumerate(coeffs):
        result += c * x**i
    return result

# Значення в проміжних точках
x_02 = 0.2
x_05 = 0.5
print(f"\nЗначення в проміжних точках:")
print(f"  f(0.2) = {poly(x_02):.4f}")
print(f"  f(0.5) = {poly(x_05):.4f}")

# Графік
x_plot = np.linspace(0.05, 0.75, 200)
y_plot = np.array([poly(xi) for xi in x_plot])

plt.figure(figsize=(8, 5))
plt.scatter(x_nodes, y_nodes, color='blue', s=100, zorder=5, label='Вузли інтерполяції')
plt.plot(x_plot, y_plot, color='green', linewidth=2, label='Інтерполяційний поліном (степінь 4)')
plt.scatter([x_02, x_05],
            [poly(x_02), poly(x_05)],
            color='red', s=100, zorder=5,
            label=f'f(0.2)={poly(x_02):.3f}, f(0.5)={poly(x_05):.3f}')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Завдання 3. Інтерполяція поліномом 4-го степеня')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('task3_interpolation.png', dpi=150)
plt.show()
