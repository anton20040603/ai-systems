# LR_8_task_1
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# --- Параметри ---
n_samples  = 1000   # кількість точок
batch_size = 100    # розмір міні-батча
num_steps  = 20000  # кількість кроків навчання

# --- Генерація вхідних даних ---
# 1000 випадкових точок на інтервалі [1, 10]
X_data = np.random.uniform(1, 10, (n_samples, 1))
# Правильна відповідь: y = 2x + 1 + шум (дисперсія 2)
y_data = 2 * X_data + 1 + np.random.normal(0, 2, (n_samples, 1))

# --- Оголошення заглушок tf.placeholder ---
# X і y — вхідні дані розмірності (batch_size x 1)
X = tf.placeholder(tf.float32, shape=(batch_size, 1))
y = tf.placeholder(tf.float32, shape=(batch_size, 1))

# --- Ініціалізація змінних k та b ---
with tf.variable_scope('linear-regression'):
    k = tf.Variable(tf.random_normal((1, 1)), name='slope')  # нахил
    b = tf.Variable(tf.zeros((1,)),            name='bias')   # зміщення

# --- Модель та функція помилки ---
y_pred = tf.matmul(X, k) + b                   # прогноз: y = kx + b
loss   = tf.reduce_sum((y - y_pred) ** 2)       # MSE: сума квадратів відхилень

# --- Оптимізатор: стохастичний градієнтний спуск ---
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(loss)

# --- Навчання ---
display_step = 100
loss_history = []

with tf.Session() as sess:
    # Ініціалізація всіх змінних
    sess.run(tf.global_variables_initializer())
    for i in range(num_steps):
        # Випадкове підмножина індексів розміром batch_size
        indices = np.random.choice(n_samples, batch_size)
        X_batch, y_batch = X_data[indices], y_data[indices]

        # Обчислення optimizer, loss, k, b за один крок
        _, loss_val, k_val, b_val = sess.run(
            [optimizer, loss, k, b],
            feed_dict={X: X_batch, y: y_batch}
        )

        loss_history.append(loss_val)

        # Виведення кожні display_step епох
        if (i + 1) % display_step == 0:
            print('Епоха %d: %.8f, k=%.4f, b=%.4f' %
                  (i + 1, loss_val, k_val[0][0], b_val[0]))

    # Фінальні значення
    print('\n=== Final parameters ===')
    print(f'k (incline)    = {k_val[0][0]:.4f}  (expected ~2.0)')
    print(f'b (displacement) = {b_val[0]:.4f}  (expected ~1.0)')

# --- Графік 1: крива навчання (loss) ---
plt.figure(figsize=(10, 4))
plt.plot(loss_history)
plt.title('Training curve (loss function)')
plt.xlabel('Training step')
plt.ylabel('Loss (MSE)')
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Графік 2: дані та знайдена пряма ---
plt.figure(figsize=(8, 5))
plt.scatter(X_data, y_data, s=5, alpha=0.4, label='Вхідні дані')
x_line = np.linspace(1, 10, 200)
y_line = k_val[0][0] * x_line + b_val[0]
plt.plot(x_line, y_line, 'r-', linewidth=2,
         label=f'Found line: y = {k_val[0][0]:.2f}x + {b_val[0]:.2f}')
plt.plot(x_line, 2 * x_line + 1, 'g--', linewidth=1.5,
         label='Ideal line: y = 2x + 1')
plt.title('Linear regression (TensorFlow)')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()