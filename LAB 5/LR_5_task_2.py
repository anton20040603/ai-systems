# LR_5_task_2.py
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utilities import visualize_classifier
import matplotlib
matplotlib.use('TkAgg')

# Завантаження даних
input_file = r'D:\users\Документы\СШІ\LAB 5\data_imbalance.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Поділ на два класи
class_0 = np.array(X[y==0])
class_1 = np.array(X[y==1])

# Візуалізація вхідних даних
plt.figure()
plt.scatter(class_0[:, 0], class_0[:, 1], s=75,
            facecolors='black', edgecolors='black',
            linewidth=1, marker='s')
plt.scatter(class_1[:, 0], class_1[:, 1], s=75,
            facecolors='white', edgecolors='black',
            linewidth=1, marker='o')
plt.title('Вхідні дані')
plt.show()

# Розбиття на навчальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=5)

# --- БЕЗ врахування дисбалансу ---
params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}

# Перевіряємо чи передано аргумент 'balance'
balance = True

if balance:
    params['class_weight'] = 'balanced'
    print("Mode: With classes imbalance (balanced)")
else:
    print("Mode: Without classes imbalance")

classifier = ExtraTreesClassifier(**params)
classifier.fit(X_train, y_train)

# Візуалізація навчального набору
visualize_classifier(classifier, X_train, y_train, 'Training dataset')

# Прогноз та візуалізація тестового набору
y_test_pred = classifier.predict(X_test)
visualize_classifier(classifier, X_test, y_test, 'Test dataset')

# Метрики
class_names = ['Class-0', 'Class-1']
print("\n" + "#"*40)
print("Classifier performance on training dataset:")
print(classification_report(y_train, classifier.predict(X_train),
                             target_names=class_names))
print("#"*40)
print("Classifier performance on test dataset:")
print(classification_report(y_test, y_test_pred,
                             target_names=class_names,
                             zero_division=0))
print("#"*40)

plt.show()
