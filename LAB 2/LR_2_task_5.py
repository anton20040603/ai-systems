# LR_2_task_5.py
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from io import BytesIO
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# Завантаження даних
iris = load_iris()
X, y = iris.data, iris.target

# Розбиття на навчальну і тестову вибірки
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.3, random_state=0)

# Створення класифікатора Ridge
clf = RidgeClassifier(tol=1e-2, solver="sag")

# Навчання
clf.fit(Xtrain, ytrain)

# Прогноз
ypred = clf.predict(Xtest)

# Метрики
print("Accuracy:", np.round(metrics.accuracy_score(ytest, ypred), 4))
print("Precision:", np.round(metrics.precision_score(ytest, ypred, average='weighted'), 4))
print("Recall:", np.round(metrics.recall_score(ytest, ypred, average='weighted'), 4))
print("F1 Score:", np.round(metrics.f1_score(ytest, ypred, average='weighted'), 4))
print("Cohen Kappa Score:", np.round(metrics.cohen_kappa_score(ytest, ypred), 4))
print("Matthews Corrcoef:", np.round(metrics.matthews_corrcoef(ytest, ypred), 4))

print("\nClassification Report:\n")
print(metrics.classification_report(ytest, ypred, target_names=iris.target_names))

# Матриця помилок
mat = confusion_matrix(ytest, ypred)

plt.figure(figsize=(6, 5))
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False, cmap='Purples',
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted label')
plt.ylabel('Actual label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig("Confusion.jpg")
plt.show()

# Save SVG in a fake file object
f = BytesIO()
plt.savefig(f, format="svg")