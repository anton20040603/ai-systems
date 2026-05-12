# LR_2_task_3.py
import numpy as np
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import matplotlib
matplotlib.use('TkAgg')

# Завантаження датасету
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['Length of sepal', 'Width of sepal', 'Length of petal', 'Width of petal', 'Class']
dataset = read_csv(url, names=names)

# Розмір
print("Shape:", dataset.shape)

# Перші рядки
print(dataset.head())

# Статистика
print(dataset.describe())

# Розподіл класів
print("\nDistribution of classes:")
print(dataset.groupby('Class').size())

# Візуалізація (можеш залишити або закоментити якщо довго)
dataset.plot(kind='box', subplots=True, layout=(2,2), figsize=(8,6))
pyplot.show()

dataset.hist()
pyplot.show()

scatter_matrix(dataset, figsize=(10,8))
pyplot.show()

# Підготовка даних
array = dataset.values
X = array[:,0:4]
y = array[:,4]

# Розбиття
X_train, X_validation, Y_train, Y_validation = train_test_split(
    X, y, test_size=0.20, random_state=1
)

# Моделі
models = []
models.append(('LR', LogisticRegression(solver='lbfgs', max_iter=200)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# Оцінка
results = []
names_models = []

for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names_models.append(name)
    print(f"{name}: {cv_results.mean():.4f} ({cv_results.std():.4f})")

# Графік
pyplot.boxplot(results, labels=names_models)
pyplot.title('Algorithm Comparison')
pyplot.show()

# Тест на кращій моделі
model = SVC()
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

print("\nAccuracy:", accuracy_score(Y_validation, predictions))
print("\nConfusion matrix:\n", confusion_matrix(Y_validation, predictions))
print("\nReport:\n", classification_report(Y_validation, predictions))
