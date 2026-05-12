# LR_5_task_1.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from utilities import visualize_classifier
import matplotlib
matplotlib.use('TkAgg')
# 'rf' = випадковий ліс, 'erf' = гранично випадковий
classifier_type = 'erf'
# Завантаження даних
input_file = r'D:\users\Документы\СШІ\LAB 5\data_random_forests.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]
# Розбиття на три класи
class_0 = np.array(X[y==0])
class_1 = np.array(X[y==1])
class_2 = np.array(X[y==2])
# Візуалізація вхідних даних
plt.figure()
plt.scatter(class_0[:, 0], class_0[:, 1], s=75, facecolors='white',
            edgecolors='black', linewidth=1, marker='s')
plt.scatter(class_1[:, 0], class_1[:, 1], s=75, facecolors='white',
            edgecolors='black', linewidth=1, marker='o')
plt.scatter(class_2[:, 0], class_2[:, 1], s=75, facecolors='white',
            edgecolors='black', linewidth=1, marker='^')
plt.title('Вхідні дані')
# Розбиття на навчальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=5)
# Класифікатор
params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}
if classifier_type == 'rf':
    classifier = RandomForestClassifier(**params)
    print("Використовується: Random Forest")
else:
    classifier = ExtraTreesClassifier(**params)
    print("Використовується: Extra Trees")
classifier.fit(X_train, y_train)
visualize_classifier(classifier, X_train, y_train, 'Training dataset')
y_test_pred = classifier.predict(X_test)
visualize_classifier(classifier, X_test, y_test, 'Test dataset')
# Звіт
class_names = ['Class-0', 'Class-1', 'Class-2']
print("\n" + "#"*40)
print("Classifier performance on training dataset:")
print(classification_report(y_train, classifier.predict(X_train),
                             target_names=class_names))
print("#"*40)
print("Classifier performance on test dataset:")
print(classification_report(y_test, y_test_pred,
                             target_names=class_names))
# Рівні довіри
test_datapoints = np.array([[5,5],[3,6],[6,4],[7,2],[4,4],[5,2]])
print("\nConfidence measure:")
for datapoint in test_datapoints:
    probabilities = classifier.predict_proba([datapoint])[0]
    predicted_class = 'Class-' + str(np.argmax(probabilities))
    print('\nDatapoint:', datapoint)
    print('Predicted class:', predicted_class)

visualize_classifier(classifier, test_datapoints,
                     [0]*len(test_datapoints), 'Test points')
plt.show()
