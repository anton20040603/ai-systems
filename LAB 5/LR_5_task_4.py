# LR_5_task_4.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib
matplotlib.use('TkAgg')

# Завантаження даних (California Housing замість Boston)
housing_data = fetch_california_housing()
X, y = shuffle(housing_data.data, housing_data.target, random_state=7)
# Розбиття на навчальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=7)
# Модель AdaBoost
regressor = AdaBoostRegressor(
    DecisionTreeRegressor(max_depth=4),
    n_estimators=400,
    random_state=7
)
regressor.fit(X_train, y_train)
# Оцінка ефективності
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)
print("\nADABOOST REGRESSOR")
print("Mean squared error =", round(mse, 2))
print("Explained variance score =", round(evs, 2))
# Важливість ознак
feature_importances = regressor.feature_importances_
feature_names = housing_data.feature_names
# Нормалізація
feature_importances = 100.0 * (feature_importances / max(feature_importances))
# Сортування
index_sorted = np.flipud(np.argsort(feature_importances))
# Мітки по осі X
pos = np.arange(index_sorted.shape[0]) + 0.5
# Стовпчаста діаграма
plt.figure(figsize=(12, 6))
plt.bar(pos, feature_importances[index_sorted], align='center')
plt.xticks(pos, np.array(feature_names)[index_sorted], rotation=15)
plt.ylabel('Relative importance (%)')
plt.title('The importance of features using AdaBoost regressor')
plt.tight_layout()
plt.show()
print("\nThe importance of features (sorted):")
for i in index_sorted:
    print(f"  {feature_names[i]}: {feature_importances[i]:.2f}%")
