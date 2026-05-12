# LR_7_task_4.py
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn import covariance, cluster
import matplotlib
matplotlib.use('TkAgg')
import yfinance as yf
from datetime import datetime

# Вхідний файл із символічними позначеннями компаній
input_file = r'D:\users\Документы\СШІ\LAB 7\company_symbol_mapping.json'

# Завантаження прив'язок символів компаній до їх повних назв
with open(input_file, 'r') as f:
    company_symbols_map = json.loads(f.read())

symbols, names = np.array(list(company_symbols_map.items())).T

# Завантаження архівних даних котирувань через yfinance
start_date = '2003-07-03'
end_date   = '2007-05-04'

print('Завантаження даних котирувань...')
quotes_data = yf.download(list(symbols), start=start_date, end=end_date,
                           auto_adjust=True, progress=False)

# Вилучення котирувань відкриття та закриття біржі
opening_quotes = quotes_data['Open'].values.T
closing_quotes = quotes_data['Close'].values.T

# Обчислення різниці між двома видами котирувань
quotes_diff = closing_quotes - opening_quotes

# Нормалізація даних
X = quotes_diff.copy().T
X /= X.std(axis=0)

# Створення моделі графа (GraphLassoCV будує разріджену коваріаційну матрицю)
edge_model = covariance.GraphicalLassoCV()

# Навчання моделі
with np.errstate(invalid='ignore'):
    edge_model.fit(X)

# Створення моделі кластеризації на основі поширення подібності
_, labels = cluster.affinity_propagation(edge_model.covariance_,
                                          random_state=0)
num_labels = labels.max()

# Виведення результатів – компанії згруповані по кластерах
print('\nResults of clusterization:')
for i in range(num_labels + 1):
    cluster_members = ', '.join(names[labels == i])
    print(f'Cluster {i+1} ==> {cluster_members}')
