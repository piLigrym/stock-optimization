import pandas as pd
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import yfinance as yf
from arch import arch_model
from statsmodels.tsa.stattools import adfuller

# Новий список компаній для аналізу, включаючи менш відомі
tickers = [
    'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'NFLX', 'ADBE', 'PYPL',
    'XOM', 'CVX', 'BA', 'JPM', 'BAC', 'GS', 'WFC', 'IBM', 'INTC', 'CSCO', 'ORCL',
    'SPWR', 'FSLR', 'RUN', 'SEDG', 'PLUG', 'ENPH', 'FCEL', 'BLDP', 'NIO', 'LI'
]

# Завантаження даних для кожної компанії
data = yf.download(tickers, start='2000-01-01', end='2023-01-01')['Adj Close']

# Заповнення відсутніх значень
data.fillna(method='ffill', inplace=True)

# Розрахунок відсоткових змін
returns = data.pct_change().dropna()


# Перевірка станціонарності даних
def check_stationarity(data):
    p_values = []
    for column in data.columns:
        p_value = adfuller(data[column])[1]
        p_values.append(p_value)
    return all(p < 0.05 for p in p_values)


if not check_stationarity(returns):
    print("Дані не станціонарні. Виконується диференціювання.")
    returns = returns.diff().dropna()

# Масштабування даних
scaler = StandardScaler()
scaled_returns = scaler.fit_transform(returns)


# Використання моделі GARCH для кожної акції
def fit_garch(data):
    models = []
    for i in range(data.shape[1]):
        model = arch_model(data[:, i], vol='Garch', p=1, q=1)
        res = model.fit(disp="off")
        models.append(res)
    return models


models = fit_garch(scaled_returns)


# Симуляція спрямованих викидів волатильності
def calculate_spillovers(models):
    from_spillovers = np.zeros(len(models))
    to_spillovers = np.zeros(len(models))
    for i, model in enumerate(models):
        params = model.params
        from_spillovers[i] = params['alpha[1]'] + params['beta[1]']
        to_spillovers[i] = np.mean([m.params['alpha[1]'] + m.params['beta[1]'] for j, m in enumerate(models) if j != i])
    return from_spillovers, to_spillovers


from_spillovers, to_spillovers = calculate_spillovers(models)

# Підготовка даних для кластеризації
spillover_data = np.vstack((from_spillovers, to_spillovers))

# Застосування алгоритму нечіткої C-середніх
n_clusters = 3
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    spillover_data, n_clusters, 2, error=0.005, maxiter=1000, init=None)

# Визначення найбільш ймовірного кластера для кожного зразка
cluster_membership = np.argmax(u, axis=0)

# Візуалізація результатів кластеризації
plt.figure(figsize=(12, 8))
colors = ['b', 'g', 'r']

for i in range(n_clusters):
    cluster_indices = np.where(cluster_membership == i)[0]
    plt.scatter(from_spillovers[cluster_indices], to_spillovers[cluster_indices], color=colors[i],
                label=f'Cluster {i + 1}', s=100)

# Додавання підписів для кожної точки
for i, ticker in enumerate(tickers):
    plt.text(from_spillovers[i], to_spillovers[i], ticker, fontsize=12)

plt.title('Fuzzy C-Means Clustering of Volatility Spillovers')
plt.xlabel('From Spillovers')
plt.ylabel('To Spillovers')
plt.legend()
plt.grid(True)
plt.show()

# Оцінка якості кластеризації
print(f'Fuzzy Partition Coefficient: {fpc}')
