import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

# Завантаження даних про акції Apple з Yahoo Finance
ticker = 'AAPL'
data = yf.download(ticker, start='2010-01-01', end='2023-01-01')
data = data['Close'].values.reshape(-1, 1)

# Нормалізація даних
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Функція для створення набору даних
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

# Визначення параметрів
time_step = 100
X, y = create_dataset(scaled_data, time_step)

# Розподіл даних на навчальну та тестову вибірки
train_size = int(len(X) * 0.7)
test_size = len(X) - train_size
X_train, X_test = X[0:train_size], X[train_size:len(X)]
y_train, y_test = y[0:train_size], y[train_size:len(y)]

# Перетворення даних у формат [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Побудова моделі LSTM
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Навчання моделі
model.fit(X_train, y_train, batch_size=1, epochs=1)

# Прогнозування
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Повернення даних до початкового масштабу
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Нечітка логіка для прогнозування
# Генерація нечітких множин для змінних


# Підготовка даних для візуалізації
train_range = range(time_step, time_step + len(train_predict))
test_range = range(len(train_predict) + (time_step * 2), len(train_predict) + (time_step * 2) + len(test_predict))

# Візуалізація даних
plt.figure(figsize=(16, 8))
plt.plot(scaler.inverse_transform(scaled_data), label='Реальні дані')
plt.plot(train_range, train_predict, label='Прогноз навчальної вибірки')
plt.plot(test_range, test_predict, label='Прогноз тестової вибірки')
plt.title('Прогнозування цін на акції Apple')
plt.xlabel('Час')
plt.ylabel('Ціна акції')
plt.legend()
plt.show()