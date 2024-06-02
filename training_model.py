import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import joblib


def train_and_save_model(crypto_name, file_path, time_interval):
    data = pd.read_csv(file_path)
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values('date')
    prices = data['close'].values.reshape(-1, 1)  # »спользуем 'close' в качестве цены закрыти€

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices)

    def create_dataset(data, time_step=1):
        X, y = [], []
        for i in range(len(data) - time_step - 1):
            a = data[i:(i + time_step), 0]
            X.append(a)
            y.append(data[i + time_step, 0])
        return np.array(X), np.array(y)

    time_step = 10 if time_interval == 'minute' else 24 if time_interval == 'hour' else 7  # »змен€ем шаг в зависимости от временного интервала
    X, y = create_dataset(scaled_prices, time_step)

    model = LinearRegression()
    model.fit(X, y)

    joblib.dump(model, f'{crypto_name}_{time_interval}_model.pkl')
    joblib.dump(scaler, f'{crypto_name}_{time_interval}_scaler.pkl')

# ќбучаем и сохран€ем модель дл€ Bitcoin по разным временным интервалам
train_and_save_model('bitcoin', 'bitcoin_minute.csv', 'minute')
train_and_save_model('bitcoin', 'bitcoin_hour.csv', 'hour')
train_and_save_model('bitcoin', 'bitcoin_day.csv', 'day')
