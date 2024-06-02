# Измененная модель
import joblib

def load_model_and_scaler(crypto_name, time_interval):
    model_file = f'{crypto_name}_{time_interval}_model.pkl'
    scaler_file = f'{crypto_name}_{time_interval}_scaler.pkl'

    try:
        model = joblib.load(model_file)
        scaler = joblib.load(scaler_file)
        return model, scaler
    except FileNotFoundError:
        raise FileNotFoundError(f'Модель или скейлер для {crypto_name} с временным интервалом {time_interval} не найдены.')
