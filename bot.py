import telebot
from telebot import types
import numpy as np
from model import load_model_and_scaler
from config import token

# Telegram bot token
TOKEN = token
bot = telebot.TeleBot(TOKEN)


# Обработчики сообщений бота
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message,
                 "Привет! Я бот, который предсказывает будущие цены криптовалют. Просто отправь /predict, чтобы начать предсказание.")


@bot.message_handler(commands=['predict'])
def start_prediction(message):
    markup = types.ReplyKeyboardMarkup(row_width=1, resize_keyboard=True)
    item_bitcoin = types.KeyboardButton('Bitcoin')
    item_ethereum = types.KeyboardButton('Ethereum')
    markup.add(item_bitcoin, item_ethereum)

    bot.reply_to(message, "Отлично! Теперь выбери криптовалюту:", reply_markup=markup)
    bot.register_next_step_handler(message, choose_crypto)


def choose_crypto(message):
    try:
        markup = types.ReplyKeyboardMarkup(row_width=1, resize_keyboard=True)
        item_minute = types.KeyboardButton('minute')
        item_hour = types.KeyboardButton('hour')
        item_day = types.KeyboardButton('day')
        markup.add(item_minute, item_hour, item_day)

        bot.reply_to(message, f"Отлично! Ты выбрал {message.text.lower()}.\nТеперь выбери временной интервал:",
                     reply_markup=markup)
        bot.register_next_step_handler(message, process_time_interval, message.text.lower())
    except Exception as e:
        bot.reply_to(message, f'Произошла ошибка: {str(e)}')


def process_time_interval(message, crypto_name):
    try:
        time_interval = message.text.lower()
        if time_interval == 'minute':
            example = "Пример для предсказания минут: 32000 32500 33000 33500 34000 34500 35000 35500 36000 36500"
            prices_count = 10
        elif time_interval == 'hour':
            example = "Пример для предсказания часа: 32000 32500 33000 33500 34000 34500 35000 35500 36000 36500 37000 37500 38000 38500 39000 39500 40000 40500 41000 41500 42000 42500 43000 43500"
            prices_count = 24
        elif time_interval == 'day':
            example = "Пример для предсказания дня: 32000 32500 33000 33500 34000 34500 35000"
            prices_count = 7

        bot.reply_to(message, f"Отлично! Ты выбрал временной интервал {time_interval}")
        bot.reply_to(message, f"Введи последние {prices_count} цен {crypto_name}\nНапример:\n{example}\n")
        bot.register_next_step_handler(message, process_prices, crypto_name, time_interval, prices_count)
    except Exception as e:
        bot.reply_to(message, f'Произошла ошибка: {str(e)}')


def process_prices(message, crypto_name, time_interval, prices_count):
    try:
        last_prices = message.text.split()
        if len(last_prices) != prices_count:
            bot.reply_to(message, f"Пожалуйста, введи ровно {prices_count} цен.")
            return

        last_prices = [float(price) for price in last_prices]

        bot.reply_to(message, f"Отлично! Ты ввел последние {prices_count} цен для {crypto_name}.")
        process_prediction(message, crypto_name, time_interval, last_prices)
    except ValueError:
        bot.reply_to(message, "Пожалуйста, введи числа.")


def process_prediction(message, crypto_name, time_interval, last_prices):
    try:
        # Загрузка модели и скейлера для указанной криптовалюты и временного интервала
        model, scaler = load_model_and_scaler(crypto_name, time_interval)

        # Нормализация данных
        last_days = np.array(last_prices).reshape(-1, 1)
        scaled_last_days = scaler.transform(last_days)

        # Подготовка данных для предсказания
        X_test = scaled_last_days.reshape(1, -1)

        # Предсказание
        prediction = model.predict(X_test)
        prediction = scaler.inverse_transform(prediction.reshape(-1, 1))

        bot.reply_to(message, f'Предсказанная цена для {crypto_name}: {prediction[0][0]}')
    except FileNotFoundError:
        bot.reply_to(message,
                     f'Модель или скейлер для {crypto_name} с временным интервалом {time_interval} не найдены.')
    except Exception as e:
        bot.reply_to(message, f'Произошла ошибка: {str(e)}')


if __name__ == '__main__':
    bot.polling()


