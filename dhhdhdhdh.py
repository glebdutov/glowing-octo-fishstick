import telebot
from pytube import YouTube
import os

BOT_TOKEN = '8531711781:AAEpsVFDDYhcH5ny5nq0lNBIvH7dFQbeM4A'
bot = telebot.TeleBot(BOT_TOKEN)

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Привет! Отправь мне ссылку на видео с YouTube.")

@bot.message_handler(func=lambda message: True)
def download_video(message):
    url = message.text
    chat_id = message.chat.id

    # Простая проверка на YouTube-ссылку
    if "youtube.com" not in url and "youtu.be" not in url:
        bot.send_message(chat_id, "Пожалуйста, отправьте корректную ссылку на YouTube.")
        return

    try:
        bot.send_message(chat_id, "Начинаю загрузку...")
        yt = YouTube(url)
        # Выбор потока с наивысшим разрешением
        video_stream = yt.streams.get_highest_resolution()
        # Скачивание во временный файл
        video_file = video_stream.download(output_path='./downloads')
        # Отправка видео пользователю
        with open(video_file, 'rb') as video:
            bot.send_video(chat_id, video)
        # Удаление временного файла
        os.remove(video_file)
        bot.send_message(chat_id, "Готово!")
    except Exception as e:
        bot.send_message(chat_id, f"Произошла ошибка: {e}")

if __name__ == '__main__':
    bot.polling()