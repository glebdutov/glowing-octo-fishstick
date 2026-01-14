import os
import logging
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Загрузка ключа
load_dotenv()
API_KEY = os.getenv('MDE5YmIxMTAtZDBhNi03NmZhLTk3NzQtYzBmYjVhYTY2ODdkOjJmNDM4NTUwLThjOWMtNGQzMy1hOWJhLTk4YzA5MDZmOWQxNA==')
BOT_TOKEN = os.getenv('8531711781:AAEpsVFDDYhcH5ny5nq0lNBIvH7dFQbeM4A')  # Токен от @BotFather

# Логирование
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

# Функция для запроса к AI-API (заглушка - нужна документация к вашему сервису)
async def get_ai_response(user_message: str, api_key: str) -> str:
    """
    Отправляет запрос к AI-сервису и возвращает ответ.
    Вам нужно адаптировать эту функцию под конкретное API.
    """
    # Пример структуры для условного API
    headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
    data = {'model': 'gpt-4', 'messages': [{'role': 'user', 'content': user_message}]}
    
    try:
        # response = requests.post('URL_ВАШЕГО_СЕРВИСА', headers=headers, json=data)
        # return response.json()['choices'][0]['message']['content']
        return "Здесь будет ответ от нейросети, когда вы настроите API."
    except Exception as e:
        logging.error(f"Ошибка API: {e}")
        return "Произошла ошибка при обращении к нейросети."

# Обработчик текстовых сообщений
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    logging.info(f"Пользователь {update.effective_user.id} написал: {user_message}")
    
    # Получаем ответ от AI
    ai_response = await get_ai_response(user_message, API_KEY)
    
    # Отправляем ответ пользователю
    await update.message.reply_text(ai_response)

# Обработчик команды /start
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Привет! Я бот RedRu. Напишите мне что-нибудь.')

# Главная функция
def main():
    if not BOT_TOKEN:
        logging.error("Не найден TELEGRAM_BOT_TOKEN в переменных окружения.")
        return
    
    app = Application.builder().token(BOT_TOKEN).build()
    
    # Регистрация обработчиков
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Запуск бота
    logging.info("Бот запущен...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()