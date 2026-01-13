import os
import json
import requests
from openai import OpenAI
import warnings
warnings.filterwarnings('ignore')

# ========================================
# КОНФИГУРАЦИЯ: ВАШИ КЛЮЧИ И НАСТРОЙКИ
# ========================================

CONFIG = {
    # Вставьте ваши ключи ниже. НИКОМУ НЕ ПОКАЗЫВАЙТЕ этот файл!
    "API_KEYS": {
        "OPENAI_API_KEY": "sk-LdreSAkVN3GQES9geZ7tbaILLQaWZ9xFERCkuvW5v3BnxW84",       # Ключ от ChatGPT
        "STABILITY_API_KEY": "sk-LdreSAkVN3GQES9geZ7tbaILLQaWZ9xFERCkuvW5v3BnxW84",      # Ключ от Stability AI
        "DEEPSEEK_API_KEY": "sk-2ec45d0128cc41b88e49ba721034fb67"              # Необязательно
    },
    
    "DEEPSEEK_SETTINGS": {
        "use_local_fallback": True,     # Автоматически использовать локальную модель если API не работает
        "local_model_path": "./models/deepseek-model",  # Путь к локальной модели
        "api_base_url": "https://api.deepseek.com/v1"
    },
    
    "STABILITY_SETTINGS": {
        "engine_id": "stable-diffusion-xl-1024-v1-0",
        "api_host": "https://api.stability.ai"
    }
}

# ========================================
# 1. СОХРАНЕНИЕ КЛЮЧЕЙ В БЕЗОПАСНЫЙ ФАЙЛ
# ========================================

def save_config_to_file():
    """Сохраняет конфигурацию в безопасный файл config.json"""
    try:
        with open('config.json', 'w', encoding='utf-8') as f:
            json.dump(CONFIG, f, indent=4, ensure_ascii=False)
        print("✅ Конфигурация сохранена в файл 'config.json'")
        print("⚠️  НЕ ДЕЛИТЕСЬ этим файлом! Добавьте 'config.json' в .gitignore!")
    except Exception as e:
        print(f"❌ Ошибка сохранения: {e}")

# ========================================
# 2. РЕШЕНИЕ ПРОБЛЕМЫ DEEPSEEK
# ========================================

class DeepSeekUniversal:
    """Умный клиент DeepSeek с автоматическим переключением между API и локальной моделью"""
    
    def __init__(self, config):
        self.config = config
        self.api_key = config["API_KEYS"].get("DEEPSEEK_API_KEY")
        self.use_local = config["DEEPSEEK_SETTINGS"]["use_local_fallback"]
        self.local_path = config["DEEPSEEK_SETTINGS"]["local_model_path"]
        self.api_url = f"{config['DEEPSEEK_SETTINGS']['api_base_url']}/chat/completions"
        
    def query(self, prompt, use_api_first=True):
        """
        Умный запрос: пытается использовать API, при ошибке 402/429/503
        автоматически переключается на локальную модель
        """
        
        # 1. Пробуем официальное API (если есть ключ и разрешено)
        if use_api_first and self.api_key:
            try:
                response = requests.post(
                    self.api_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "deepseek-chat",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 1000
                    },
                    timeout=30
                )
                
                # Если успешно
                if response.status_code == 200:
                    result = response.json()
                    return f"🤖 [DeepSeek API]: {result['choices'][0]['message']['content']}"
                
                # Если ошибка лимита (402) или перегрузка (429, 503)
                elif response.status_code in [402, 429, 503]:
                    error_msg = {
                        402: "⛔ ЛИМИТ ЗАПРОСОВ! Использовано 5/5 бесплатных запросов.",
                        429: "🚦 СЕРВЕР ПЕРЕГРУЖЕН! Слишком много запросов.",
                        503: "🔧 СЕРВИС НЕДОСТУПЕН! Технические работы или атаки."
                    }.get(response.status_code, f"Ошибка {response.status_code}")
                    
                    print(f"⚠️  {error_msg}")
                    
                    # Автоматическое переключение на локальную модель
                    if self.use_local:
                        print("🔄 Автоматически переключаюсь на локальную модель...")
                        return self._use_local_model(prompt)
                    else:
                        return f"❌ {error_msg}\n   Решение: подождите 24 часа или оформите подписку PRO."
                
                # Другие ошибки API
                else:
                    return f"❌ Ошибка DeepSeek API: {response.status_code} - {response.text}"
                    
            except requests.exceptions.Timeout:
                return "⏰ Таймаут запроса к DeepSeek API. Сервер не отвечает."
            except Exception as e:
                print(f"⚠️  Ошибка API: {e}")
        
        # 2. Используем локальную модель (резервный вариант)
        if self.use_local:
            return self._use_local_model(prompt)
        
        return "❌ DeepSeek недоступен. Все методы не сработали."
    
    def _use_local_model(self, prompt):
        """Использование локальной версии DeepSeek (требует предварительной установки)"""
        try:
            # Попытка импорта для локальных моделей
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch
                
                # Проверяем, есть ли модель локально
                if not os.path.exists(self.local_path):
                    return ("❌ Локальная модель не найдена!\n"
                           "   Решение: скачайте модель с HuggingFace:\n"
                           "   1. pip install transformers torch\n"
                           "   2. Загрузите: from transformers import AutoModelForCausalLM\n"
                           "   3. model = AutoModelForCausalLM.from_pretrained('deepseek-ai/deepseek-llm-7b-chat')")
                
                # Загрузка модели (упрощённый пример)
                tokenizer = AutoTokenizer.from_pretrained(self.local_path)
                model = AutoModelForCausalLM.from_pretrained(self.local_path)
                
                # Генерация ответа
                inputs = tokenizer(prompt, return_tensors="pt")
                outputs = model.generate(**inputs, max_length=500)
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                return f"🏠 [Локальный DeepSeek]: {response}"
                
            except ImportError:
                return ("📦 Установите пакеты для локальной модели:\n"
                       "   pip install transformers torch accelerate\n"
                       "   или используйте LM Studio/GPT4All")
                
        except Exception as e:
            return f"❌ Ошибка локальной модели: {e}"

# ========================================
# 3. ИНТЕГРАЦИЯ ВСЕХ НЕЙРОСЕТЕЙ
# ========================================

class NeuroAssistant:
    """Универсальный помощник со всеми нейросетями"""
    
    def __init__(self, config):
        self.config = config
        self.deepseek = DeepSeekUniversal(config)
        
        # Инициализация OpenAI (ChatGPT)
        self.openai_client = OpenAI(api_key=config["API_KEYS"]["OPENAI_API_KEY"])
        
    def ask_chatgpt(self, prompt):
        """Запрос к ChatGPT"""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000
            )
            return f"🧠 [ChatGPT]: {response.choices[0].message.content}"
        except Exception as e:
            return f"❌ Ошибка ChatGPT: {e}"
    
    def generate_image(self, prompt):
        """Генерация изображения через Stability AI"""
        try:
            response = requests.post(
                f"{self.config['STABILITY_SETTINGS']['api_host']}/v1/generation/{self.config['STABILITY_SETTINGS']['engine_id']}/text-to-image",
                headers={
                    "Authorization": f"Bearer {self.config['API_KEYS']['STABILITY_API_KEY']}",
                    "Content-Type": "application/json"
                },
                json={
                    "text_prompts": [{"text": prompt, "weight": 1.0}],
                    "cfg_scale": 7,
                    "height": 1024,
                    "width": 1024,
                    "samples": 1,
                    "steps": 30
                }
            )
            
            if response.status_code == 200:
                import base64
                from datetime import datetime
                
                data = response.json()
                image_data = base64.b64decode(data["artifacts"][0]["base64"])
                
                # Сохраняем изображение
                filename = f"image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                with open(filename, "wb") as f:
                    f.write(image_data)
                
                return f"🎨 [Stable Diffusion]: Изображение сохранено как '{filename}'"
            else:
                return f"❌ Ошибка Stability AI: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"❌ Ошибка генерации изображения: {e}"
    
    def ask_deepseek(self, prompt):
        """Умный запрос к DeepSeek с авто-переключением"""
        return self.deepseek.query(prompt)
    
    def universal_ask(self, prompt, preferred="deepseek"):
        """
        Универсальный запрос с выбором нейросети
        Доступные варианты: 'deepseek', 'chatgpt', 'image'
        """
        if preferred == "chatgpt":
            return self.ask_chatgpt(prompt)
        elif preferred == "image":
            return self.generate_image(prompt)
        else:  # deepseek
            return self.ask_deepseek(prompt)

# ========================================
# 4. ИНТЕРАКТИВНЫЙ ИНТЕРФЕЙС
# ========================================

def main():
    """Главная функция с интерактивным меню"""
    
    print("=" * 60)
    print("🤖 УНИВЕРСАЛЬНЫЙ НЕЙРО-АССИСТЕНТ v2.0")
    print("=" * 60)
    
    # Сначала сохраняем конфигурацию
    save_config_to_file()
    
    # Инициализируем помощника
    assistant = NeuroAssistant(CONFIG)
    
    # Основной цикл
    while True:
        print("\n" + "=" * 60)
        print("Выберите действие:")
        print("  1. 🚀 Задать вопрос DeepSeek (с авто-исправлением проблем)")
        print("  2. 🧠 Задать вопрос ChatGPT")
        print("  3. 🎨 Сгенерировать изображение (Stable Diffusion)")
        print("  4. 🔄 Универсальный запрос (автовыбор)")
        print("  5. ⚙️  Показать/изменить конфигурацию")
        print("  6. ❌ Выйти")
        print("=" * 60)
        
        choice = input("\nВаш выбор (1-6): ").strip()
        
        if choice == "1":
            prompt = input("Ваш вопрос для DeepSeek: ").strip()
            if prompt:
                print("\n" + assistant.ask_deepseek(prompt))
        
        elif choice == "2":
            prompt = input("Ваш вопрос для ChatGPT: ").strip()
            if prompt:
                print("\n" + assistant.ask_chatgpt(prompt))
        
        elif choice == "3":
            prompt = input("Опишите изображение: ").strip()
            if prompt:
                print("\n" + assistant.generate_image(prompt))
        
        elif choice == "4":
            prompt = input("Ваш запрос: ").strip()
            if prompt:
                print("\n" + assistant.universal_ask(prompt))
        
        elif choice == "5":
            print("\nТекущая конфигурация:")
            print(json.dumps(CONFIG, indent=2, ensure_ascii=False))
            
            change = input("\nИзменить ключи? (да/нет): ").lower()
            if change == "да":
                new_openai = input("Новый ключ OpenAI: ").strip()
                new_stability = input("Новый ключ Stability AI: ").strip()
                new_deepseek = input("Новый ключ DeepSeek (опционально): ").strip()
                
                if new_openai:
                    CONFIG["API_KEYS"]["OPENAI_API_KEY"] = new_openai
                if new_stability:
                    CONFIG["API_KEYS"]["STABILITY_API_KEY"] = new_stability
                if new_deepseek:
                    CONFIG["API_KEYS"]["DEEPSEEK_API_KEY"] = new_deepseek
                
                save_config_to_file()
                print("✅ Ключи обновлены!")
        
        elif choice == "6":
            print("\nДо свидания! 👋")
            break
        
        else:
            print("❌ Неверный выбор. Попробуйте снова.")

# ========================================
# ЗАПУСК ПРОГРАММЫ
# ========================================

if __name__ == "__main__":
    # Создаем папку для моделей если её нет
    if not os.path.exists("./models"):
        os.makedirs("./models")
    
    # Запускаем программу
    main()