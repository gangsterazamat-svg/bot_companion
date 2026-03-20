import os
import json
import logging
from datetime import datetime
from typing import Dict, Optional
import asyncio

from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
import httpx

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Конфигурация
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"  # Используем бесплатную модель
BOT_USERNAME = None  # Будет определено при запуске

# Директории для хранения данных
DATA_DIR = "user_data"
os.makedirs(DATA_DIR, exist_ok=True)

# Файл для хранения системных промтов
SYSTEM_PROMPTS_FILE = os.path.join(DATA_DIR, "system_prompts.json")
USER_INFO_FILE = os.path.join(DATA_DIR, "user_info.json")

# Загрузка данных
def load_system_prompts() -> Dict:
    try:
        if os.path.exists(SYSTEM_PROMPTS_FILE):
            with open(SYSTEM_PROMPTS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Ошибка загрузки системных промтов: {e}")
    return {}

def load_user_info() -> Dict:
    try:
        if os.path.exists(USER_INFO_FILE):
            with open(USER_INFO_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Ошибка загрузки информации о пользователях: {e}")
    return {}

# Сохранение данных
def save_system_prompts(prompts: Dict):
    try:
        with open(SYSTEM_PROMPTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(prompts, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Ошибка сохранения системных промтов: {e}")

def save_user_info(user_info: Dict):
    try:
        with open(USER_INFO_FILE, 'w', encoding='utf-8') as f:
            json.dump(user_info, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Ошибка сохранения информации о пользователях: {e}")

# Глобальные переменные для данных
SYSTEM_PROMPTS = load_system_prompts()
USER_INFO = load_user_info()

# Генерация текста через OpenRouter API
async def generate_text(prompt: str, system_message: str = "", user_id: str = "") -> str:
    # Проверка наличия API ключа
    if not OPENROUTER_API_KEY:
        return "❌ Не настроен API ключ OpenRouter. Обратитесь к администратору бота."
    
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": os.getenv('BOT_DOMAIN', 'https://your-bot-domain.com'),
            "X-Title": "Personal Assistant Bot",
            "Content-Type": "application/json"
        }
        
        # Формируем сообщения
        messages = []
        
        # Добавляем системное сообщение если есть
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        # Добавляем информацию о пользователе в системное сообщение
        if user_id in USER_INFO:
            user_data = USER_INFO[user_id]
            user_info_text = []
            if user_data.get('name'):
                user_info_text.append(f"Имя пользователя: {user_data['name']}")
            if user_data.get('age'):
                user_info_text.append(f"Возраст: {user_data['age']}")
            if user_data.get('interests'):
                user_info_text.append(f"Интересы: {user_data['interests']}")
            if user_data.get('personality'):
                user_info_text.append(f"Характер: {user_data['personality']}")
            
            if user_info_text:
                user_info_prompt = "Дополнительная информация о пользователе:\n" + "\n".join(user_info_text)
                if system_message:
                    messages[0]["content"] += f"\n\n{user_info_prompt}"
                else:
                    messages.append({"role": "system", "content": user_info_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        data = {
            "model": DEFAULT_MODEL,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        # Настройка HTTP клиента с таймаутами
        timeout = httpx.Timeout(45.0, connect=15.0)  # Увеличенные таймауты для большой модели
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            # Попытка с повторами
            for attempt in range(3):
                try:
                    response = await client.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers=headers,
                        json=data
                    )
                    
                    logger.info(f"OpenRouter API Response Status: {response.status_code}")
                    
                    if response.status_code == 200:
                        result = response.json()
                        logger.debug(f"OpenRouter API Response: {result}")
                        
                        # Проверяем структуру ответа
                        if 'choices' in result and len(result['choices']) > 0:
                            content = result['choices'][0]['message'].get('content')
                            if content is not None:
                                return content.strip()
                            else:
                                return "Пустой ответ от модели."
                        else:
                            return "Неожиданный формат ответа от API."
                    elif response.status_code == 429:
                        # Слишком много запросов
                        wait_time = 2 ** attempt
                        logger.warning(f"Rate limit превышен, ждем {wait_time} секунд")
                        await asyncio.sleep(wait_time)
                        continue
                    elif response.status_code == 401:
                        return "❌ Ошибка авторизации API ключа. Проверьте правильность ключа."
                    elif response.status_code == 400:
                        try:
                            error_data = response.json()
                            error_msg = error_data.get('error', {}).get('message', 'Bad Request')
                            return f"❌ Ошибка запроса: {error_msg}"
                        except:
                            return f"❌ Ошибка запроса ({response.status_code})"
                    else:
                        error_text = response.text
                        logger.error(f"API Error {response.status_code}: {error_text}")
                        return f"❌ Ошибка API ({response.status_code}). Попробуйте позже."
                        
                except httpx.TimeoutException:
                    logger.warning(f"Таймаут при подключении к OpenRouter API, попытка {attempt + 1}")
                    if attempt < 2:
                        await asyncio.sleep(5)
                        continue
                    else:
                        return "❌ Превышено время ожидания ответа от API. Попробуйте позже."
                        
                except httpx.ConnectError as e:
                    logger.error(f"Ошибка подключения к OpenRouter API (попытка {attempt + 1}): {e}")
                    if attempt < 2:
                        await asyncio.sleep(5)
                        continue
                    else:
                        return "❌ Ошибка подключения к API. Проверьте интернет-соединение."
                        
                except httpx.NetworkError as e:
                    logger.error(f"Сетевая ошибка (попытка {attempt + 1}): {e}")
                    if attempt < 2:
                        await asyncio.sleep(5)
                        continue
                    else:
                        return "❌ Сетевая ошибка. Проверьте подключение к интернету."
                        
                except Exception as e:
                    logger.error(f"Неожиданная ошибка при запросе к API (попытка {attempt + 1}): {e}")
                    if attempt < 2:
                        await asyncio.sleep(5)
                        continue
                    else:
                        return "❌ Произошла ошибка при обращении к API. Попробуйте позже."
                        
            return "❌ Не удалось получить ответ от API после нескольких попыток."
            
    except Exception as e:
        logger.error(f"Критическая ошибка в generate_text: {e}")
        return "❌ Произошла критическая ошибка. Попробуйте позже."

# Получение системного промта пользователя
def get_user_system_prompt(user_id: str) -> str:
    return SYSTEM_PROMPTS.get(user_id, "Ты helpful ассистент.")

# Сохранение системного промта пользователя
def set_user_system_prompt(user_id: str, prompt: str):
    SYSTEM_PROMPTS[user_id] = prompt
    save_system_prompts(SYSTEM_PROMPTS)

# Получение информации о пользователе
def get_user_info(user_id: str) -> Dict:
    return USER_INFO.get(user_id, {})

# Сохранение информации о пользователе
def set_user_info(user_id: str, info: Dict):
    USER_INFO[user_id] = info
    save_user_info(USER_INFO)

# Обработчик команды /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if not update.effective_user or not update.message:
            return
            
        user_id = str(update.effective_user.id)
        username = update.effective_user.username or update.effective_user.first_name
        
        welcome_message = (
            f"👋 Привет, {username}!\n\n"
            "Я ваш персональный ассистент с возможностью настройки характера и поведения.\n"
            f"Использую мощную модель: {DEFAULT_MODEL}\n\n"
            "Доступные команды:\n"
            "/setup - Настроить системный промт и информацию о себе\n"
            "/profile - Посмотреть/изменить ваш профиль\n"
            "/help - Показать справку\n\n"
            "Вы можете общаться со мной в личных сообщениях или упоминать меня в группах "
            f"(например: @{BOT_USERNAME} привет!)"
        )
        
        await update.message.reply_text(welcome_message)
    except Exception as e:
        logger.error(f"Ошибка в start: {e}")

# Обработчик команды /help
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if not update.message:
            return
            
        help_text = (
            "🤖 Справка по боту\n\n"
            "В личных сообщениях:\n"
            "Просто напишите мне сообщение и я отвечу вам\n\n"
            "В группах:\n"
            f"Упомяните меня @{BOT_USERNAME} и ваше сообщение, чтобы я ответил\n\n"
            "Команды:\n"
            "/start - Начать работу с ботом\n"
            "/setup - Настроить системный промт и информацию о себе\n"
            "/profile - Посмотреть/изменить ваш профиль\n"
            "/help - Показать эту справку\n\n"
            f"Используемая модель: {DEFAULT_MODEL}"
        )
        
        await update.message.reply_text(help_text)
    except Exception as e:
        logger.error(f"Ошибка в help_command: {e}")

# Обработчик команды /setup
async def setup_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if not update.effective_user or not update.message:
            return
            
        user_id = str(update.effective_user.id)
        USER_STATES[user_id] = "SETUP_SYSTEM_PROMPT"
        
        current_prompt = get_user_system_prompt(user_id)
        
        setup_message = (
            "🔧 Настройка вашего персонального ассистента\n\n"
            "Сначала давайте определим характер и поведение бота.\n"
            "Опишите, каким вы хотите видеть бота (например: веселый помощник, серьезный консультант, друг и т.д.)\n\n"
            f"Текущий промт: {current_prompt if current_prompt else 'не задан'}\n\n"
            "Введите новый системный промт или отправьте 'пропустить' чтобы оставить текущий:"
        )
        
        await update.message.reply_text(setup_message)
    except Exception as e:
        logger.error(f"Ошибка в setup_command: {e}")

# Обработчик команды /profile
async def profile_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if not update.effective_user or not update.message:
            return
            
        user_id = str(update.effective_user.id)
        user_info = get_user_info(user_id)
        
        profile_message = "👤 Ваш профиль:\n\n"
        
        if user_info:
            if user_info.get('name'):
                profile_message += f"Имя: {user_info['name']}\n"
            if user_info.get('age'):
                profile_message += f"Возраст: {user_info['age']}\n"
            if user_info.get('interests'):
                profile_message += f"Интересы: {user_info['interests']}\n"
            if user_info.get('personality'):
                profile_message += f"Характер: {user_info['personality']}\n"
        else:
            profile_message += "Информация не заполнена\n"
        
        profile_message += "\nХотите обновить информацию? Ответьте 'да' или 'нет'"
        
        USER_STATES[user_id] = "PROFILE_UPDATE_CONFIRM"
        await update.message.reply_text(profile_message)
    except Exception as e:
        logger.error(f"Ошибка в profile_command: {e}")

# Состояния пользователей
USER_STATES = {}

# Обработчик текстовых сообщений
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        # Проверка наличия необходимых данных
        if not update.effective_user or not update.message or not update.message.text:
            return
            
        user_id = str(update.effective_user.id)
        message_text = update.message.text.strip()
        chat_type = update.effective_chat.type if update.effective_chat else 'private'
        
        # Если это группа и бот не упомянут, игнорируем
        if chat_type in ['group', 'supergroup']:
            if not update.message.entities:
                return
                
            bot_mentioned = False
            for entity in update.message.entities:
                if entity.type == 'mention':
                    mention = message_text[entity.offset:entity.offset + entity.length]
                    if BOT_USERNAME and mention.lower() == f'@{BOT_USERNAME}'.lower():
                        bot_mentioned = True
                        break
            
            if not bot_mentioned:
                return
            
            # Убираем упоминание бота из сообщения
            clean_message = message_text
            if BOT_USERNAME:
                for entity in update.message.entities:
                    if entity.type == 'mention':
                        mention = message_text[entity.offset:entity.offset + entity.length]
                        if mention.lower() == f'@{BOT_USERNAME}'.lower():
                            clean_message = clean_message.replace(mention, '').strip()
                            break
            message_text = clean_message or message_text
        
        # Если сообщение пустое после очистки
        if not message_text:
            await update.message.reply_text("Пожалуйста, введите текст сообщения.")
            return
        
        # Обработка состояний настройки
        if user_id in USER_STATES:
            state = USER_STATES[user_id]
            
            if state == "SETUP_SYSTEM_PROMPT":
                if message_text.lower() != 'пропустить':
                    set_user_system_prompt(user_id, message_text)
                    await update.message.reply_text("✅ Системный промт сохранен!")
                else:
                    await update.message.reply_text("Системный промт оставлен без изменений.")
                
                USER_STATES[user_id] = "SETUP_NAME"
                await update.message.reply_text("Как вас зовут? (или 'пропустить')")
                return
            
            elif state == "SETUP_NAME":
                if message_text.lower() != 'пропустить':
                    user_info = get_user_info(user_id)
                    user_info['name'] = message_text
                    set_user_info(user_id, user_info)
                    await update.message.reply_text(f"✅ Имя сохранено: {message_text}")
                else:
                    await update.message.reply_text("Имя пропущено.")
                
                USER_STATES[user_id] = "SETUP_AGE"
                await update.message.reply_text("Сколько вам лет? (или 'пропустить')")
                return
            
            elif state == "SETUP_AGE":
                if message_text.lower() != 'пропустить':
                    try:
                        age = int(message_text)
                        user_info = get_user_info(user_id)
                        user_info['age'] = age
                        set_user_info(user_id, user_info)
                        await update.message.reply_text(f"✅ Возраст сохранен: {age}")
                    except ValueError:
                        await update.message.reply_text("Возраст должен быть числом. Пропущено.")
                else:
                    await update.message.reply_text("Возраст пропущен.")
                
                USER_STATES[user_id] = "SETUP_INTERESTS"
                await update.message.reply_text("Какими у вас интересы? (или 'пропустить')")
                return
            
            elif state == "SETUP_INTERESTS":
                if message_text.lower() != 'пропустить':
                    user_info = get_user_info(user_id)
                    user_info['interests'] = message_text
                    set_user_info(user_id, user_info)
                    await update.message.reply_text(f"✅ Интересы сохранены: {message_text}")
                else:
                    await update.message.reply_text("Интересы пропущены.")
                
                USER_STATES[user_id] = "SETUP_PERSONALITY"
                await update.message.reply_text("Опишите ваш характер/личность: (или 'пропустить')")
                return
            
            elif state == "SETUP_PERSONALITY":
                if message_text.lower() != 'пропустить':
                    user_info = get_user_info(user_id)
                    user_info['personality'] = message_text
                    set_user_info(user_id, user_info)
                    await update.message.reply_text(f"✅ Характер сохранен: {message_text}")
                else:
                    await update.message.reply_text("Характер пропущен.")
                
                USER_STATES.pop(user_id, None)
                await update.message.reply_text("✅ Настройка завершена! Теперь я знаю о вас больше.")
                return
            
            elif state == "PROFILE_UPDATE_CONFIRM":
                if message_text.lower() in ['да', 'yes', 'y']:
                    USER_STATES[user_id] = "SETUP_NAME"
                    await update.message.reply_text("Как вас зовут? (или 'пропустить')")
                else:
                    USER_STATES.pop(user_id, None)
                    await update.message.reply_text("Хорошо, оставим как есть.")
                return
        
        # Генерация ответа
        system_prompt = get_user_system_prompt(user_id)
        
        # Добавляем контекст в промпт
        full_prompt = message_text
        
        await update.message.reply_text("🤔 Думаю...")
        response = await generate_text(full_prompt, system_prompt, user_id)
        
        # Проверка ответа
        if not response:
            response = "Извините, не удалось сгенерировать ответ."
        
        # Отправляем ответ с обработкой ошибок
        try:
            # Разбиваем длинные сообщения
            if len(response) > 4096:
                # Разбиваем на части по 4096 символов
                for i in range(0, len(response), 4096):
                    await update.message.reply_text(response[i:i+4096])
            else:
                await update.message.reply_text(response)
        except Exception as e:
            logger.error(f"Ошибка отправки сообщения: {e}")
            await update.message.reply_text("✅ Ответ сгенерирован, но возникла ошибка при отправке.")
            
    except Exception as e:
        logger.error(f"Ошибка в handle_message: {e}")
        try:
            await update.message.reply_text("❌ Произошла ошибка при обработке вашего сообщения. Попробуйте позже.")
        except:
            pass

# Получение username бота при запуске
async def post_init(application: Application) -> None:
    global BOT_USERNAME
    try:
        bot: Bot = application.bot
        me = await bot.get_me()
        BOT_USERNAME = me.username
        logger.info(f"Бот @{BOT_USERNAME} запущен!")
        logger.info(f"Используется модель: {DEFAULT_MODEL}")
    except Exception as e:
        logger.error(f"Ошибка при инициализации бота: {e}")

# Обработчик ошибок
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error(f"Ошибка обработки обновления {update}: {context.error}")

# Основная функция
def main():
    try:
        # Проверка наличия токенов
        token = os.getenv('TELEGRAM_BOT_TOKEN')
        if not token:
            logger.error("❌ Не найден TELEGRAM_BOT_TOKEN в переменных окружения")
            print("Пожалуйста, установите переменную окружения TELEGRAM_BOT_TOKEN")
            return
            
        if not OPENROUTER_API_KEY:
            logger.warning("⚠️  Не найден OPENROUTER_API_KEY в переменных окружения")
            print("Предупреждение: API ключ OpenRouter не найден. Бот будет работать с ограниченной функциональностью.")
            
        # Создаем приложение бота с настройками таймаута
        application = (
            Application.builder()
            .token(token)
            .post_init(post_init)
            .connect_timeout(30.0)
            .read_timeout(30.0)
            .write_timeout(30.0)
            .pool_timeout(30.0)
            .build()
        )
        
        # Регистрируем обработчик ошибок
        application.add_error_handler(error_handler)
        
        # Регистрируем обработчики
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("help", help_command))
        application.add_handler(CommandHandler("setup", setup_command))
        application.add_handler(CommandHandler("profile", profile_command))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        
        # Запускаем бота
        logger.info("Бот запущен...")
        print(f"✅ Бот успешно запущен!")
        print(f"🤖 Используется модель: {DEFAULT_MODEL}")
        if BOT_USERNAME:
            print(f"🔗 Username бота: @{BOT_USERNAME}")
        application.run_polling(drop_pending_updates=True)
        
    except Exception as e:
        logger.error(f"Критическая ошибка при запуске бота: {e}")
        print(f"❌ Критическая ошибка при запуске бота: {e}")

if __name__ == '__main__':
    # Установка правильной кодировки для Windows
    import sys
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    
    main()