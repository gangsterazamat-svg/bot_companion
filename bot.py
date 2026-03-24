import os
import json
import logging
from datetime import datetime
from typing import Dict, Optional, List
import asyncio

from telegram import Update, Bot, ReplyKeyboardMarkup, KeyboardButton
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
DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
BOT_USERNAME = None  # Будет определено при запуске

# Доступные модели
AVAILABLE_MODELS = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "nvidia/llama-nemotron-embed-vl-1b-v2:free",
    "minimax/minimax-m2.5:free"
]

# Директории для хранения данных
DATA_DIR = "user_data"
os.makedirs(DATA_DIR, exist_ok=True)

# Файл для хранения системных промтов
SYSTEM_PROMPTS_FILE = os.path.join(DATA_DIR, "system_prompts.json")
USER_INFO_FILE = os.path.join(DATA_DIR, "user_info.json")
USER_MODELS_FILE = os.path.join(DATA_DIR, "user_models.json")

# Создание постоянной клавиатуры с командами
def create_main_keyboard() -> ReplyKeyboardMarkup:
    keyboard = [
        [KeyboardButton("🤖 Настроить бота"), KeyboardButton("🧠 Выбор модели")],
        [KeyboardButton("👤 Мой профиль"), KeyboardButton("❓ Помощь")],
        [KeyboardButton("🔄 Начать заново")]
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=False)

# Создание клавиатуры выбора модели
def create_model_keyboard() -> ReplyKeyboardMarkup:
    keyboard = []
    for i, model in enumerate(AVAILABLE_MODELS):
        if i % 2 == 0:
            keyboard.append([KeyboardButton(model)])
        else:
            keyboard[-1].append(KeyboardButton(model))
    
    # Добавляем кнопку возврата
    keyboard.append([KeyboardButton("🔙 Назад")])
    
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=False)

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

def load_user_models() -> Dict:
    try:
        if os.path.exists(USER_MODELS_FILE):
            with open(USER_MODELS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Ошибка загрузки пользовательских моделей: {e}")
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

def save_user_models(user_models: Dict):
    try:
        with open(USER_MODELS_FILE, 'w', encoding='utf-8') as f:
            json.dump(user_models, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Ошибка сохранения пользовательских моделей: {e}")

# Глобальные переменные для данных
SYSTEM_PROMPTS = load_system_prompts()
USER_INFO = load_user_info()
USER_MODELS = load_user_models()

# Получение модели пользователя
def get_user_model(user_id: str) -> str:
    return USER_MODELS.get(user_id, DEFAULT_MODEL)

# Сохранение модели пользователя
def set_user_model(user_id: str, model: str):
    USER_MODELS[user_id] = model
    save_user_models(USER_MODELS)

# Разделение длинного текста на части
def split_long_message(text: str, max_length: int = 4000) -> List[str]:
    """Разделяет длинный текст на части, сохраняя целостность предложений"""
    if len(text) <= max_length:
        return [text]
    
    parts = []
    current_part = ""
    
    # Разбиваем текст на предложения
    sentences = []
    current_sentence = ""
    
    for char in text:
        current_sentence += char
        if char in '.!?':
            sentences.append(current_sentence.strip())
            current_sentence = ""
    
    # Добавляем остаток, если есть
    if current_sentence.strip():
        sentences.append(current_sentence.strip())
    
    # Собираем части
    for sentence in sentences:
        if len(current_part + sentence) <= max_length:
            current_part += sentence + " "
        else:
            if current_part:
                parts.append(current_part.strip())
            current_part = sentence + " "
    
    # Добавляем последнюю часть
    if current_part:
        parts.append(current_part.strip())
    
    # Если какая-то часть все равно слишком длинная, разбиваем по словам
    final_parts = []
    for part in parts:
        if len(part) <= max_length:
            final_parts.append(part)
        else:
            # Разбиваем по словам
            words = part.split()
            current_chunk = ""
            for word in words:
                if len(current_chunk + word) <= max_length:
                    current_chunk += word + " "
                else:
                    if current_chunk:
                        final_parts.append(current_chunk.strip())
                    current_chunk = word + " "
            if current_chunk:
                final_parts.append(current_chunk.strip())
    
    return final_parts if final_parts else [text[:max_length]]

# Генерация текста через OpenRouter API
async def generate_text(prompt: str, system_message: str = "", user_id: str = "") -> str:
    # Проверка наличия API ключа
    if not OPENROUTER_API_KEY:
        return "❌ Не настроен API ключ OpenRouter. Обратитесь к администратору бота."
    
    # Получаем модель пользователя
    model = get_user_model(user_id)
    
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
            "model": model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 2000  # Увеличено для больших моделей
        }
        
        # Настройка HTTP клиента с таймаутами
        timeout = httpx.Timeout(60.0, connect=20.0)  # Увеличенные таймауты
        
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
                        await asyncio.sleep(10)  # Увеличено время ожидания
                        continue
                    else:
                        return "❌ Превышено время ожидания ответа от API. Попробуйте позже."
                        
                except httpx.ConnectError as e:
                    logger.error(f"Ошибка подключения к OpenRouter API (попытка {attempt + 1}): {e}")
                    if attempt < 2:
                        await asyncio.sleep(10)
                        continue
                    else:
                        return "❌ Ошибка подключения к API. Проверьте интернет-соединение."
                        
                except httpx.NetworkError as e:
                    logger.error(f"Сетевая ошибка (попытка {attempt + 1}): {e}")
                    if attempt < 2:
                        await asyncio.sleep(10)
                        continue
                    else:
                        return "❌ Сетевая ошибка. Проверьте подключение к интернету."
                        
                except Exception as e:
                    logger.error(f"Неожиданная ошибка при запросе к API (попытка {attempt + 1}): {e}")
                    if attempt < 2:
                        await asyncio.sleep(10)
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
            f"Использую модель: {get_user_model(user_id)}\n\n"
            "Используйте кнопки внизу экрана для навигации по боту 🔽"
        )
        
        # Отправляем сообщение с постоянной клавиатурой
        await update.message.reply_text(
            welcome_message, 
            reply_markup=create_main_keyboard()
        )
    except Exception as e:
        logger.error(f"Ошибка в start: {e}")

# Обработчик команды /help
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if not update.message:
            return
            
        user_id = str(update.effective_user.id) if update.effective_user else "unknown"
        current_model = get_user_model(user_id)
        
        help_text = (
            "🤖 Справка по боту\n\n"
            "В личных сообщениях:\n"
            "Просто напишите мне сообщение и я отвечу вам\n\n"
            "В группах:\n"
            f"Упомяните меня @{BOT_USERNAME} и ваше сообщение, чтобы я ответил\n\n"
            "Команды:\n"
            "🤖 Настроить бота - Настроить системный промт и информацию о себе\n"
            "🧠 Выбор модели - Выбрать модель ИИ для общения\n"
            "👤 Мой профиль - Посмотреть/изменить ваш профиль\n"
            "❓ Помощь - Показать эту справку\n"
            "🔄 Начать заново - Перезапустить настройку бота\n\n"
            f"Текущая модель: {current_model}\n"
            f"Доступные модели: {', '.join(AVAILABLE_MODELS)}"
        )
        
        await update.message.reply_text(help_text, reply_markup=create_main_keyboard())
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
        
        await update.message.reply_text(setup_message, reply_markup=create_main_keyboard())
    except Exception as e:
        logger.error(f"Ошибка в setup_command: {e}")

# Обработчик команды /profile
async def profile_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if not update.effective_user or not update.message:
            return
            
        user_id = str(update.effective_user.id)
        user_info = get_user_info(user_id)
        current_model = get_user_model(user_id)
        
        profile_message = (
            "👤 Ваш профиль:\n\n"
            f"Модель ИИ: {current_model}\n\n"
        )
        
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
        await update.message.reply_text(profile_message, reply_markup=create_main_keyboard())
    except Exception as e:
        logger.error(f"Ошибка в profile_command: {e}")

# Обработчик команды сброса
async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if not update.effective_user or not update.message:
            return
            
        user_id = str(update.effective_user.id)
        
        # Сбрасываем состояние пользователя
        if user_id in USER_STATES:
            USER_STATES.pop(user_id, None)
        
        reset_message = (
            "🔄 Настройки бота сброшены!\n\n"
            "Теперь вы можете заново настроить бота с помощью кнопки '🤖 Настроить бота'"
        )
        
        await update.message.reply_text(reset_message, reply_markup=create_main_keyboard())
    except Exception as e:
        logger.error(f"Ошибка в reset_command: {e}")

# Обработчик выбора модели
async def model_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if not update.effective_user or not update.message:
            return
            
        user_id = str(update.effective_user.id)
        current_model = get_user_model(user_id)
        
        model_message = (
            f"🧠 Выбор модели ИИ\n\n"
            f"Текущая модель: {current_model}\n\n"
            "Выберите одну из доступных моделей:"
        )
        
        USER_STATES[user_id] = "MODEL_SELECTION"
        await update.message.reply_text(model_message, reply_markup=create_model_keyboard())
    except Exception as e:
        logger.error(f"Ошибка в model_command: {e}")

# Состояния пользователей
USER_STATES = {}

# Обработчик текстовых сообщений (включая нажатия кнопок)
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        # Проверка наличия необходимых данных
        if not update.effective_user or not update.message or not update.message.text:
            return
            
        user_id = str(update.effective_user.id)
        message_text = update.message.text.strip()
        chat_type = update.effective_chat.type if update.effective_chat else 'private'
        
        # Обработка нажатий кнопок
        if message_text == "🤖 Настроить бота":
            await setup_command(update, context)
            return
        elif message_text == "🧠 Выбор модели":
            await model_command(update, context)
            return
        elif message_text == "👤 Мой профиль":
            await profile_command(update, context)
            return
        elif message_text == "❓ Помощь":
            await help_command(update, context)
            return
        elif message_text == "🔄 Начать заново":
            await reset_command(update, context)
            return
        elif message_text == "🔙 Назад":
            # Возвращаемся к основной клавиатуре
            USER_STATES.pop(user_id, None)
            await update.message.reply_text("Возвращаемся к основному меню", reply_markup=create_main_keyboard())
            return
        
        # Обработка состояния выбора модели
        if user_id in USER_STATES and USER_STATES[user_id] == "MODEL_SELECTION":
            if message_text in AVAILABLE_MODELS:
                set_user_model(user_id, message_text)
                USER_STATES.pop(user_id, None)
                await update.message.reply_text(
                    f"✅ Модель успешно изменена на: {message_text}", 
                    reply_markup=create_main_keyboard()
                )
                return
            elif message_text == "🔙 Назад":
                USER_STATES.pop(user_id, None)
                await update.message.reply_text("Возвращаемся к основному меню", reply_markup=create_main_keyboard())
                return
            else:
                await update.message.reply_text(
                    "Пожалуйста, выберите модель из списка или нажмите 'Назад'", 
                    reply_markup=create_model_keyboard()
                )
                return
        
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
            await update.message.reply_text(
                "Пожалуйста, введите текст сообщения или используйте кнопки.", 
                reply_markup=create_main_keyboard()
            )
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
                await update.message.reply_text("Как вас зовут? (или 'пропустить')", reply_markup=create_main_keyboard())
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
                await update.message.reply_text("Сколько вам лет? (или 'пропустить')", reply_markup=create_main_keyboard())
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
                await update.message.reply_text("Какими у вас интересы? (или 'пропустить')", reply_markup=create_main_keyboard())
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
                await update.message.reply_text("Опишите ваш характер/личность: (или 'пропустить')", reply_markup=create_main_keyboard())
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
                completion_message = (
                    "✅ Настройка завершена! Теперь я знаю о вас больше.\n"
                    "Используйте кнопки внизу для дальнейшей работы с ботом."
                )
                await update.message.reply_text(completion_message, reply_markup=create_main_keyboard())
                return
            
            elif state == "PROFILE_UPDATE_CONFIRM":
                if message_text.lower() in ['да', 'yes', 'y']:
                    USER_STATES[user_id] = "SETUP_NAME"
                    await update.message.reply_text("Как вас зовут? (или 'пропустить')", reply_markup=create_main_keyboard())
                else:
                    USER_STATES.pop(user_id, None)
                    await update.message.reply_text("Хорошо, оставим как есть.", reply_markup=create_main_keyboard())
                return
        
        # Генерация ответа
        system_prompt = get_user_system_prompt(user_id)
        
        # Добавляем контекст в промпт
        full_prompt = message_text
        
        await update.message.reply_text("🤔 Думаю...", reply_markup=create_main_keyboard())
        response = await generate_text(full_prompt, system_prompt, user_id)
        
        # Проверка ответа
        if not response:
            response = "Извините, не удалось сгенерировать ответ."
        
        # Отправляем ответ с обработкой ошибок и разделением на части
        try:
            # Разбиваем длинные сообщения
            message_parts = split_long_message(response)
            
            for i, part in enumerate(message_parts):
                if i == 0:
                    # Первое сообщение с клавиатурой
                    await update.message.reply_text(part, reply_markup=create_main_keyboard())
                else:
                    # Последующие сообщения без клавиатуры
                    await update.message.reply_text(part)
                    
                # Небольшая задержка между сообщениями для лучшего UX
                if len(message_parts) > 1 and i < len(message_parts) - 1:
                    await asyncio.sleep(0.5)
                    
        except Exception as e:
            logger.error(f"Ошибка отправки сообщения: {e}")
            await update.message.reply_text(
                "✅ Ответ сгенерирован, но возникла ошибка при отправке.", 
                reply_markup=create_main_keyboard()
            )
            
    except Exception as e:
        logger.error(f"Ошибка в handle_message: {e}")
        try:
            await update.message.reply_text(
                "❌ Произошла ошибка при обработке вашего сообщения. Попробуйте позже.", 
                reply_markup=create_main_keyboard()
            )
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
        logger.info(f"Доступные модели: {AVAILABLE_MODELS}")
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
            .read_timeout(60.0)  # Увеличен таймаут чтения
            .write_timeout(60.0)  # Увеличен таймаут записи
            .pool_timeout(30.0)
            .build()
        )
        
        # Регистрируем обработчик ошибок
        application.add_error_handler(error_handler)
        
        # Регистрируем обработчики команд
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("help", help_command))
        application.add_handler(CommandHandler("setup", setup_command))
        application.add_handler(CommandHandler("profile", profile_command))
        application.add_handler(CommandHandler("reset", reset_command))
        
        # Регистрируем обработчик текстовых сообщений (включая кнопки)
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        
        # Запускаем бота
        logger.info("Бот запущен...")
        print(f"✅ Бот успешно запущен!")
        print(f"🤖 Доступные модели: {AVAILABLE_MODELS}")
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