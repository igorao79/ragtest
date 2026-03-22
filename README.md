# RAG Telegram Bot

Telegram-бот с RAG (Retrieval-Augmented Generation), отвечающий на вопросы по загруженным документам. Полностью бесплатный локальный стек: Ollama + Llama 3.2:3b + ChromaDB.

## Предварительные требования

- Python 3.11+
- [Ollama](https://ollama.ai) установлен и запущен
- Telegram Bot Token (получить через [@BotFather](https://t.me/BotFather))

## Установка

### 1. Установка Ollama и модели

```bash
# Установить Ollama: https://ollama.ai
ollama pull llama3.2:3b
```

### 2. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 3. Настройка

```bash
cp .env.example .env
# Отредактировать .env — вставить TELEGRAM_BOT_TOKEN
```

### 4. Запуск

```bash
# Убедитесь, что Ollama запущена
ollama serve

# В другом терминале:
python -m bot.main
```

## Docker-вариант

```bash
docker-compose up -d
docker exec ollama ollama pull llama3.2:3b
```

## Использование

1. Отправьте боту файл (PDF, DOCX, TXT, MD) — он извлечёт текст и сохранит в базу знаний.
2. Задайте вопрос текстовым сообщением — бот найдёт релевантные фрагменты и сгенерирует ответ.

### Команды

| Команда  | Описание                              |
|----------|---------------------------------------|
| `/start` | Приветствие и краткая инструкция      |
| `/help`  | Подробная инструкция                  |
| `/stats` | Количество чанков в вашей базе        |
| `/clear` | Удалить все ваши документы            |

### Поддерживаемые форматы

- `.pdf` — PDF-документы
- `.docx` — Word-документы
- `.txt` — текстовые файлы
- `.md` — Markdown-файлы

Максимальный размер файла: 20 МБ.

## Архитектура

```
Telegram ←→ Bot Server (python-telegram-bot)
                │
                ├── /start, /help → приветствие, инструкции
                ├── файл → DocumentLoader → Chunker → ChromaDB
                ├── /clear → очистка коллекции пользователя
                └── текст → RAGPipeline → ответ
                        │
                        ├── 1. Embed вопроса → ChromaDB similarity search
                        ├── 2. Формирование prompt с контекстом
                        └── 3. Ollama API → генерация ответа
```

## Переменные окружения

| Переменная          | По умолчанию                | Описание                    |
|---------------------|-----------------------------|-----------------------------|
| `TELEGRAM_BOT_TOKEN`| —                           | Токен Telegram-бота         |
| `OLLAMA_BASE_URL`   | `http://localhost:11434`    | URL Ollama API              |
| `OLLAMA_MODEL`      | `llama3.2:3b`               | Модель Ollama               |
| `CHUNK_SIZE`        | `512`                       | Размер чанка (символы)      |
| `CHUNK_OVERLAP`     | `64`                        | Перекрытие чанков (символы) |
| `TOP_K`             | `4`                         | Кол-во результатов поиска   |
| `MAX_FILE_SIZE_MB`  | `20`                        | Макс. размер файла (МБ)    |
