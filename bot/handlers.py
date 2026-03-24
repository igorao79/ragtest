"""Telegram-обработчики команд и сообщений."""

import logging
import os
import tempfile
import time
from pathlib import Path

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import ContextTypes

from bot.config import (
    AGENT_MODE,
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE_MB,
    OLLAMA_VISION_MODEL,
    RATE_LIMIT_MESSAGES,
    RATE_LIMIT_WINDOW,
)
from rag.document_loader import DocumentLoadError
from rag.llm_client import OllamaConnectionError, OllamaTimeoutError
from rag.pipeline import RAGPipeline
from rag.rate_limiter import RateLimiter
from rag.whisper_client import WhisperError

logger = logging.getLogger(__name__)

_TG_MAX_LENGTH = 4096

_rate_limiter = RateLimiter(
    max_requests=RATE_LIMIT_MESSAGES, window_seconds=RATE_LIMIT_WINDOW
)

_START_TEXT = (
    "👋 *Привет\\! Я RAG\\-ассистент\\.*\n\n"
    "Загрузи документы, и я буду отвечать на вопросы по их содержимому\\.\n\n"
    "📄 *Форматы:* PDF, DOCX, TXT, MD, CSV, XLSX\n"
    "🖼 *Изображения:* отправь фото для OCR\\-анализа\n"
    "🎤 *Голос:* отправь голосовое — распознаю и отвечу\n\n"
    "🗂 *Сессии — начни с создания:*\n"
    "`/create название` — создать сессию\n"
    "`/sessions` — список сессий\n"
    "`/switch название` — переключиться\n\n"
    "_Просто отправь файл, а затем задай вопрос\\!_"
)

_HELP_TEXT = (
    "📖 *Как пользоваться ботом:*\n\n"
    "1\\. Создайте сессию: `/create работа`\n"
    "2\\. Отправьте файл — бот сохранит в текущую сессию\\.\n"
    "3\\. Задайте вопрос — бот сам выберет инструмент\\.\n"
    "4\\. Фото → OCR, голосовое → распознавание речи\\.\n"
    "5\\. Бот помнит контекст — можно спрашивать «подробнее»\\.\n\n"
    "*🗂 Сессии:*\n"
    "`/create название` — создать и переключиться\n"
    "`/sessions` — список всех сессий\n"
    "`/switch название` — переключиться\n"
    "`/clear название` — удалить сессию\n\n"
    "*📄 Документы:*\n"
    "`/files` — список файлов в сессии\n"
    "`/delete имя_файла` — удалить файл\n"
    "`/url ссылка` — загрузить веб\\-страницу\n"
    "`/summary` — пересказ документов\n"
    "`/stats` — статистика\n\n"
    f"*Лимиты:*\n"
    f"• Макс\\. размер файла: {MAX_FILE_SIZE_MB} МБ\n"
    f"• Форматы: {', '.join(ALLOWED_EXTENSIONS)}\n"
    f"• Сессии удаляются через 7 дней неактивности\n\n"
    "_LLM работает локально — ваши данные не покидают сервер\\._"
)

_OLLAMA_ERROR = "⚠️ LLM недоступна\\. Убедитесь, что Ollama запущена\\."
_RATE_LIMIT_ERROR = "⏳ Слишком много запросов\\. Подождите немного\\."


def _split_message(text: str) -> list[str]:
    """Разбить длинное сообщение на части по лимиту Telegram."""
    if len(text) <= _TG_MAX_LENGTH:
        return [text]
    parts: list[str] = []
    while text:
        if len(text) <= _TG_MAX_LENGTH:
            parts.append(text)
            break
        cut = text.rfind("\n", 0, _TG_MAX_LENGTH)
        if cut == -1 or cut < _TG_MAX_LENGTH // 2:
            cut = _TG_MAX_LENGTH
        parts.append(text[:cut])
        text = text[cut:].lstrip("\n")
    return parts


async def _safe_reply_md(message, text: str) -> None:
    """Отправить сообщение с MarkdownV2, fallback на plain text."""
    for part in _split_message(text):
        try:
            await message.reply_text(part, parse_mode=ParseMode.MARKDOWN_V2)
        except Exception:
            await message.reply_text(part)


async def _safe_reply(message, text: str) -> None:
    """Отправить сообщение с Markdown, fallback на plain text."""
    for part in _split_message(text):
        try:
            await message.reply_text(
                part, parse_mode=ParseMode.MARKDOWN,
                disable_web_page_preview=True,
            )
        except Exception:
            await message.reply_text(part, disable_web_page_preview=True)


def _check_rate_limit(user_id: int) -> bool:
    """Проверить rate limit."""
    return _rate_limiter.is_allowed(user_id)


def _get_session(pipeline: "RAGPipeline", user_id: int) -> str | None:
    """Получить имя активной сессии пользователя."""
    return pipeline.sessions.get_collection_name(user_id)


async def _ensure_session(
    pipeline: "RAGPipeline", user_id: int, message
) -> str | None:
    """Проверить наличие сессии. Если нет — мигрировать старые данные или подсказать.

    Возвращает имя коллекции (или None для legacy).
    """
    col = _get_session(pipeline, user_id)
    if col is not None:
        return col

    # Нет активной сессии — проверяем, есть ли старые данные без сессии
    legacy_count = pipeline.vector_store.get_doc_count(user_id, None)

    if legacy_count > 0:
        # Автоматически мигрируем в сессию "default"
        pipeline.sessions.create(user_id, "default")
        # Данные уже в коллекции user_{id} — а сессия "default" создаст user_{id}_default
        # Нужно перенести данные
        _migrate_legacy_data(pipeline, user_id)
        await message.reply_text(
            f"📦 Найдено {legacy_count} чанков из прошлых загрузок.\n"
            f"Перенесено в сессию «default».\n\n"
            f"Создайте новые сессии: `/create название`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return "default"

    # Нет данных, нет сессии — просто подсказка
    return None


def _migrate_legacy_data(pipeline: "RAGPipeline", user_id: int) -> None:
    """Перенести данные из legacy-коллекции user_{id} в сессию default."""
    try:
        old_collection = pipeline.vector_store.get_or_create_collection(user_id, None)
        if old_collection.count() == 0:
            return

        # Получаем всё из старой коллекции
        data = old_collection.get(include=["documents", "metadatas"])
        ids = data.get("ids", [])
        docs = data.get("documents", [])
        metas = data.get("metadatas", [])

        if not ids:
            return

        # Добавляем в новую коллекцию (default)
        new_collection = pipeline.vector_store.get_or_create_collection(user_id, "default")
        new_collection.upsert(ids=ids, documents=docs, metadatas=metas)

        # Удаляем старую
        pipeline.vector_store.delete_collection(user_id, None)
        logger.info("Мигрировано %d чанков в сессию default для user_%d", len(ids), user_id)
    except Exception as e:
        logger.error("Ошибка миграции legacy данных: %s", e)


def _get_inline_buttons(user_id: int) -> InlineKeyboardMarkup:
    """Создать inline-кнопки после ответа."""
    buttons = [
        [
            InlineKeyboardButton("🔍 Искать в интернете", callback_data="web_search"),
            InlineKeyboardButton("📝 Подробнее", callback_data="more_detail"),
        ],
    ]
    return InlineKeyboardMarkup(buttons)


# === Command Handlers ===


async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик /start."""
    await update.message.reply_text(
        _START_TEXT, parse_mode=ParseMode.MARKDOWN_V2
    )


async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик /help."""
    await update.message.reply_text(
        _HELP_TEXT, parse_mode=ParseMode.MARKDOWN_V2
    )


async def stats_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик /stats."""
    pipeline: RAGPipeline = context.bot_data["pipeline"]
    user_id = update.effective_user.id
    col = _get_session(pipeline, user_id)
    try:
        count = pipeline.vector_store.get_doc_count(user_id, col)
        files = pipeline.vector_store.get_file_list(user_id, col)
        session_label = pipeline.sessions.get_active_display(user_id)
        text = f"📊 Сессия «{session_label}»: *{count}* чанков из *{len(files)}* файлов."
        await _safe_reply(update.message, text)
    except Exception as e:
        logger.error("Ошибка при получении статистики: %s", e)
        await update.message.reply_text("Не удалось получить статистику.")


async def files_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик /files."""
    pipeline: RAGPipeline = context.bot_data["pipeline"]
    user_id = update.effective_user.id
    col = _get_session(pipeline, user_id)
    try:
        files = pipeline.vector_store.get_file_list(user_id, col)
        if not files:
            await update.message.reply_text("📂 Ваша база знаний пуста.")
            return
        col_label = f" (коллекция: {col})" if col else ""
        lines = [f"📂 *Загруженные файлы{col_label}:*\n"]
        for f in files:
            lines.append(f"• `{f['name']}` — {f['chunks']} чанков")
        lines.append(f"\n_Всего: {sum(f['chunks'] for f in files)} чанков_")
        await _safe_reply(update.message, "\n".join(lines))
    except Exception as e:
        logger.error("Ошибка при получении списка файлов: %s", e)
        await update.message.reply_text("Не удалось получить список файлов.")


async def delete_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик /delete <filename>."""
    pipeline: RAGPipeline = context.bot_data["pipeline"]
    user_id = update.effective_user.id
    col = _get_session(pipeline, user_id)
    filename = " ".join(context.args) if context.args else ""

    if not filename.strip():
        await update.message.reply_text(
            "Укажите имя файла.\nПример: /delete document.pdf\n"
            "Список файлов: /files"
        )
        return

    try:
        deleted = pipeline.vector_store.delete_file(user_id, filename.strip(), col)
        if deleted:
            pipeline.cache.invalidate_user(user_id)
            await update.message.reply_text(
                f"🗑 Удалено {deleted} чанков файла «{filename.strip()}»."
            )
        else:
            await update.message.reply_text(
                f"Файл «{filename.strip()}» не найден. Проверьте /files."
            )
    except Exception as e:
        logger.error("Ошибка при удалении файла: %s", e)
        await update.message.reply_text("Не удалось удалить файл.")


async def url_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик /url <ссылка>."""
    pipeline: RAGPipeline = context.bot_data["pipeline"]
    user_id = update.effective_user.id
    col = _get_session(pipeline, user_id)
    url = " ".join(context.args) if context.args else ""

    if not url.strip():
        await update.message.reply_text(
            "Укажите URL.\nПример: /url https://example.com/article"
        )
        return

    await update.message.chat.send_action(ChatAction.TYPING)
    try:
        chunk_count = await pipeline.ingest_url(user_id, url.strip(), col)
        await update.message.reply_text(
            f"🌐 Загружено {chunk_count} чанков с {url.strip()}"
        )
    except DocumentLoadError as e:
        await update.message.reply_text(f"Ошибка загрузки URL: {e}")
    except Exception as e:
        logger.error("Ошибка при загрузке URL: %s", e, exc_info=True)
        await update.message.reply_text("Не удалось загрузить страницу.")


async def summary_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик /summary."""
    pipeline: RAGPipeline = context.bot_data["pipeline"]
    user_id = update.effective_user.id
    col = _get_session(pipeline, user_id)

    if not _check_rate_limit(user_id):
        await _safe_reply_md(update.message, _RATE_LIMIT_ERROR)
        return

    await update.message.chat.send_action(ChatAction.TYPING)
    try:
        answer = await pipeline.summarize(user_id, col)
        await _safe_reply(update.message, answer)
    except (OllamaConnectionError, OllamaTimeoutError):
        await _safe_reply_md(update.message, _OLLAMA_ERROR)
    except Exception as e:
        logger.error("Ошибка суммаризации: %s", e, exc_info=True)
        await update.message.reply_text("Ошибка при суммаризации.")


async def clear_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик /clear <название> — удаление сессии."""
    pipeline: RAGPipeline = context.bot_data["pipeline"]
    user_id = update.effective_user.id
    name = " ".join(context.args) if context.args else ""

    if not name.strip():
        await update.message.reply_text(
            "Укажите имя сессии для удаления.\n"
            "Пример: `/clear работа`\n"
            "Список сессий: /sessions",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    try:
        # Удаляем коллекцию ChromaDB
        col_name = pipeline.sessions._sanitize(name.strip())
        pipeline.vector_store.delete_collection(user_id, col_name)
        pipeline.cache.invalidate_user(user_id)
        pipeline.conversation.clear(user_id)

        # Удаляем сессию из менеджера
        deleted = pipeline.sessions.delete(user_id, name.strip())
        if deleted:
            active = pipeline.sessions.get_active_display(user_id)
            await update.message.reply_text(
                f"🗑 Сессия «{name.strip()}» удалена.\n"
                f"Активная сессия: {active}"
            )
        else:
            await update.message.reply_text(
                f"Сессия «{name.strip()}» не найдена. Проверьте /sessions."
            )
    except Exception as e:
        logger.error("Ошибка при удалении сессии: %s", e)
        await update.message.reply_text("Не удалось удалить сессию.")


async def create_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик /create <название> — создание сессии."""
    pipeline: RAGPipeline = context.bot_data["pipeline"]
    user_id = update.effective_user.id
    name = " ".join(context.args) if context.args else ""

    if not name.strip():
        await update.message.reply_text(
            "Укажите название сессии.\nПример: `/create работа`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    try:
        session = pipeline.sessions.create(user_id, name.strip())
        # Создаём коллекцию в ChromaDB
        pipeline.vector_store.get_or_create_collection(user_id, session.name)
        count = pipeline.vector_store.get_doc_count(user_id, session.name)
        await update.message.reply_text(
            f"🗂 Сессия «{session.name}» создана и активна ({count} чанков).\n"
            f"Теперь загружайте файлы и задавайте вопросы!"
        )
    except Exception as e:
        logger.error("Ошибка создания сессии: %s", e)
        await update.message.reply_text("Не удалось создать сессию.")


async def switch_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик /switch <название> — переключение сессии."""
    pipeline: RAGPipeline = context.bot_data["pipeline"]
    user_id = update.effective_user.id
    name = " ".join(context.args) if context.args else ""

    if not name.strip():
        await update.message.reply_text(
            "Укажите название сессии.\nПример: `/switch учёба`\n"
            "Список сессий: /sessions",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    session = pipeline.sessions.switch(user_id, name.strip())
    if session:
        count = pipeline.vector_store.get_doc_count(user_id, session.name)
        await update.message.reply_text(
            f"🗂 Переключено на «{session.name}» ({count} чанков)."
        )
    else:
        await update.message.reply_text(
            f"Сессия «{name.strip()}» не найдена.\n"
            f"Создайте: `/create {name.strip()}`\n"
            f"Или посмотрите список: /sessions",
            parse_mode=ParseMode.MARKDOWN,
        )


async def sessions_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик /sessions — список всех сессий."""
    pipeline: RAGPipeline = context.bot_data["pipeline"]
    user_id = update.effective_user.id

    # Автомиграция старых данных
    await _ensure_session(pipeline, user_id, update.message)

    sessions = pipeline.sessions.list_sessions(user_id)
    if not sessions:
        await update.message.reply_text(
            "🗂 У вас нет сессий.\n"
            "Создайте первую: `/create название`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    lines = ["🗂 *Ваши сессии:*\n"]
    for s in sessions:
        marker = " ✅" if s["active"] else ""
        count = pipeline.vector_store.get_doc_count(
            user_id, s["name"]
        )
        # Дней с последней активности
        days_ago = (time.time() - s["last_active"]) / 86400
        if days_ago < 1:
            age = "сегодня"
        else:
            age = f"{int(days_ago)}д назад"
        lines.append(f"• `{s['name']}` — {count} чанков, {age}{marker}")

    lines.append(f"\n_Сессии удаляются через 7 дней неактивности._")
    await _safe_reply(update.message, "\n".join(lines))


async def search_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик /search."""
    pipeline: RAGPipeline = context.bot_data["pipeline"]
    user_id = update.effective_user.id
    query = " ".join(context.args) if context.args else ""

    if not query.strip():
        await update.message.reply_text(
            "Укажите запрос.\nПример: /search что такое RAG"
        )
        return

    if not _check_rate_limit(user_id):
        await _safe_reply_md(update.message, _RATE_LIMIT_ERROR)
        return

    await update.message.chat.send_action(ChatAction.TYPING)
    try:
        answer = await pipeline.web_answer(query.strip())
        await _safe_reply(update.message, answer)
    except (OllamaConnectionError, OllamaTimeoutError):
        await _safe_reply_md(update.message, _OLLAMA_ERROR)
    except Exception as e:
        logger.error("Ошибка веб-поиска: %s", e, exc_info=True)
        await update.message.reply_text("Ошибка при поиске.")


async def websearch_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик /websearch."""
    pipeline: RAGPipeline = context.bot_data["pipeline"]
    user_id = update.effective_user.id
    query = " ".join(context.args) if context.args else ""

    if not query.strip():
        await update.message.reply_text(
            "Укажите запрос.\nПример: /websearch подробнее про тему из документа"
        )
        return

    if not _check_rate_limit(user_id):
        await _safe_reply_md(update.message, _RATE_LIMIT_ERROR)
        return

    await update.message.chat.send_action(ChatAction.TYPING)
    try:
        answer = await pipeline.combined_answer(user_id, query.strip())
        await _safe_reply(update.message, answer)
    except (OllamaConnectionError, OllamaTimeoutError):
        await _safe_reply_md(update.message, _OLLAMA_ERROR)
    except Exception as e:
        logger.error("Ошибка комбинированного поиска: %s", e, exc_info=True)
        await update.message.reply_text("Ошибка при поиске.")


# === Message Handlers ===


async def document_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик загрузки файлов."""
    pipeline: RAGPipeline = context.bot_data["pipeline"]
    user_id = update.effective_user.id
    col = await _ensure_session(pipeline, user_id, update.message)
    document = update.message.document

    if document is None:
        await update.message.reply_text("Не удалось получить файл.")
        return

    if document.file_size and document.file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
        await update.message.reply_text(
            f"Файл слишком большой. Максимум: {MAX_FILE_SIZE_MB} МБ."
        )
        return

    filename = document.file_name or "unknown"
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        await update.message.reply_text(
            f"Неподдерживаемый формат: {ext}\n"
            f"Поддерживаемые: {', '.join(ALLOWED_EXTENSIONS)}"
        )
        return

    await update.message.chat.send_action(ChatAction.TYPING)

    tmp_path = None
    try:
        file = await document.get_file()
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp_path = tmp.name
            await file.download_to_drive(tmp_path)

        chunk_count = await pipeline.ingest(user_id, tmp_path, filename, col)
        col_label = f" в коллекцию «{col}»" if col else ""
        await update.message.reply_text(
            f"✅ Загружено {chunk_count} чанков из файла «{filename}»{col_label}."
        )

        # Если файл отправлен с подписью — ответить на вопрос
        caption = update.message.caption
        if caption and caption.strip():
            await update.message.chat.send_action(ChatAction.TYPING)
            try:
                answer = await pipeline.answer(user_id, caption.strip(), col)
                await _safe_reply(update.message, answer)
            except (OllamaConnectionError, OllamaTimeoutError):
                await _safe_reply_md(update.message, _OLLAMA_ERROR)
    except DocumentLoadError as e:
        logger.warning("Ошибка загрузки документа: %s", e)
        await update.message.reply_text(f"Ошибка обработки файла: {e}")
    except Exception as e:
        logger.error("Ошибка при загрузке файла: %s", e, exc_info=True)
        await update.message.reply_text("Ошибка при обработке файла.")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


async def photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик фотографий — OCR через vision-модель."""
    pipeline: RAGPipeline = context.bot_data["pipeline"]
    user_id = update.effective_user.id

    if not _check_rate_limit(user_id):
        await _safe_reply_md(update.message, _RATE_LIMIT_ERROR)
        return

    photo = update.message.photo[-1]
    caption = update.message.caption or "Опиши что на этом изображении. Извлеки весь текст."

    await update.message.chat.send_action(ChatAction.TYPING)

    tmp_path = None
    try:
        file = await photo.get_file()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp_path = tmp.name
            await file.download_to_drive(tmp_path)

        image_data = Path(tmp_path).read_bytes()
        answer = await pipeline.analyze_image(
            caption, image_data, vision_model=OLLAMA_VISION_MODEL
        )
        await _safe_reply(update.message, answer)
    except (OllamaConnectionError, OllamaTimeoutError):
        await _safe_reply_md(update.message, _OLLAMA_ERROR)
    except Exception as e:
        logger.error("Ошибка OCR: %s", e, exc_info=True)
        await update.message.reply_text("Ошибка при анализе изображения.")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


async def voice_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик голосовых сообщений — распознавание через Whisper."""
    pipeline: RAGPipeline = context.bot_data["pipeline"]
    user_id = update.effective_user.id
    col = _get_session(pipeline, user_id)

    if not _check_rate_limit(user_id):
        await _safe_reply_md(update.message, _RATE_LIMIT_ERROR)
        return

    voice = update.message.voice or update.message.audio
    if not voice:
        return

    await update.message.chat.send_action(ChatAction.TYPING)

    tmp_path = None
    try:
        file = await voice.get_file()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as tmp:
            tmp_path = tmp.name
            await file.download_to_drive(tmp_path)

        # Транскрибируем
        text = await pipeline.transcribe_voice(tmp_path)
        await update.message.reply_text(f"🎤 Распознано: _{text}_", parse_mode=ParseMode.MARKDOWN)

        # Отвечаем на распознанный текст как на обычный вопрос
        await update.message.chat.send_action(ChatAction.TYPING)
        answer = await pipeline.answer(user_id, text, col)
        await _safe_reply(update.message, answer)

    except WhisperError as e:
        logger.warning("Whisper ошибка: %s", e)
        await update.message.reply_text(
            f"🎤 Не удалось распознать речь: {e}\n"
            "Попробуйте написать вопрос текстом."
        )
    except (OllamaConnectionError, OllamaTimeoutError):
        await _safe_reply_md(update.message, _OLLAMA_ERROR)
    except Exception as e:
        logger.error("Ошибка обработки голоса: %s", e, exc_info=True)
        await update.message.reply_text(
            "🎤 Ошибка при обработке голосового сообщения. Напишите вопрос текстом."
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик текстовых сообщений.

    В режиме AGENT_MODE LLM сама решает какой инструмент использовать.
    В обычном режиме — стриминг RAG-ответа.
    """
    pipeline: RAGPipeline = context.bot_data["pipeline"]
    user_id = update.effective_user.id
    question = update.message.text

    if not question or not question.strip():
        return

    col = await _ensure_session(pipeline, user_id, update.message)

    if not _check_rate_limit(user_id):
        await _safe_reply_md(update.message, _RATE_LIMIT_ERROR)
        return

    await update.message.chat.send_action(ChatAction.TYPING)

    try:
        if AGENT_MODE:
            # Агентский режим — LLM выбирает инструмент автоматически
            status_message = None

            async def _on_tool_call(tool_name: str, display_text: str) -> None:
                nonlocal status_message
                try:
                    if status_message:
                        await status_message.delete()
                    status_message = await update.message.reply_text(
                        display_text, parse_mode=ParseMode.MARKDOWN,
                    )
                except Exception:
                    pass

            answer = await pipeline.agent_answer(
                user_id, question.strip(), col,
                on_tool_call=_on_tool_call,
            )

            # Удаляем статус-сообщение
            if status_message:
                try:
                    await status_message.delete()
                except Exception:
                    # Если нет прав на удаление — редактируем в пустоту
                    try:
                        await status_message.edit_text("✅")
                    except Exception:
                        pass

            await _safe_reply(update.message, answer)
            context.user_data["last_question"] = question.strip()
        else:
            # Классический режим — стриминг RAG
            cached = pipeline.cache.get(user_id, question.strip())
            if cached:
                await _safe_reply(update.message, cached)
                return

            sent_message = None
            buffer = ""
            last_sent = ""
            chunk_count = 0

            async for token in pipeline.answer_stream(user_id, question.strip(), col):
                buffer += token
                chunk_count += 1

                if chunk_count % 15 == 0 and buffer.strip() != last_sent:
                    try:
                        if sent_message is None:
                            sent_message = await update.message.reply_text(buffer.strip() + " ▌")
                        else:
                            await sent_message.edit_text(buffer.strip() + " ▌")
                        last_sent = buffer.strip()
                    except Exception:
                        pass

            final_text = buffer.strip()
            if final_text:
                if sent_message is None:
                    sent_message = await update.message.reply_text(final_text)
                else:
                    try:
                        await sent_message.edit_text(final_text)
                    except Exception:
                        pass

                try:
                    context.user_data["last_question"] = question.strip()
                    buttons = _get_inline_buttons(user_id)
                    await sent_message.edit_reply_markup(reply_markup=buttons)
                except Exception:
                    pass

    except (OllamaConnectionError, OllamaTimeoutError):
        await _safe_reply_md(update.message, _OLLAMA_ERROR)
    except Exception as e:
        logger.error("Ошибка генерации: %s", e, exc_info=True)
        await update.message.reply_text("Ошибка при генерации ответа.")


async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик inline-кнопок."""
    query = update.callback_query
    await query.answer()

    pipeline: RAGPipeline = context.bot_data["pipeline"]
    user_id = query.from_user.id
    col = _get_session(pipeline, user_id)
    last_question = context.user_data.get("last_question", "")

    if not last_question:
        await query.message.reply_text("Задайте вопрос ещё раз.")
        return

    if not _check_rate_limit(user_id):
        await query.message.reply_text("⏳ Подождите немного.")
        return

    await query.message.chat.send_action(ChatAction.TYPING)

    try:
        if query.data == "web_search":
            answer = await pipeline.web_answer(last_question)
            await _safe_reply(query.message, f"🔍 *Результаты из интернета:*\n\n{answer}")

        elif query.data == "more_detail":
            detailed_q = f"Ответь подробнее на вопрос: {last_question}"
            answer = await pipeline.answer(user_id, detailed_q, col)
            await _safe_reply(query.message, answer)

    except (OllamaConnectionError, OllamaTimeoutError):
        await _safe_reply_md(query.message, _OLLAMA_ERROR)
    except Exception as e:
        logger.error("Ошибка callback: %s", e, exc_info=True)
        await query.message.reply_text("Произошла ошибка.")
