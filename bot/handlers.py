"""Telegram-обработчики команд и сообщений."""

import logging
import os
import tempfile
from pathlib import Path

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes

from bot.config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_MB
from rag.document_loader import DocumentLoadError
from rag.llm_client import OllamaConnectionError, OllamaTimeoutError
from rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)

# Максимальная длина сообщения Telegram
_TG_MAX_LENGTH = 4096

_START_TEXT = (
    "Привет! Я RAG-ассистент. Загрузи документы, и я буду отвечать "
    "на вопросы по их содержимому.\n\n"
    "Поддерживаемые форматы: PDF, DOCX, TXT, MD\n\n"
    "Команды:\n"
    "/help — подробная инструкция\n"
    "/stats — статистика загруженных документов\n"
    "/clear — очистить все мои документы\n"
    "/search — поиск в интернете\n"
    "/websearch — поиск в интернете + документы\n\n"
    "Просто отправь файл, а затем задай вопрос!"
)

_HELP_TEXT = (
    "Как пользоваться ботом:\n\n"
    "1. Отправьте файл (PDF, DOCX, TXT или MD) — бот извлечёт текст "
    "и сохранит в базу знаний.\n"
    "2. Задайте вопрос текстовым сообщением — бот найдёт релевантные "
    "фрагменты и сгенерирует ответ.\n\n"
    "Команды:\n"
    "/start — приветствие\n"
    "/help — эта инструкция\n"
    "/stats — количество чанков в вашей базе\n"
    "/clear — удалить все ваши документы\n"
    "/search <запрос> — поиск в интернете\n"
    "/websearch <запрос> — ответ из документов + интернет\n\n"
    f"Лимиты:\n"
    f"• Максимальный размер файла: {MAX_FILE_SIZE_MB} МБ\n"
    f"• Форматы: {', '.join(ALLOWED_EXTENSIONS)}\n\n"
    "LLM работает локально через Ollama — ваши данные не покидают сервер."
)

_OLLAMA_ERROR = (
    "LLM недоступна. Убедитесь, что Ollama запущена и модель загружена."
)


def _split_message(text: str) -> list[str]:
    """Разбить длинное сообщение на части по лимиту Telegram."""
    if len(text) <= _TG_MAX_LENGTH:
        return [text]
    parts: list[str] = []
    while text:
        if len(text) <= _TG_MAX_LENGTH:
            parts.append(text)
            break
        # Ищем перенос строки вблизи лимита
        cut = text.rfind("\n", 0, _TG_MAX_LENGTH)
        if cut == -1 or cut < _TG_MAX_LENGTH // 2:
            cut = _TG_MAX_LENGTH
        parts.append(text[:cut])
        text = text[cut:].lstrip("\n")
    return parts


async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /start."""
    await update.message.reply_text(_START_TEXT)


async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /help."""
    await update.message.reply_text(_HELP_TEXT)


async def stats_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /stats — статистика документов пользователя."""
    pipeline: RAGPipeline = context.bot_data["pipeline"]
    user_id = update.effective_user.id
    try:
        count = pipeline.vector_store.get_doc_count(user_id)
        await update.message.reply_text(
            f"В вашей базе знаний: {count} чанков."
        )
    except Exception as e:
        logger.error("Ошибка при получении статистики: %s", e)
        await update.message.reply_text("Не удалось получить статистику.")


async def clear_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /clear — очистка коллекции пользователя."""
    pipeline: RAGPipeline = context.bot_data["pipeline"]
    user_id = update.effective_user.id
    try:
        pipeline.vector_store.delete_collection(user_id)
        await update.message.reply_text(
            "Ваша база знаний очищена."
        )
    except Exception as e:
        logger.error("Ошибка при очистке коллекции: %s", e)
        await update.message.reply_text("Не удалось очистить базу знаний.")


async def search_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик /search — поиск в интернете."""
    pipeline: RAGPipeline = context.bot_data["pipeline"]
    query = " ".join(context.args) if context.args else ""
    if not query.strip():
        await update.message.reply_text(
            "Укажите запрос после команды.\nПример: /search что такое RAG"
        )
        return

    await update.message.chat.send_action(ChatAction.TYPING)
    try:
        answer = await pipeline.web_answer(query.strip())
        for part in _split_message(answer):
            await update.message.reply_text(part, disable_web_page_preview=True)
    except (OllamaConnectionError, OllamaTimeoutError) as e:
        logger.error("Ошибка Ollama при веб-поиске: %s", e)
        await update.message.reply_text(_OLLAMA_ERROR)
    except Exception as e:
        logger.error("Ошибка при веб-поиске: %s", e, exc_info=True)
        await update.message.reply_text("Произошла ошибка при поиске. Попробуйте позже.")


async def websearch_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик /websearch — документы + интернет."""
    pipeline: RAGPipeline = context.bot_data["pipeline"]
    user_id = update.effective_user.id
    query = " ".join(context.args) if context.args else ""
    if not query.strip():
        await update.message.reply_text(
            "Укажите запрос после команды.\n"
            "Пример: /websearch подробнее про тему из моего документа"
        )
        return

    await update.message.chat.send_action(ChatAction.TYPING)
    try:
        answer = await pipeline.combined_answer(user_id, query.strip())
        for part in _split_message(answer):
            await update.message.reply_text(part, disable_web_page_preview=True)
    except (OllamaConnectionError, OllamaTimeoutError) as e:
        logger.error("Ошибка Ollama при комбинированном поиске: %s", e)
        await update.message.reply_text(_OLLAMA_ERROR)
    except Exception as e:
        logger.error("Ошибка при комбинированном поиске: %s", e, exc_info=True)
        await update.message.reply_text("Произошла ошибка при поиске. Попробуйте позже.")


async def document_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик загрузки файлов."""
    pipeline: RAGPipeline = context.bot_data["pipeline"]
    user_id = update.effective_user.id
    document = update.message.document

    if document is None:
        await update.message.reply_text("Не удалось получить файл.")
        return

    # Проверка размера
    if document.file_size and document.file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
        await update.message.reply_text(
            f"Файл слишком большой. Максимум: {MAX_FILE_SIZE_MB} МБ."
        )
        return

    # Проверка расширения
    filename = document.file_name or "unknown"
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        await update.message.reply_text(
            f"Неподдерживаемый формат: {ext}\n"
            f"Поддерживаемые: {', '.join(ALLOWED_EXTENSIONS)}"
        )
        return

    await update.message.chat.send_action(ChatAction.TYPING)

    # Скачивание во временную директорию
    tmp_path = None
    try:
        file = await document.get_file()
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp_path = tmp.name
            await file.download_to_drive(tmp_path)

        chunk_count = await pipeline.ingest(user_id, tmp_path, filename)
        await update.message.reply_text(
            f"Загружено {chunk_count} чанков из файла «{filename}»."
        )

        # Если файл отправлен с подписью (caption) — ответить на вопрос
        caption = update.message.caption
        if caption and caption.strip():
            await update.message.chat.send_action(ChatAction.TYPING)
            try:
                answer = await pipeline.answer(user_id, caption.strip())
                for part in _split_message(answer):
                    await update.message.reply_text(part)
            except (OllamaConnectionError, OllamaTimeoutError) as e:
                logger.error("Ошибка Ollama: %s", e)
                await update.message.reply_text(_OLLAMA_ERROR)
    except DocumentLoadError as e:
        logger.warning("Ошибка загрузки документа: %s", e)
        await update.message.reply_text(f"Ошибка обработки файла: {e}")
    except Exception as e:
        logger.error("Непредвиденная ошибка при загрузке файла: %s", e, exc_info=True)
        await update.message.reply_text(
            "Произошла ошибка при обработке файла. Попробуйте ещё раз."
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик текстовых сообщений — вопросы к RAG."""
    pipeline: RAGPipeline = context.bot_data["pipeline"]
    user_id = update.effective_user.id
    question = update.message.text

    if not question or not question.strip():
        return

    await update.message.chat.send_action(ChatAction.TYPING)

    try:
        answer = await pipeline.answer(user_id, question.strip())
        for part in _split_message(answer):
            await update.message.reply_text(part)
    except (OllamaConnectionError, OllamaTimeoutError) as e:
        logger.error("Ошибка Ollama: %s", e)
        await update.message.reply_text(_OLLAMA_ERROR)
    except Exception as e:
        logger.error("Ошибка при генерации ответа: %s", e, exc_info=True)
        await update.message.reply_text(
            "Произошла ошибка при генерации ответа. Попробуйте позже."
        )
