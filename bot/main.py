"""Entrypoint: запуск Telegram-бота с RAG."""

import logging
import sys

from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Инициализация и запуск бота."""
    # Загрузка .env, если есть python-dotenv
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    from bot.config import (
        CHROMA_PERSIST_DIR,
        CHUNK_OVERLAP,
        CHUNK_SIZE,
        OLLAMA_BASE_URL,
        OLLAMA_MODEL,
        TELEGRAM_BOT_TOKEN,
    )
    from bot.handlers import (
        clear_handler,
        document_handler,
        help_handler,
        message_handler,
        start_handler,
        stats_handler,
    )
    from rag.llm_client import OllamaClient
    from rag.pipeline import RAGPipeline
    from rag.vector_store import VectorStore

    # Инициализация компонентов
    vector_store = VectorStore(CHROMA_PERSIST_DIR)
    llm_client = OllamaClient(OLLAMA_BASE_URL, OLLAMA_MODEL)
    pipeline = RAGPipeline(vector_store, llm_client, CHUNK_SIZE, CHUNK_OVERLAP)

    logger.info("Ollama: %s, модель: %s", OLLAMA_BASE_URL, OLLAMA_MODEL)

    # Сборка Telegram-приложения
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # Передаём pipeline через bot_data
    app.bot_data["pipeline"] = pipeline

    # Регистрация обработчиков
    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(CommandHandler("help", help_handler))
    app.add_handler(CommandHandler("stats", stats_handler))
    app.add_handler(CommandHandler("clear", clear_handler))
    app.add_handler(MessageHandler(filters.Document.ALL, document_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))

    # Запуск
    logger.info("Бот запущен. Нажмите Ctrl+C для остановки.")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
