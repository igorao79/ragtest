"""Загрузка и парсинг документов (PDF, DOCX, TXT, MD)."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DocumentLoadError(Exception):
    """Ошибка при загрузке документа."""


class DocumentLoader:
    """Загружает текст из файлов различных форматов."""

    def load(self, file_path: str) -> str:
        """Загрузить текст из файла.

        Args:
            file_path: Путь к файлу.

        Returns:
            Извлечённый текст.

        Raises:
            DocumentLoadError: Если файл не удалось прочитать.
        """
        path = Path(file_path)
        if not path.exists():
            raise DocumentLoadError(f"Файл не найден: {file_path}")

        ext = path.suffix.lower()
        try:
            if ext == ".pdf":
                return self._load_pdf(path)
            elif ext == ".docx":
                return self._load_docx(path)
            elif ext in (".txt", ".md"):
                return self._load_text(path)
            else:
                raise DocumentLoadError(f"Неподдерживаемый формат файла: {ext}")
        except DocumentLoadError:
            raise
        except Exception as e:
            raise DocumentLoadError(f"Ошибка при чтении файла {path.name}: {e}") from e

    def _load_pdf(self, path: Path) -> str:
        """Извлечь текст из PDF."""
        from PyPDF2 import PdfReader

        reader = PdfReader(str(path))
        pages: list[str] = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        if not pages:
            raise DocumentLoadError("PDF не содержит извлекаемого текста")
        return "\n\n".join(pages)

    def _load_docx(self, path: Path) -> str:
        """Извлечь текст из DOCX."""
        from docx import Document

        doc = Document(str(path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        if not paragraphs:
            raise DocumentLoadError("DOCX не содержит текста")
        return "\n\n".join(paragraphs)

    def _load_text(self, path: Path) -> str:
        """Прочитать текстовый файл с автоопределением кодировки."""
        import chardet

        raw = path.read_bytes()
        if not raw:
            raise DocumentLoadError("Файл пуст")
        detected = chardet.detect(raw)
        encoding = detected.get("encoding", "utf-8") or "utf-8"
        return raw.decode(encoding)
