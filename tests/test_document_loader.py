"""Тесты для DocumentLoader."""

import tempfile
from pathlib import Path

import pytest

from rag.document_loader import DocumentLoadError, DocumentLoader


@pytest.fixture
def loader():
    return DocumentLoader()


def test_load_txt(loader):
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w", encoding="utf-8") as f:
        f.write("Hello, this is a test document with enough text.")
        f.flush()
        text = loader.load(f.name)
    assert "Hello" in text
    Path(f.name).unlink()


def test_load_md(loader):
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w", encoding="utf-8") as f:
        f.write("# Title\n\nSome markdown content for testing.")
        f.flush()
        text = loader.load(f.name)
    assert "Title" in text
    Path(f.name).unlink()


def test_load_csv(loader):
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", encoding="utf-8") as f:
        f.write("name,age,city\nAlice,30,Moscow\nBob,25,SPB\n")
        f.flush()
        text = loader.load(f.name)
    assert "Alice" in text
    assert "name: Alice" in text
    Path(f.name).unlink()


def test_load_xlsx(loader):
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.append(["Name", "Value"])
    ws.append(["Test", "123"])
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
        wb.save(f.name)
        f.flush()
        text = loader.load(f.name)
    assert "Test" in text
    Path(f.name).unlink()


def test_load_nonexistent(loader):
    with pytest.raises(DocumentLoadError, match="не найден"):
        loader.load("/nonexistent/file.txt")


def test_load_unsupported(loader):
    with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
        f.write(b"data")
        f.flush()
        with pytest.raises(DocumentLoadError, match="Неподдерживаемый"):
            loader.load(f.name)
    Path(f.name).unlink()


def test_load_empty_txt(loader):
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.flush()
        with pytest.raises(DocumentLoadError, match="пуст"):
            loader.load(f.name)
    Path(f.name).unlink()


def test_load_url(loader):
    # Тест load_url — только проверяем что ошибка нормально поднимается
    with pytest.raises(DocumentLoadError):
        loader.load_url("http://nonexistent.invalid.domain.test")
