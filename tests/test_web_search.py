"""Тесты для WebSearchClient."""

from rag.web_search import SearchResult, WebSearchClient


def test_format_results():
    results = [
        SearchResult(title="Title 1", url="https://example.com/1", snippet="Snippet 1"),
        SearchResult(title="Title 2", url="https://example.com/2", snippet="Snippet 2"),
    ]
    text = WebSearchClient.format_results(results)
    assert "[1] Title 1" in text
    assert "[2] Title 2" in text
    assert "Snippet 1" in text


def test_format_results_empty():
    assert WebSearchClient.format_results([]) == ""


def test_format_results_short():
    results = [
        SearchResult(title="Title 1", url="https://example.com/1", snippet="Snippet 1"),
    ]
    text = WebSearchClient.format_results_short(results)
    assert "1." in text
    assert "Title 1" in text
    assert "https://example.com/1" in text


def test_format_results_short_empty():
    text = WebSearchClient.format_results_short([])
    assert "Ничего не найдено" in text
