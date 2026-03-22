"""Тесты для WebSearchClient."""

from rag.web_search import SearchResult, WebSearchClient, clean_search_query


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


def test_clean_query_removes_bot_commands():
    result = clean_search_query("расскажи что еще интересного о Николае втором")
    assert "расскажи" not in result.lower().split()
    assert "интересного" not in result.lower().split()
    assert "николае" in result.lower()


def test_clean_query_preserves_question_words():
    """Вопросительные слова важны для поиска — не удалять."""
    result = clean_search_query("кто правил после николая второго")
    assert "кто" in result.lower()
    assert "правил" in result.lower()
    assert "николая" in result.lower()


def test_clean_query_preserves_terms():
    result = clean_search_query("найди информацию о машинном обучении")
    assert "машинном" in result
    assert "обучении" in result


def test_clean_query_all_commands():
    # Если всё — команды боту, возвращаем как есть
    result = clean_search_query("расскажи мне пожалуйста")
    assert len(result) > 0


def test_clean_query_web_junk():
    result = clean_search_query("в интернете найди")
    assert len(result) > 0


def test_clean_query_keeps_numbers():
    result = clean_search_query("кто правил после 1917 года")
    assert "1917" in result
    assert "кто" in result
    assert "правил" in result
