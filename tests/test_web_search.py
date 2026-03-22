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


def test_clean_query_removes_stop_words():
    result = clean_search_query("что еще интересного можешь сказать о Николае втором")
    assert "что" not in result.lower().split()
    assert "интересного" not in result.lower().split()
    assert "николае" in result.lower()


def test_clean_query_preserves_names():
    result = clean_search_query("расскажи про Николая второго")
    assert "Николая" in result
    assert "второго" in result


def test_clean_query_preserves_terms():
    result = clean_search_query("найди информацию о машинном обучении")
    assert "машинном" in result
    assert "обучении" in result


def test_clean_query_all_stop_words():
    # Если всё стоп-слова — возвращаем всё
    result = clean_search_query("что это как")
    assert len(result) > 0


def test_clean_query_web_search_junk():
    result = clean_search_query("в интернете найди")
    # Должно вернуть хоть что-то
    assert len(result) > 0


def test_clean_query_keeps_numbers():
    result = clean_search_query("кто правил после 1917 года")
    assert "1917" in result
    assert "правил" in result
