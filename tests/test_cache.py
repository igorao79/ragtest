"""Тесты для ResponseCache."""

import time

from rag.cache import ResponseCache


def test_put_and_get():
    cache = ResponseCache(max_size=10, ttl=60)
    cache.put(1, "hello?", "world!")
    assert cache.get(1, "hello?") == "world!"


def test_case_insensitive():
    cache = ResponseCache(max_size=10, ttl=60)
    cache.put(1, "Hello?", "world!")
    assert cache.get(1, "hello?") == "world!"
    assert cache.get(1, "HELLO?") == "world!"


def test_miss():
    cache = ResponseCache(max_size=10, ttl=60)
    assert cache.get(1, "unknown") is None


def test_ttl_expiry():
    cache = ResponseCache(max_size=10, ttl=1)
    cache.put(1, "q", "a")
    assert cache.get(1, "q") == "a"
    time.sleep(1.1)
    assert cache.get(1, "q") is None


def test_max_size():
    cache = ResponseCache(max_size=3, ttl=60)
    cache.put(1, "q1", "a1")
    cache.put(1, "q2", "a2")
    cache.put(1, "q3", "a3")
    cache.put(1, "q4", "a4")
    # q1 должен быть вытеснен
    assert cache.get(1, "q1") is None
    assert cache.get(1, "q4") == "a4"


def test_invalidate_user():
    cache = ResponseCache(max_size=10, ttl=60)
    cache.put(1, "q1", "a1")
    cache.put(2, "q2", "a2")
    cache.invalidate_user(1)
    assert cache.get(1, "q1") is None
    assert cache.get(2, "q2") == "a2"


def test_clear():
    cache = ResponseCache(max_size=10, ttl=60)
    cache.put(1, "q1", "a1")
    cache.put(2, "q2", "a2")
    cache.clear()
    assert cache.get(1, "q1") is None
    assert cache.get(2, "q2") is None
