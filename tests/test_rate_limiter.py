"""Тесты для RateLimiter."""

import time

from rag.rate_limiter import RateLimiter


def test_allows_within_limit():
    limiter = RateLimiter(max_requests=3, window_seconds=60)
    assert limiter.is_allowed(1) is True
    assert limiter.is_allowed(1) is True
    assert limiter.is_allowed(1) is True


def test_blocks_over_limit():
    limiter = RateLimiter(max_requests=2, window_seconds=60)
    assert limiter.is_allowed(1) is True
    assert limiter.is_allowed(1) is True
    assert limiter.is_allowed(1) is False


def test_different_users():
    limiter = RateLimiter(max_requests=1, window_seconds=60)
    assert limiter.is_allowed(1) is True
    assert limiter.is_allowed(2) is True
    assert limiter.is_allowed(1) is False
    assert limiter.is_allowed(2) is False


def test_window_expiry():
    limiter = RateLimiter(max_requests=1, window_seconds=1)
    assert limiter.is_allowed(1) is True
    assert limiter.is_allowed(1) is False
    time.sleep(1.1)
    assert limiter.is_allowed(1) is True


def test_remaining():
    limiter = RateLimiter(max_requests=3, window_seconds=60)
    assert limiter.remaining(1) == 3
    limiter.is_allowed(1)
    assert limiter.remaining(1) == 2
    limiter.is_allowed(1)
    assert limiter.remaining(1) == 1


def test_retry_after():
    limiter = RateLimiter(max_requests=1, window_seconds=10)
    limiter.is_allowed(1)
    retry = limiter.retry_after(1)
    assert 0 < retry <= 10
