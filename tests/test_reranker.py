"""Тесты для модуля реранкинга."""

import pytest

from rag.reranker import LLMReranker


class TestParseScores:
    def test_normal_scores(self):
        scores = LLMReranker._parse_scores("8, 5, 9, 3", 4)
        assert scores == [8.0, 5.0, 9.0, 3.0]

    def test_scores_with_text(self):
        scores = LLMReranker._parse_scores("Оценки: 7, 4, 8, 2", 4)
        assert scores == [7.0, 4.0, 8.0, 2.0]

    def test_scores_cap_at_10(self):
        scores = LLMReranker._parse_scores("15, 8, 12", 3)
        assert scores == [10.0, 8.0, 10.0]

    def test_missing_scores_filled(self):
        scores = LLMReranker._parse_scores("8, 5", 4)
        assert len(scores) == 4
        assert scores[0] == 8.0
        assert scores[1] == 5.0
        assert scores[2] == 5.0  # default fill
        assert scores[3] == 5.0

    def test_empty_response(self):
        scores = LLMReranker._parse_scores("", 3)
        assert scores == [5.0, 5.0, 5.0]

    def test_float_scores(self):
        scores = LLMReranker._parse_scores("7.5, 8.2, 3.0", 3)
        assert scores == [7.5, 8.2, 3.0]

    def test_extra_scores_trimmed(self):
        scores = LLMReranker._parse_scores("1, 2, 3, 4, 5, 6", 3)
        assert len(scores) == 3
        assert scores == [1.0, 2.0, 3.0]
