"""Автоматический реранкинг результатов через LLM."""

import logging

from rag.llm_client import OllamaClient

logger = logging.getLogger(__name__)

_RERANK_SYSTEM = (
    "Ты — помощник для оценки релевантности текстовых фрагментов. "
    "Оцени, насколько каждый фрагмент релевантен вопросу. "
    "Ответь ТОЛЬКО числами через запятую — оценки от 0 до 10 для каждого фрагмента. "
    "Ничего больше не пиши."
)

_RERANK_PROMPT = (
    "Вопрос: {question}\n\n"
    "Фрагменты:\n{fragments}\n\n"
    "Оценки релевантности (0-10) через запятую:"
)


class LLMReranker:
    """Переоценка релевантности результатов поиска через LLM.

    После similarity search в ChromaDB прогоняет top-K результатов
    через LLM для более точной оценки релевантности.
    """

    def __init__(self, llm_client: OllamaClient) -> None:
        self.llm_client = llm_client

    async def rerank(
        self, question: str, results: list[dict], top_k: int = 4
    ) -> list[dict]:
        """Переранжировать результаты по релевантности через LLM.

        Args:
            question: Вопрос пользователя.
            results: Результаты из vector_store.query().
            top_k: Сколько лучших результатов вернуть.

        Returns:
            Переранжированный список результатов.
        """
        if len(results) <= 1:
            return results

        # Формируем фрагменты для оценки
        fragments_text = "\n\n".join(
            f"[{i + 1}] {r['text'][:300]}" for i, r in enumerate(results)
        )

        prompt = _RERANK_PROMPT.format(
            question=question, fragments=fragments_text
        )

        try:
            response = await self.llm_client.generate(prompt, system=_RERANK_SYSTEM)
            scores = self._parse_scores(response, len(results))

            # Привязываем оценки к результатам
            scored = list(zip(scores, results))
            scored.sort(key=lambda x: x[0], reverse=True)

            reranked = [r for _, r in scored[:top_k]]
            logger.info(
                "Реранкинг: оценки %s, порядок %s",
                scores,
                [scored.index((s, r)) for s, r in scored[:top_k]],
            )
            return reranked

        except Exception as e:
            logger.warning("Реранкинг не удался, возвращаем оригинал: %s", e)
            return results[:top_k]

    @staticmethod
    def _parse_scores(response: str, expected_count: int) -> list[float]:
        """Извлечь оценки из ответа LLM."""
        import re

        # Ищем все числа в ответе
        numbers = re.findall(r"\d+(?:\.\d+)?", response)
        scores = [min(float(n), 10.0) for n in numbers[:expected_count]]

        # Если не хватает оценок — заполняем средним
        while len(scores) < expected_count:
            scores.append(5.0)

        return scores
