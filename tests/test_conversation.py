"""Тесты для модуля диалоговой памяти."""

import time

from rag.conversation import ConversationMemory


class TestConversationMemory:
    def test_add_and_get_history(self):
        mem = ConversationMemory(max_messages=10, ttl=300)
        mem.add_user_message(1, "Привет")
        mem.add_assistant_message(1, "Здравствуйте!")

        history = mem.get_history(1)
        assert len(history) == 2
        assert history[0].role == "user"
        assert history[0].text == "Привет"
        assert history[1].role == "assistant"
        assert history[1].text == "Здравствуйте!"

    def test_max_messages_trim(self):
        mem = ConversationMemory(max_messages=3, ttl=300)
        for i in range(5):
            mem.add_user_message(1, f"msg {i}")

        history = mem.get_history(1)
        assert len(history) == 3
        assert history[0].text == "msg 2"

    def test_ttl_expiration(self):
        mem = ConversationMemory(max_messages=10, ttl=1)
        mem.add_user_message(1, "old message")
        time.sleep(1.1)
        history = mem.get_history(1)
        assert len(history) == 0

    def test_context_string(self):
        mem = ConversationMemory(max_messages=10, ttl=300)
        mem.add_user_message(1, "Что такое RAG?")
        mem.add_assistant_message(1, "RAG — это Retrieval-Augmented Generation.")

        ctx = mem.get_context_string(1)
        assert "Пользователь: Что такое RAG?" in ctx
        assert "Ассистент: RAG" in ctx

    def test_context_string_empty(self):
        mem = ConversationMemory()
        ctx = mem.get_context_string(999)
        assert ctx == ""

    def test_clear(self):
        mem = ConversationMemory()
        mem.add_user_message(1, "test")
        mem.clear(1)
        assert len(mem.get_history(1)) == 0

    def test_separate_users(self):
        mem = ConversationMemory()
        mem.add_user_message(1, "user1 msg")
        mem.add_user_message(2, "user2 msg")

        assert len(mem.get_history(1)) == 1
        assert len(mem.get_history(2)) == 1
        assert mem.get_history(1)[0].text == "user1 msg"

    def test_context_string_max_chars(self):
        mem = ConversationMemory()
        mem.add_user_message(1, "A" * 1000)
        mem.add_user_message(1, "B" * 1000)

        ctx = mem.get_context_string(1, max_chars=500)
        # Should only include what fits
        assert len(ctx) <= 1020  # label + text
