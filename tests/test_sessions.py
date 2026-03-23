"""Тесты для менеджера сессий."""

import shutil
import tempfile
import time

import pytest

from rag.sessions import Session, SessionManager


@pytest.fixture
def manager():
    tmpdir = tempfile.mkdtemp()
    sm = SessionManager(tmpdir, inactive_ttl=5)
    yield sm
    shutil.rmtree(tmpdir, ignore_errors=True)


class TestSessionManager:
    def test_create_session(self, manager: SessionManager):
        s = manager.create(1, "работа")
        assert s.name == "работа"
        assert manager.get_active(1) == "работа"

    def test_create_duplicate_switches(self, manager: SessionManager):
        manager.create(1, "работа")
        manager.create(1, "учёба")
        assert manager.get_active(1) == "учёба"
        # Create existing — just switches
        manager.create(1, "работа")
        assert manager.get_active(1) == "работа"

    def test_switch_session(self, manager: SessionManager):
        manager.create(1, "a")
        manager.create(1, "b")
        assert manager.get_active(1) == "b"
        result = manager.switch(1, "a")
        assert result is not None
        assert manager.get_active(1) == "a"

    def test_switch_nonexistent(self, manager: SessionManager):
        result = manager.switch(1, "nope")
        assert result is None

    def test_delete_session(self, manager: SessionManager):
        manager.create(1, "temp")
        assert manager.delete(1, "temp")
        assert manager.get_active(1) is None

    def test_delete_switches_to_another(self, manager: SessionManager):
        manager.create(1, "a")
        manager.create(1, "b")
        manager.switch(1, "b")
        manager.delete(1, "b")
        # Should switch to remaining session
        assert manager.get_active(1) == "a"

    def test_delete_nonexistent(self, manager: SessionManager):
        assert not manager.delete(1, "nope")

    def test_list_sessions(self, manager: SessionManager):
        manager.create(1, "alpha")
        manager.create(1, "beta")
        sessions = manager.list_sessions(1)
        names = [s["name"] for s in sessions]
        assert "alpha" in names
        assert "beta" in names
        # beta is active (last created)
        active = [s for s in sessions if s["active"]]
        assert len(active) == 1
        assert active[0]["name"] == "beta"

    def test_list_empty(self, manager: SessionManager):
        assert manager.list_sessions(999) == []

    def test_get_active_display_no_session(self, manager: SessionManager):
        display = manager.get_active_display(999)
        assert "нет" in display

    def test_separate_users(self, manager: SessionManager):
        manager.create(1, "user1_session")
        manager.create(2, "user2_session")
        assert manager.get_active(1) == "user1_session"
        assert manager.get_active(2) == "user2_session"

    def test_sanitize_name(self, manager: SessionManager):
        s = manager.create(1, "Моя Работа!!!")
        assert s.name == "моя_работа___"

    def test_persistence(self, manager: SessionManager):
        manager.create(1, "persist_test")
        # Reload from same dir
        manager2 = SessionManager(
            str(manager._persist_dir), inactive_ttl=5
        )
        assert manager2.get_active(1) == "persist_test"

    def test_expired_cleanup(self):
        tmpdir = tempfile.mkdtemp()
        try:
            sm = SessionManager(tmpdir, inactive_ttl=1)
            sm.create(1, "old")
            time.sleep(1.5)
            removed = sm.cleanup_all_expired()
            assert len(removed) == 1
            assert removed[0] == (1, "old")
            assert sm.get_active(1) is None
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_touch_updates_activity(self, manager: SessionManager):
        manager.create(1, "active_one")
        s = manager._get_user(1).sessions["active_one"]
        old_time = s.last_active
        time.sleep(0.1)
        manager.touch(1)
        assert s.last_active > old_time

    def test_get_collection_name(self, manager: SessionManager):
        assert manager.get_collection_name(1) is None
        manager.create(1, "test")
        assert manager.get_collection_name(1) == "test"
