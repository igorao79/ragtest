"""Менеджер сессий — каждая сессия = отдельная база знаний + история.

Сессии автоматически удаляются через INACTIVE_TTL (по умолчанию 7 дней).
Метаданные хранятся в JSON-файле рядом с ChromaDB.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_SESSIONS_FILE = "sessions.json"
_INACTIVE_TTL = 7 * 24 * 3600  # 7 дней в секундах


@dataclass
class Session:
    """Одна сессия пользователя."""
    name: str
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    def touch(self) -> None:
        """Обновить время последней активности."""
        self.last_active = time.time()

    def is_expired(self, ttl: int = _INACTIVE_TTL) -> bool:
        """Проверить, истекла ли сессия."""
        return time.time() - self.last_active > ttl


@dataclass
class UserSessions:
    """Все сессии одного пользователя."""
    active: str | None = None
    sessions: dict[str, Session] = field(default_factory=dict)


class SessionManager:
    """Управляет сессиями всех пользователей.

    Хранит метаданные в JSON-файле, коллекции ChromaDB
    именуются как user_{user_id}_{session_name}.
    """

    def __init__(self, persist_dir: str, inactive_ttl: int = _INACTIVE_TTL) -> None:
        self._persist_dir = Path(persist_dir)
        self._file = self._persist_dir / _SESSIONS_FILE
        self._ttl = inactive_ttl
        self._data: dict[int, UserSessions] = {}
        self._load()

    # === Public API ===

    def create(self, user_id: int, name: str) -> Session:
        """Создать сессию и переключиться на неё."""
        user = self._get_user(user_id)
        clean_name = self._sanitize(name)

        if clean_name in user.sessions:
            # Уже существует — просто переключаемся
            user.active = clean_name
            user.sessions[clean_name].touch()
            self._save()
            return user.sessions[clean_name]

        session = Session(name=clean_name)
        user.sessions[clean_name] = session
        user.active = clean_name
        self._save()
        logger.info("Сессия создана: user_%d/%s", user_id, clean_name)
        return session

    def switch(self, user_id: int, name: str) -> Session | None:
        """Переключиться на существующую сессию."""
        user = self._get_user(user_id)
        clean_name = self._sanitize(name)

        if clean_name not in user.sessions:
            return None

        user.active = clean_name
        user.sessions[clean_name].touch()
        self._save()
        return user.sessions[clean_name]

    def delete(self, user_id: int, name: str) -> bool:
        """Удалить сессию. Возвращает True если удалена."""
        user = self._get_user(user_id)
        clean_name = self._sanitize(name)

        if clean_name not in user.sessions:
            return False

        del user.sessions[clean_name]
        if user.active == clean_name:
            # Переключаемся на другую или None
            user.active = next(iter(user.sessions), None)
        self._save()
        logger.info("Сессия удалена: user_%d/%s", user_id, clean_name)
        return True

    def get_active(self, user_id: int) -> str | None:
        """Получить имя активной сессии (для collection_name в VectorStore)."""
        user = self._get_user(user_id)
        if user.active and user.active in user.sessions:
            user.sessions[user.active].touch()
            return user.active
        return None

    def get_active_display(self, user_id: int) -> str:
        """Получить имя активной сессии для отображения."""
        name = self.get_active(user_id)
        return name or "нет активной сессии"

    def list_sessions(self, user_id: int) -> list[dict]:
        """Список сессий пользователя."""
        user = self._get_user(user_id)
        self._cleanup_expired(user_id)
        result = []
        for name, s in sorted(user.sessions.items()):
            result.append({
                "name": name,
                "active": name == user.active,
                "created_at": s.created_at,
                "last_active": s.last_active,
            })
        return result

    def touch(self, user_id: int) -> None:
        """Обновить время активности текущей сессии."""
        user = self._get_user(user_id)
        if user.active and user.active in user.sessions:
            user.sessions[user.active].touch()

    def cleanup_all_expired(self) -> list[tuple[int, str]]:
        """Очистить все истекшие сессии. Возвращает список (user_id, name)."""
        removed = []
        for user_id in list(self._data.keys()):
            removed.extend(self._cleanup_expired(user_id))
        if removed:
            self._save()
        return removed

    def get_collection_name(self, user_id: int) -> str | None:
        """Получить имя коллекции ChromaDB для активной сессии."""
        return self.get_active(user_id)

    # === Private ===

    def _get_user(self, user_id: int) -> UserSessions:
        if user_id not in self._data:
            self._data[user_id] = UserSessions()
        return self._data[user_id]

    def _cleanup_expired(self, user_id: int) -> list[tuple[int, str]]:
        """Удалить истекшие сессии пользователя."""
        user = self._get_user(user_id)
        expired = [
            name for name, s in user.sessions.items()
            if s.is_expired(self._ttl)
        ]
        removed = []
        for name in expired:
            del user.sessions[name]
            removed.append((user_id, name))
            logger.info("Сессия истекла: user_%d/%s", user_id, name)
        if user.active and user.active not in user.sessions:
            user.active = next(iter(user.sessions), None)
        return removed

    @staticmethod
    def _sanitize(name: str) -> str:
        """Очистить имя сессии для ChromaDB."""
        import re
        clean = re.sub(r"[^\w\-]", "_", name.strip().lower())
        return clean[:50] or "default"

    def _load(self) -> None:
        """Загрузить метаданные из файла."""
        if not self._file.exists():
            return
        try:
            raw = json.loads(self._file.read_text(encoding="utf-8"))
            for uid_str, udata in raw.items():
                uid = int(uid_str)
                user = UserSessions(active=udata.get("active"))
                for sname, sdata in udata.get("sessions", {}).items():
                    user.sessions[sname] = Session(
                        name=sdata["name"],
                        created_at=sdata["created_at"],
                        last_active=sdata["last_active"],
                    )
                self._data[uid] = user
            logger.info("Загружено %d пользователей из sessions.json", len(self._data))
        except Exception as e:
            logger.error("Ошибка загрузки sessions.json: %s", e)

    def _save(self) -> None:
        """Сохранить метаданные в файл."""
        raw = {}
        for uid, user in self._data.items():
            raw[str(uid)] = {
                "active": user.active,
                "sessions": {
                    name: asdict(s) for name, s in user.sessions.items()
                },
            }
        try:
            self._file.parent.mkdir(parents=True, exist_ok=True)
            self._file.write_text(
                json.dumps(raw, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.error("Ошибка сохранения sessions.json: %s", e)
