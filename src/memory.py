"""
In-memory session store for conversation context.

Design decision: in-memory (not Postgres/SQLite).
Rationale: Assignment scope is a single-process demo. In-memory avoids
infra complexity, has zero latency, and is trivially testable. A real
production system would replace _store with a Redis or Postgres backend
behind the same interface — no other code changes needed.

TTL: sessions expire after SESSION_TTL_SECONDS of inactivity.
"""
from __future__ import annotations
import time
from typing import Optional

SESSION_TTL_SECONDS = 3600  # 1 hour


class Session:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.turns: list[dict] = []
        self.last_active = time.time()

    def add_turn(self, role: str, content: str) -> None:
        self.turns.append({"role": role, "content": content})
        self.last_active = time.time()

    def prior_user_turns(self) -> list[str]:
        return [t["content"] for t in self.turns if t["role"] == "user"]

    def context_window(self, max_turns: int = 6) -> list[dict]:
        """Return the last N turns for LLM context."""
        return self.turns[-max_turns:]

    def is_expired(self) -> bool:
        return (time.time() - self.last_active) > SESSION_TTL_SECONDS


class MemoryStore:
    def __init__(self):
        self._store: dict[str, Session] = {}

    def get_or_create(self, session_id: str) -> Session:
        self._evict_expired()
        if session_id not in self._store:
            self._store[session_id] = Session(session_id)
        return self._store[session_id]

    def get(self, session_id: str) -> Optional[Session]:
        return self._store.get(session_id)

    def _evict_expired(self) -> None:
        expired = [sid for sid, s in self._store.items() if s.is_expired()]
        for sid in expired:
            del self._store[sid]


# Module-level singleton
memory = MemoryStore()