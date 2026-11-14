"""Port: SessionRepository —— 请求上下文。

v4 保留的三个 Port 之一。第一版实现为进程内字典；
未来跨进程共享（如持久化请求状态）时新增实现即可。
"""

from __future__ import annotations

from typing import Protocol

from ucm_router.domain.request import RequestSession


class SessionRepository(Protocol):
    def add(self, session: RequestSession) -> None: ...

    def remove(self, request_id: str) -> RequestSession | None: ...

    def active_count(self) -> int: ...


class InMemorySessionRepository:
    def __init__(self) -> None:
        self._sessions: dict[str, RequestSession] = {}

    def add(self, session: RequestSession) -> None:
        self._sessions[session.request_id] = session

    def remove(self, request_id: str) -> RequestSession | None:
        return self._sessions.pop(request_id, None)

    def active_count(self) -> int:
        return len(self._sessions)
