"""端到端测试基建：uvicorn 跑在后台线程里，真实 TCP 上验证 router。"""

from __future__ import annotations

import socket
import threading
import time
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
import uvicorn

from tests.mock_ucm_engine import EngineSettings, build_app

MODEL = "mock-model"


class ThreadServer:
    """uvicorn.Server 跑在后台线程（各自独立事件循环），供测试跨进程语义验证。"""

    def __init__(self, app, graceful_shutdown: float | None = None) -> None:
        self.host = "127.0.0.1"
        self.port = free_port()
        self.server = uvicorn.Server(
            uvicorn.Config(app, host=self.host, port=self.port,
                           log_level="warning", access_log=False,
                           timeout_graceful_shutdown=graceful_shutdown)
        )
        self._thread: threading.Thread | None = None

    def start(self) -> "ThreadServer":
        self._thread = threading.Thread(target=self.server.run, daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if self._reachable():
                return self
            time.sleep(0.05)
        raise RuntimeError(f"server on port {self.port} did not start")

    def stop(self) -> None:
        if self._thread is None:
            return
        self.server.should_exit = True
        self._thread.join(timeout=10)
        self._thread = None

    def _reachable(self) -> bool:
        with socket.socket() as sock:
            sock.settimeout(0.2)
            return sock.connect_ex((self.host, self.port)) == 0

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def make_engine(store: Path, role: str, model: str = MODEL, **overrides) -> ThreadServer:
    settings = replace(EngineSettings(role=role, model=model, store=store), **overrides)
    return ThreadServer(build_app(settings))


def make_router(
    groups: list[tuple[str, list[str], list[str]]],
    decode_dispatch: str = "sequential",
    health_interval: float = 3600.0,
    max_inflight_per_node: int = 0,
    session_affinity: bool = False,
) -> ThreadServer:
    from ucm_router.config import (
        EngineConfig,
        HealthConfig,
        RouterConfig,
        RoutingConfig,
        ServerConfig,
        ServingGroupConfig,
    )
    from ucm_router.main import create_app

    config = RouterConfig(
        server=ServerConfig(),
        decode_dispatch=decode_dispatch,
        engine=EngineConfig(connect_timeout_seconds=2.0,
                            prefill_timeout_seconds=20.0,
                            decode_timeout_seconds=30.0),
        health=HealthConfig(interval_seconds=health_interval, timeout_seconds=1.0),
        routing=RoutingConfig(
            serving_groups=[
                ServingGroupConfig(model=model, prefill=prefill, decode=decode)
                for model, prefill, decode in groups
            ],
            max_inflight_per_node=max_inflight_per_node,
            session_affinity=session_affinity,
        ),
    )
    return ThreadServer(create_app(config))


@pytest.fixture
def servers():
    """跟踪测试里创建的所有服务，结束时统一关停。"""
    created: list[ThreadServer] = []

    def track(server: ThreadServer) -> ThreadServer:
        created.append(server)
        return server.start()

    yield track
    for server in created:
        server.stop()


@pytest.fixture
def store_dir(tmp_path: Path) -> Path:
    store = tmp_path / "ucm-shared"
    store.mkdir()
    return store


@pytest.fixture
async def http():
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        yield client


def chat_payload(prompt: str, max_tokens: int = 8, stream: bool = True, model: str = MODEL) -> dict:
    return {
        "model": model,
        "stream": stream,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
