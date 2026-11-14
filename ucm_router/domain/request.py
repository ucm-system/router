"""Request 域：请求生命周期状态（SessionRepository Port 的承载对象）。

生命周期阶段：received → routed → prefilling → decoding → completed / failed。
Router 管理的是请求从进到出的生命周期；请求在引擎内部的执行时机
（D 何时真正开始 decode）不在其中——那是引擎与 UCM Connector 的事。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class RequestPhase(str, Enum):
    RECEIVED = "received"
    ROUTED = "routed"
    PREFILLING = "prefilling"
    DECODING = "decoding"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class RequestSession:
    """一次 PD 请求的上下文。request_id 贯穿 P、D 两段调用（X-Request-Id）。"""

    request_id: str
    model: str
    phase: RequestPhase = RequestPhase.RECEIVED
    created_at: float = field(default_factory=time.monotonic)
    affinity_key: str | None = None   # 会话亲和 key（session_affinity 开启时派生）

    prefill_node: str | None = None
    decode_node: str | None = None
    prefill_finished_at: float | None = None
    first_byte_at: float | None = None
    released: bool = False   # 幂等释放标记（流式响应的清理可能晚于 handle 返回）

    def record_routed(self, prefill_url: str, decode_url: str) -> None:
        self.phase = RequestPhase.ROUTED
        self.prefill_node = prefill_url
        self.decode_node = decode_url

    def record_prefill_finished(self) -> None:
        self.phase = RequestPhase.DECODING
        self.prefill_finished_at = time.monotonic()

    def record_first_byte(self) -> None:
        self.first_byte_at = time.monotonic()

    @property
    def ttft_seconds(self) -> float | None:
        """Router 视角的 TTFT：请求进来到 D 返回第一个字节（含 prefill 时间）。"""
        if self.first_byte_at is None:
            return None
        return self.first_byte_at - self.created_at

    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.created_at
