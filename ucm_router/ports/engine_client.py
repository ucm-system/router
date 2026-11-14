"""Port: EngineClient —— 向 P/D 提交 OpenAI 兼容请求。

v4 分工的直接体现：
- prefill 请求只是普通 OpenAI 请求的改写（stream=False、max_tokens=1），
  响应 body（含 P 的 usage，即 prefill 侧 cached_tokens）返回给编排层用于
  API 契约合并——只做数据管道，不参与任何路由决策；
- decode 请求原样透传，不注入 kv_transfer_params、不构造预取参数——
  KV 如何就绪是 UCM Connector 与引擎的事，Router 完全不感知。
- D 的响应做字节级透传（不解析 SSE chunk），Router 不理解语义。
"""

from __future__ import annotations

from typing import AsyncIterator, Protocol

import httpx

from ucm_router.config import EngineConfig
from ucm_router.domain.routing import ServingNode
from ucm_router.errors import DecodeRequestFailed, PrefillRequestFailed

# 连接级失败：换一个节点重试是有意义的（见 _BackendRequestFailed.retryable）
_RETRYABLE_CONNECT_ERRORS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadError,
    httpx.PoolTimeout,
    httpx.RemoteProtocolError,
)


class DecodeStream:
    """已建立的 D 侧响应流。

    内部只持有一个 body 迭代器：read_first_chunk 预读的首个 chunk 之后由
    iter_bytes 顺序回放（httpx 的 body 流不支持多个并行迭代器）。
    """

    def __init__(self, response: httpx.Response) -> None:
        self._response = response
        self._iterator = response.aiter_bytes()
        self.content_type = response.headers.get("content-type", "application/octet-stream")

    async def read_first_chunk(self) -> bytes:
        """预读首个 body chunk（提交响应头之前），供 orchestrator preopen。"""
        try:
            return await anext(self._iterator)
        except StopAsyncIteration:
            return b""

    async def iter_bytes(self) -> AsyncIterator[bytes]:
        async for chunk in self._iterator:
            yield chunk

    async def read_all(self) -> bytes:
        chunks: list[bytes] = []
        async for chunk in self._iterator:
            chunks.append(chunk)
        return b"".join(chunks)

    async def aclose(self) -> None:
        await self._response.aclose()


def rewrite_for_prefill(payload: dict) -> dict:
    """prefill 请求改写：阻塞拿响应、只产一个 token（vllm-ascend toy proxy 同语义）。

    - stream=False、max_tokens=1、min_tokens=1（确保 P 真的产出 token）；
    - OpenAI 新口径 max_completion_tokens 同步置 1，否则引擎可能优先用它；
    - stream_options 只在 stream=True 下合法，随 stream=False 一并移除。
    """
    body = dict(payload)
    body["stream"] = False
    body["max_tokens"] = 1
    body["min_tokens"] = 1
    if "max_completion_tokens" in body:
        body["max_completion_tokens"] = 1
    body.pop("stream_options", None)
    return body


def _looks_unreachable(response: httpx.Response) -> bool:
    """空 body 的 5xx 视为节点不可达。

    推理引擎自身不会返回没有 body 的 5xx；这是网关/透明代理在目标端口
    已死时的典型应答（本机若有代理客户端拦截 loopback 也会如此）。
    按不可达处理可触发被动健康摘除；重试策略不受影响（仍可 failover）。
    """
    return response.status_code >= 500 and not response.content.strip()


class EngineClient(Protocol):
    async def submit_prefill(
        self, node: ServingNode, path: str, payload: dict, request_id: str
    ) -> dict: ...

    async def open_decode(
        self, node: ServingNode, path: str, payload: dict, request_id: str
    ) -> DecodeStream: ...

    async def get(self, node: ServingNode, path: str, timeout_seconds: float = 5.0) -> tuple[int, bytes]: ...

    async def post(self, node: ServingNode, path: str, timeout_seconds: float = 5.0) -> tuple[int, bytes]: ...

    async def aclose(self) -> None: ...


class VllmOpenAIClient:
    """vLLM（及任何 OpenAI 兼容引擎）的 HTTP 实现。"""

    def __init__(self, engine: EngineConfig) -> None:
        self._engine = engine
        self._client = httpx.AsyncClient()

    async def submit_prefill(
        self, node: ServingNode, path: str, payload: dict, request_id: str
    ) -> dict:
        """阻塞直到 P 的 HTTP 响应返回——P 响应即 prefill 完成的信号点。

        返回 P 的响应 body（小响应，max_tokens=1），编排层从中取 prefill 侧
        usage 供 cached_tokens 双口径合并；Router 不据此做任何调度。

        注意：P 返回成功不等于 KV 已被 D 可读，那个时间差由 D 侧 UCM
        Connector 的就绪语义兜底，Router 不感知。
        """
        timeout = self._timeout(self._engine.prefill_timeout_seconds)
        try:
            response = await self._client.post(
                f"{node.base_url}{path}",
                json=rewrite_for_prefill(payload),
                headers={"X-Request-Id": request_id},
                timeout=timeout,
            )
        except httpx.HTTPError as exc:
            raise PrefillRequestFailed(
                f"prefill node {node.base_url} unreachable: {exc!r}",
                node=node,
                retryable=isinstance(exc, _RETRYABLE_CONNECT_ERRORS),
                unreachable=isinstance(exc, _RETRYABLE_CONNECT_ERRORS),
            ) from exc
        if response.status_code >= 400:
            raise PrefillRequestFailed(
                f"prefill node {node.base_url} returned "
                f"{response.status_code}: {response.text[:200]}",
                node=node,
                retryable=response.status_code >= 500,
                unreachable=_looks_unreachable(response),
            )
        try:
            return response.json()
        except ValueError:
            return {}

    async def open_decode(
        self, node: ServingNode, path: str, payload: dict, request_id: str
    ) -> DecodeStream:
        """向 D 提交原始请求并建立流。decode 请求不带任何缓存/传输参数。"""
        timeout = self._timeout(self._engine.decode_timeout_seconds)
        request = self._client.build_request(
            "POST",
            f"{node.base_url}{path}",
            json=payload,
            headers={"X-Request-Id": request_id},
            timeout=timeout,
        )
        try:
            response = await self._client.send(request, stream=True)
        except httpx.HTTPError as exc:
            raise DecodeRequestFailed(
                f"decode node {node.base_url} unreachable: {exc!r}",
                node=node,
                retryable=isinstance(exc, _RETRYABLE_CONNECT_ERRORS),
                unreachable=isinstance(exc, _RETRYABLE_CONNECT_ERRORS),
            ) from exc
        if response.status_code >= 400:
            await response.aread()
            await response.aclose()
            raise DecodeRequestFailed(
                f"decode node {node.base_url} returned "
                f"{response.status_code}: {response.text[:200]}",
                node=node,
                retryable=response.status_code >= 500,
                unreachable=_looks_unreachable(response),
            )
        return DecodeStream(response)

    async def get(self, node: ServingNode, path: str, timeout_seconds: float = 5.0) -> tuple[int, bytes]:
        """GET 一个小端点（如 /v1/models）。status=0 表示连接级失败。"""
        try:
            response = await self._client.get(
                f"{node.base_url}{path}", timeout=self._timeout(timeout_seconds))
        except httpx.HTTPError:
            return 0, b""
        return response.status_code, response.content

    async def post(self, node: ServingNode, path: str, timeout_seconds: float = 5.0) -> tuple[int, bytes]:
        """POST 一个无 body 的管理端点（如 /reset_prefix_cache）。status=0 表示连接级失败。"""
        try:
            response = await self._client.post(
                f"{node.base_url}{path}", timeout=self._timeout(timeout_seconds))
        except httpx.HTTPError:
            return 0, b""
        return response.status_code, response.content

    def _timeout(self, read_seconds: float) -> httpx.Timeout:
        return httpx.Timeout(
            connect=self._engine.connect_timeout_seconds,
            read=read_seconds,
            write=30.0,
            pool=self._engine.connect_timeout_seconds,
        )

    async def aclose(self) -> None:
        await self._client.aclose()


__all__ = [
    "DecodeStream",
    "EngineClient",
    "VllmOpenAIClient",
    "rewrite_for_prefill",
]
