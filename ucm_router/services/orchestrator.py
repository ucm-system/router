"""请求生命周期编排。

v4 分工的落点：Router 只管选路与生命周期——
  Router 决定请求去哪里，UCM 决定 KV 如何准备好，Engine 决定什么时候执行计算。

Decode 下发时序（decode_dispatch）：
- sequential（默认）：P 的 prefill 完成（HTTP 响应返回）后再发 D。
  与 UCM 官方 toy_proxy_server.py 同语义，不依赖 D 侧就绪等待能力。
- early：选完 P/D 后尽快发 D，不等 P 完成。D 侧 UCM Connector 负责
  等待 KV 就绪（待验证前提①③）；P 失败则中止 D 请求。

容错（fail-fast 之上的最小增强）：
- 同池换节点重试一次，仅限可重试错误（连接级失败 / 后端 5xx）。
  重试是安全的：UCM 集中式模式下 KV 块按哈希内容寻址，重复 prefill
  写出一模一样的块文件，天然幂等。
- 连接级失败触发被动健康摘除（不等探测周期）。
- 在途到达上限返回 429，不排队（背压由客户端退避承担）。

cached_tokens 双口径：P 的 usage（prefill 侧复用，计费口径）覆盖响应中的
cached_tokens；D 的原值保留在 pd.decode / 响应头。Router 只做数据管道，
不据此做任何调度决策——缓存感知调度属于 UCM 侧能力（v4 边界）。
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import time
import uuid
from dataclasses import replace

import httpx
from fastapi import Response
from fastapi.responses import StreamingResponse

from ucm_router.config import (
    DECODE_DISPATCH_EARLY,
    DECODE_DISPATCH_SEQUENTIAL,
    RouterConfig,
)
from ucm_router.domain.request import RequestPhase, RequestSession
from ucm_router.domain.routing import ROLE_DECODE, ROLE_PREFILL, RouteAssignment, ServingNode
from ucm_router.errors import (
    DecodeRequestFailed,
    NoServingGroup,
    PrefillRequestFailed,
    RouterError,
)
from ucm_router.ports.engine_client import DecodeStream, EngineClient
from ucm_router.ports.node_provider import NodeProvider
from ucm_router.ports.session_repository import SessionRepository
from ucm_router.metrics import RouterMetrics
from ucm_router.services.health import HealthChecker
from ucm_router.services.scheduler import Scheduler

logger = logging.getLogger("ucm_router.orchestrator")


class Orchestrator:
    def __init__(
        self,
        provider: NodeProvider,
        scheduler: Scheduler,
        engine: EngineClient,
        sessions: SessionRepository,
        health: HealthChecker,
        config: RouterConfig,
        metrics: RouterMetrics,
    ) -> None:
        self._provider = provider
        self._scheduler = scheduler
        self._engine = engine
        self._sessions = sessions
        self._health = health
        self._metrics = metrics
        self._dispatch = config.decode_dispatch
        self._affinity = config.routing.session_affinity
        if self._dispatch not in (DECODE_DISPATCH_SEQUENTIAL, DECODE_DISPATCH_EARLY):
            raise ValueError(f"unknown decode_dispatch: {self._dispatch!r}")

    # --- 入口 ---------------------------------------------------------------

    async def handle(self, path: str, payload: dict, headers=None) -> Response:
        request_id = uuid.uuid4().hex
        session = RequestSession(request_id=request_id, model=payload.get("model", ""))
        if self._affinity:
            session.affinity_key = self._session_key(payload, headers)
        self._sessions.add(session)
        try:
            return await self._dispatch_request(path, payload, session)
        except Exception as exc:
            if getattr(exc, "unreachable", False) and getattr(exc, "node", None) is not None:
                self._health.mark_unreachable(exc.node)
            session.phase = RequestPhase.FAILED
            self._release(session)
            self._metrics.requests.inc(
                {"model": session.model, "dispatch": self._dispatch, "outcome": "failed"})
            if isinstance(exc, RouterError):
                self._metrics.failures.inc({"code": exc.code})
            raise

    def _session_key(self, payload: dict, headers) -> str | None:
        """会话标识派生（vllm-ascend toy proxy 的 conversation_affinity_key 设计）。

        不哈希持续增长的消息历史：优先 user 字段 / X-Session-Id 头，否则用
        model + 前置 system/developer 消息 + 第一条 user 消息——多轮对话中
        这些保持稳定，因此 key 稳定。仅作确定性放置的输入，不感知 KV 状态。
        """
        user = payload.get("user")
        if not (isinstance(user, str) and user):
            header_sid = headers.get("x-session-id") if headers is not None else None
            user = header_sid if isinstance(header_sid, str) and header_sid else None

        if isinstance(user, str) and user:
            identity = [payload.get("model"), "session", user]
        else:
            messages = payload.get("messages") or []
            first_user_index = next(
                (i for i, m in enumerate(messages)
                 if isinstance(m, dict) and m.get("role") == "user"),
                None,
            )
            if first_user_index is None:
                return None
            instructions = [
                m for m in messages[:first_user_index]
                if isinstance(m, dict) and m.get("role") in ("system", "developer")
            ]
            identity = [payload.get("model"), instructions, messages[first_user_index]]

        encoded = json.dumps(
            identity, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    async def _dispatch_request(
        self, path: str, payload: dict, session: RequestSession
    ) -> Response:
        group = self._provider.find_group(payload["model"])
        if group is None:
            raise NoServingGroup(f"no serving group matches model {payload['model']!r}")

        assignment = self._scheduler.assign(group, session, session.affinity_key)
        self._acquire(session, assignment)
        session.record_routed(assignment.prefill_node.base_url, assignment.decode_node.base_url)
        logger.info(
            "req=%s model=%s p=%s d=%s dispatch=%s affinity=%s",
            session.request_id, group.model,
            assignment.prefill_node.base_url, assignment.decode_node.base_url,
            self._dispatch, "on" if session.affinity_key else "off",
        )

        dispatch_started = time.monotonic()
        if self._dispatch == DECODE_DISPATCH_EARLY:
            decode, assignment, prefill_body = await self._submit_early(
                assignment, path, payload, session, dispatch_started)
        else:
            decode, assignment, prefill_body = await self._submit_sequential(
                assignment, path, payload, session)
        session.phase = RequestPhase.DECODING

        if payload.get("stream"):
            # preopen（ascend toy proxy 同语义）：提交 200 响应头之前先读 D 的
            # 第一个 chunk——首字节前的失败仍能以真实状态码返回，而不是空 200
            # 加一条截断流。
            try:
                first_chunk = await decode.read_first_chunk()
            except Exception as exc:
                await decode.aclose()
                raise DecodeRequestFailed(
                    f"decode node {assignment.decode_node.base_url} failed "
                    f"before first byte: {exc!r}",
                    node=assignment.decode_node,
                ) from exc
            return StreamingResponse(
                self._proxy_stream(session, decode, first_chunk),
                media_type=decode.content_type,
                headers=self._response_headers(session, assignment, prefill_body),
            )
        return await self._proxy_json(session, assignment, decode, prefill_body)

    # --- 两阶段提交 ----------------------------------------------------------

    async def _submit_sequential(self, assignment, path, payload, session):
        prefill_body, assignment = await self._submit_prefill_with_failover(
            assignment, path, payload, session)
        session.record_prefill_finished()
        decode, assignment = await self._open_decode_with_failover(
            assignment, path, payload, session)
        return decode, assignment, prefill_body

    async def _submit_early(self, assignment, path, payload, session, dispatch_started):
        """D 尽早收到 Decode 请求，等待 KV 就绪发生在 D 侧 Connector。

        P 失败（含换节点重试后仍失败）则中止已建立的 D 请求。
        """
        session.phase = RequestPhase.PREFILLING
        prefill_task = asyncio.create_task(
            self._submit_prefill_with_failover(assignment, path, payload, session),
            name=f"prefill-{session.request_id}",
        )
        decode: DecodeStream | None = None
        try:
            decode, assignment = await self._open_decode_with_failover(
                assignment, path, payload, session)
            prefill_body, assignment = await prefill_task
        except BaseException:
            if not prefill_task.done():
                prefill_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, RouterError):
                    await prefill_task
            if decode is not None:
                await decode.aclose()
            raise
        self._metrics.dispatch_lead.observe(
            {"model": session.model}, time.monotonic() - dispatch_started)
        session.record_prefill_finished()
        return decode, assignment, prefill_body

    # --- failover 包装：可重试错误时同池换节点重试一次 -----------------------

    async def _submit_prefill_with_failover(self, assignment, path, payload, session):
        started = time.monotonic()
        result, assignment = await self._call_with_failover(
            session, assignment, ROLE_PREFILL,
            lambda node: self._engine.submit_prefill(node, path, payload, session.request_id),
        )
        self._metrics.prefill_duration.observe(
            {"model": session.model}, time.monotonic() - started)
        logger.info(
            "req=%s prefill ok on %s in %.3fs",
            session.request_id, session.prefill_node, time.monotonic() - started)
        return result, assignment

    async def _open_decode_with_failover(self, assignment, path, payload, session):
        return await self._call_with_failover(
            session, assignment, ROLE_DECODE,
            lambda node: self._engine.open_decode(node, path, payload, session.request_id),
        )

    async def _call_with_failover(self, session, assignment, role, attempt):
        node: ServingNode = (
            assignment.prefill_node if role == ROLE_PREFILL else assignment.decode_node)
        while True:
            try:
                return await attempt(node), assignment
            except (PrefillRequestFailed, DecodeRequestFailed) as exc:
                if not exc.retryable:
                    raise
                pool = assignment.group.pool_for(role)
                new_node = pool.pick(key=session.affinity_key, exclude=node.base_url)
                if new_node is None:
                    raise
                logger.warning(
                    "req=%s %s node %s failed (%s), failing over to %s",
                    session.request_id, role, node.base_url, exc.message, new_node.base_url)
                self._metrics.retries.inc({"role": role})
                assignment = self._swap_node(session, assignment, role, new_node)
                node = new_node

    # --- 在途计数与释放 ------------------------------------------------------

    def _acquire(self, session, assignment) -> None:
        assignment.group.prefill_pool.acquire(assignment.prefill_node.base_url)
        assignment.group.decode_pool.acquire(assignment.decode_node.base_url)
        self._metrics.selections.inc(
            {"role": ROLE_PREFILL, "instance": assignment.prefill_node.base_url})
        self._metrics.selections.inc(
            {"role": ROLE_DECODE, "instance": assignment.decode_node.base_url})
        self._sync_runtime()

    def _swap_node(self, session, assignment, role, new_node: ServingNode):
        pool = assignment.group.pool_for(role)
        old_url = session.prefill_node if role == ROLE_PREFILL else session.decode_node
        if old_url is not None:
            pool.release(old_url)
        pool.acquire(new_node.base_url)
        self._metrics.selections.inc({"role": role, "instance": new_node.base_url})
        if role == ROLE_PREFILL:
            session.prefill_node = new_node.base_url
            assignment = replace(assignment, prefill_node=new_node)
        else:
            session.decode_node = new_node.base_url
            assignment = replace(assignment, decode_node=new_node)
        self._sync_runtime()
        return assignment

    def _release(self, session: RequestSession) -> None:
        """幂等释放：在途计数 -1、会话移除、排空完成检查。"""
        if session.released:
            return
        session.released = True
        group = self._provider.find_group(session.model)
        if group is not None:
            for role in (ROLE_PREFILL, ROLE_DECODE):
                url = session.prefill_node if role == ROLE_PREFILL else session.decode_node
                if url is None:
                    continue
                pool = group.pool_for(role)
                pool.release(url)
                if pool.try_finish_drain(url):
                    logger.info("drained node %s (%s) removed from pool", url, role)
        self._sessions.remove(session.request_id)
        self._sync_runtime()

    def _sync_runtime(self) -> None:
        self._metrics.sync_pools(self._provider)

    # --- 响应代理 -------------------------------------------------------------

    def _response_headers(self, session, assignment, prefill_body) -> dict:
        prefill = self._prefill_usage(prefill_body)
        return {
            "X-Request-Id": session.request_id,
            "X-PD-Prefill-Node": assignment.prefill_node.base_url,
            "X-PD-Decode-Node": assignment.decode_node.base_url,
            # P 口径的 cached_tokens：流式下 usage chunk 不可改写（保持字节透传），
            # 因此放在响应头。它仅用于账务/观测，Router 不参与调度决策。
            "X-PD-Prefill-Cached-Tokens": str(prefill["cached_tokens"]),
        }

    @staticmethod
    def _prefill_usage(prefill_body: dict | None) -> dict:
        usage = (prefill_body or {}).get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        cached = details.get("cached_tokens", usage.get("cached_tokens", 0)) or 0
        return {"cached_tokens": cached, "prompt_tokens": usage.get("prompt_tokens")}

    async def _proxy_stream(self, session: RequestSession, decode: DecodeStream, first_chunk: bytes = b""):
        recomputed_seen = False
        try:
            async for chunk in self._stream_chunks(decode, first_chunk):
                if session.first_byte_at is None:
                    session.record_first_byte()
                    self._metrics.ttft.observe({"model": session.model}, session.ttft_seconds or 0.0)
                    logger.info(
                        "req=%s first byte from %s, ttft=%.3fs",
                        session.request_id, session.decode_node, session.ttft_seconds,
                    )
                self._metrics.stream_bytes.inc({"model": session.model}, len(chunk))
                # recomputed 启发式：字节级透传不解析 SSE，仅做廉价子串门检
                # （finish chunk 同时含 "stop_reason" 与 "recomputed"）。
                # 误报需要模型正文恰好生成 JSON 字面量，可忽略；这正是
                # D 侧 KV 等待超时回退重算的信号（博文前提①③的失败率观测点）。
                if not recomputed_seen and b'"stop_reason"' in chunk and b"recomputed" in chunk:
                    recomputed_seen = True
                    self._metrics.recomputed.inc({"model": session.model})
                    logger.warning(
                        "req=%s decode side reported recomputed: KV wait fell back to recompute",
                        session.request_id,
                    )
                yield chunk
            session.phase = RequestPhase.COMPLETED
            self._metrics.requests.inc(
                {"model": session.model, "dispatch": self._dispatch, "outcome": "ok"})
            logger.info(
                "req=%s stream complete in %.3fs",
                session.request_id, session.elapsed_seconds(),
            )
        except Exception as exc:
            # 流已开始（200 已发出），只能记录与计数，无法改写响应状态
            session.phase = RequestPhase.FAILED
            self._metrics.failures.inc({"code": "DECODE_STREAM_INTERRUPTED"})
            self._metrics.requests.inc(
                {"model": session.model, "dispatch": self._dispatch, "outcome": "failed"})
            logger.error(
                "req=%s stream interrupted from %s after %.3fs: %r",
                session.request_id, session.decode_node, session.elapsed_seconds(), exc,
            )
            raise
        finally:
            await decode.aclose()
            self._release(session)

    async def _stream_chunks(self, decode: DecodeStream, first_chunk: bytes):
        if first_chunk:
            yield first_chunk
        async for chunk in decode.iter_bytes():
            yield chunk

    async def _proxy_json(self, session, assignment, decode: DecodeStream, prefill_body) -> Response:
        try:
            body = await decode.read_all()
            session.record_first_byte()
            self._metrics.ttft.observe({"model": session.model}, session.ttft_seconds or 0.0)
            session.phase = RequestPhase.COMPLETED
            self._metrics.requests.inc(
                {"model": session.model, "dispatch": self._dispatch, "outcome": "ok"})
            logger.info(
                "req=%s response complete in %.3fs",
                session.request_id, session.elapsed_seconds(),
            )
        finally:
            await decode.aclose()
            self._release(session)
        return Response(
            content=self._decorate_json(body, session, assignment, prefill_body),
            media_type=decode.content_type,
            headers=self._response_headers(session, assignment, prefill_body),
        )

    def _decorate_json(self, body: bytes, session, assignment, prefill_body) -> bytes:
        """cached_tokens 双口径合并：usage 以 P 为准（计费口径），D 原值入 pd 字段。

        非流式响应可精确解析，顺带做 recomputed 精确检测（流式走启发式）。
        """
        try:
            data = json.loads(body)
        except (ValueError, UnicodeDecodeError):
            return body
        if not isinstance(data, dict):
            return body
        usage = data.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        details = usage.get("prompt_tokens_details")
        details = details if isinstance(details, dict) else {}
        decode_cached = details.get("cached_tokens", usage.get("cached_tokens", 0)) or 0
        recomputed = any(
            isinstance(choice, dict) and choice.get("stop_reason") == "recomputed"
            for choice in data.get("choices") or []
        )
        if recomputed:
            self._metrics.recomputed.inc({"model": session.model})
            logger.warning(
                "req=%s decode side reported recomputed: KV wait fell back to recompute",
                session.request_id,
            )
        prefill = self._prefill_usage(prefill_body)
        if usage:
            usage["prompt_tokens_details"] = {"cached_tokens": prefill["cached_tokens"]}
            usage.setdefault("cached_tokens", prefill["cached_tokens"])
        data["pd"] = {
            "prefill": {"node": assignment.prefill_node.base_url, **prefill},
            "decode": {
                "node": assignment.decode_node.base_url,
                "cached_tokens": decode_cached,
                "recomputed": recomputed,
            },
        }
        return json.dumps(data, ensure_ascii=False).encode()
