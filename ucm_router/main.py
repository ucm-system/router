"""FastAPI 应用：OpenAI 兼容入口 + /metrics 汇总 + admin API + router 自身健康端点。"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from ucm_router.config import RouterConfig
from ucm_router.domain.routing import ServingNode
from ucm_router.errors import (
    InvalidRequest,
    NodeExists,
    NodeNotFound,
    NoServingGroup,
    RouterError,
)
from ucm_router.metrics import MetricsRegistry, RouterMetrics
from ucm_router.ports.engine_client import VllmOpenAIClient
from ucm_router.ports.node_provider import StaticNodeProvider
from ucm_router.ports.session_repository import InMemorySessionRepository
from ucm_router.services.health import HealthChecker
from ucm_router.services.metrics_proxy import MetricsProxy
from ucm_router.services.orchestrator import Orchestrator
from ucm_router.services.scheduler import Scheduler

logger = logging.getLogger("ucm_router")

PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


def create_app(config: RouterConfig) -> FastAPI:
    provider = StaticNodeProvider(config)
    registry = MetricsRegistry()
    metrics = RouterMetrics(registry)
    metrics.sync_pools(provider)
    scheduler = Scheduler()
    sessions = InMemorySessionRepository()
    engine = VllmOpenAIClient(config.engine)
    health = HealthChecker(provider, config.health, metrics)
    metrics_proxy = MetricsProxy(provider, registry, metrics)
    orchestrator = Orchestrator(
        provider, scheduler, engine, sessions, health, config, metrics)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await health.start()
        logger.info(
            "router started: %d serving group(s), decode_dispatch=%s",
            len(provider.groups()), config.decode_dispatch,
        )
        yield
        await health.stop()
        await engine.aclose()
        await metrics_proxy.aclose()

    app = FastAPI(title="ucm-router", version="0.1.0", lifespan=lifespan)
    app.state.orchestrator = orchestrator
    app.state.sessions = sessions
    app.state.provider = provider
    app.state.health = health
    app.state.metrics_proxy = metrics_proxy

    @app.exception_handler(RouterError)
    async def router_error_handler(_: Request, error: RouterError) -> JSONResponse:
        logger.warning("req failed: %s %s", error.code, error.message)
        return JSONResponse(status_code=error.http_status, content=error.to_body())

    async def generation_entry(request: Request) -> Response:
        """PD 分离对客户端透明：请求体不改写地进入编排。"""
        try:
            payload = await request.json()
        except Exception as exc:
            raise InvalidRequest("request body must be valid JSON") from exc
        if not isinstance(payload, dict):
            raise InvalidRequest("request body must be a JSON object")
        model = payload.get("model")
        if not isinstance(model, str) or not model:
            raise InvalidRequest("'model' is required")
        return await app.state.orchestrator.handle(request.url.path, payload, request.headers)

    app.add_api_route("/v1/chat/completions", generation_entry, methods=["POST"])
    app.add_api_route("/v1/completions", generation_entry, methods=["POST"])

    @app.get("/health")
    async def router_health() -> dict:
        return {
            "status": "ok",
            "active_sessions": app.state.sessions.active_count(),
            "pools": app.state.provider.pool_summary(),
        }

    def _all_healthy_nodes() -> list:
        return [
            node
            for group in provider.groups()
            for pool in (group.decode_pool, group.prefill_pool)
            for node in pool.healthy_nodes()
        ]

    @app.get("/v1/models")
    async def v1_models() -> Response:
        """代理到任一健康实例（ascend toy proxy 同语义：取第一个成功的）。"""
        failures: list[str] = []
        for node in _all_healthy_nodes():
            status, body = await engine.get(node, "/v1/models")
            if 200 <= status < 400:
                return Response(content=body, media_type="application/json")
            failures.append(node.base_url)
        return JSONResponse(
            status_code=503,
            content={"error": {
                "code": "NO_BACKEND_AVAILABLE",
                "message": "all backend instances failed to serve /v1/models",
                "failed": failures,
            }},
        )

    @app.post("/reset_prefix_cache")
    async def reset_prefix_cache() -> dict:
        """广播到所有健康实例，返回逐实例结果。"""
        nodes = _all_healthy_nodes()
        results = await asyncio.gather(
            *(engine.post(node, "/reset_prefix_cache") for node in nodes))
        failed = [
            node.base_url for node, (status, _) in zip(nodes, results)
            if status < 200 or status >= 400
        ]
        return {"reset": len(nodes) - len(failed), "failed": failed}

    @app.get("/metrics")
    async def metrics_endpoint() -> Response:
        text = await app.state.metrics_proxy.render()
        return Response(content=text, media_type=PROMETHEUS_CONTENT_TYPE)

    # --- admin API：动态增删节点（不持久化，重启以 YAML 为准） ---------------

    def _group_or_404(model: str):
        group = provider.find_group(model)
        if group is None:
            raise NoServingGroup(f"no serving group matches model {model!r}")
        return group

    def _pool_or_400(group, role: str):
        try:
            return group.pool_for(role)
        except ValueError as exc:
            raise InvalidRequest(str(exc)) from exc

    @app.get("/admin/nodes")
    async def list_nodes() -> list[dict]:
        return provider.pool_summary()

    @app.post("/admin/nodes")
    async def add_node(request: Request) -> Response:
        body = await _json_body(request, ("model", "role", "url"))
        group = _group_or_404(body["model"])
        role = body["role"]
        pool = _pool_or_400(group, role)
        url = body["url"]
        node = ServingNode(role=role, base_url=url)
        if pool.has(url):
            raise NodeExists(f"node {url} already in {role} pool of {body['model']!r}")
        if await health.probe(node):
            pool.add_node(node)
            metrics.sync_pools(provider)
            logger.info("admin: node %s added to %s pool of %r", url, role, body["model"])
            return JSONResponse(status_code=201, content={
                "added": True, "waiting": False, "pool": pool.summary()})
        # 探测未通过：不直接拒绝，进等待队列由健康检查轮询，就绪后自动激活
        # （ascend toy proxy 的 waiting_nodes 语义）
        pool.add_waiting(node)
        metrics.sync_pools(provider)
        logger.info(
            "admin: node %s not ready yet, queued as waiting (%s pool of %r, "
            "polling every %.1fs, max %d attempts)",
            url, role, body["model"], config.health.interval_seconds,
            config.health.waiting_max_retries,
        )
        return JSONResponse(status_code=202, content={
            "added": False, "waiting": True, "pool": pool.summary()})

    @app.delete("/admin/nodes")
    async def remove_node(
        model: str, role: str, url: str, drain: bool = True,
    ) -> dict:
        group = _group_or_404(model)
        pool = _pool_or_400(group, role)
        if url in pool.waiting:
            pool.drop_waiting(url)
            metrics.sync_pools(provider)
            logger.info("admin: waiting node %s dropped from %s pool of %r", url, role, model)
            return {"removed": True, "draining": False, "was_waiting": True}
        if not pool.has(url):
            raise NodeNotFound(f"node {url} not in {role} pool of {model!r}")
        if not drain:
            pool.remove_node(url)
            metrics.sync_pools(provider)
            logger.info("admin: node %s removed from %s pool of %r (no drain)", url, role, model)
            return {"removed": True, "draining": False}
        pool.mark_draining(url)
        if pool.try_finish_drain(url):
            metrics.sync_pools(provider)
            logger.info("admin: node %s removed from %s pool of %r (drain: no in-flight)", url, role, model)
            return {"removed": True, "draining": False}
        logger.info(
            "admin: node %s draining from %s pool of %r (%d in-flight)",
            url, role, model, pool.inflight.get(url, 0),
        )
        return {"removed": False, "draining": True, "inflight": pool.inflight.get(url, 0)}

    return app


async def _json_body(request: Request, required_keys: tuple[str, ...]) -> dict:
    try:
        body = await request.json()
    except Exception as exc:
        raise InvalidRequest("request body must be valid JSON") from exc
    if not isinstance(body, dict):
        raise InvalidRequest("request body must be a JSON object")
    for key in required_keys:
        value = body.get(key)
        if not isinstance(value, str) or not value:
            raise InvalidRequest(f"{key!r} is required")
    return body
