"""健康检查：周期探测各节点 GET /health，维护 NodePool 健康集合。

它不是独立领域——健康与否是"节点选择"的输入。三条来源：
1. 启动时先探测一轮再放流量（避免请求打到启动前就挂掉的节点）；
2. 周期探测（interval）；这是主动途径，探测间隔内节点死亡存在盲区；
3. 被动摘除（mark_unreachable）：请求遇到连接级失败时立即摘除，自愈盲区。
"""

from __future__ import annotations

import asyncio
import logging

import httpx

from ucm_router.config import HealthConfig
from ucm_router.domain.routing import NodePool, ServingNode
from ucm_router.ports.node_provider import NodeProvider

logger = logging.getLogger("ucm_router.health")


class HealthChecker:
    def __init__(
        self,
        provider: NodeProvider,
        config: HealthConfig,
        metrics=None,
    ) -> None:
        self._provider = provider
        self._interval = config.interval_seconds
        self._timeout = config.timeout_seconds
        self._waiting_max_retries = config.waiting_max_retries
        self._metrics = metrics
        self._client = httpx.AsyncClient()
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        await self.probe_round()
        self._task = asyncio.create_task(self._loop(), name="health-checker")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        await self._client.aclose()

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(self._interval)
            try:
                await self.probe_round()
            except Exception:  # noqa: BLE001 —— 探测循环自身绝不能死
                logger.exception("health probe round failed unexpectedly")

    async def probe_round(self) -> None:
        for group in self._provider.groups():
            await self._probe_pool(group.prefill_pool)
            await self._probe_pool(group.decode_pool)
            await self._probe_waiting(group.prefill_pool)
            await self._probe_waiting(group.decode_pool)
        self._sync_metrics()

    async def _probe_waiting(self, pool: NodePool) -> None:
        """admin 动态添加但尚未就绪的节点：轮询激活或放弃（ascend toy proxy 同语义）。"""
        for node, attempts in list(pool.waiting.values()):
            if await self.probe(node):
                pool.activate_waiting(node)
                logger.info("waiting node %s (%s) is ready: added to rotation",
                            node.base_url, pool.role)
                continue
            attempts = pool.bump_waiting(node.base_url)
            if attempts >= self._waiting_max_retries:
                pool.drop_waiting(node.base_url)
                logger.warning("waiting node %s (%s) not ready after %d probes: dropped",
                               node.base_url, pool.role, attempts)

    async def _probe_pool(self, pool: NodePool) -> None:
        for node in pool.nodes:
            healthy = await self.probe(node)
            was = node.base_url in pool.healthy
            if healthy and not was:
                pool.healthy.add(node.base_url)
                logger.info("node %s (%s) is back: added to rotation", node.base_url, pool.role)
            elif not healthy and was:
                pool.healthy.discard(node.base_url)
                logger.warning("node %s (%s) is down: removed from rotation", node.base_url, pool.role)

    async def probe(self, node: ServingNode) -> bool:
        """单节点探测。admin API 加节点前也用它。"""
        try:
            response = await self._client.get(node.health_url, timeout=self._timeout)
        except httpx.HTTPError:
            return False
        return response.status_code < 400

    def mark_unreachable(self, node: ServingNode) -> None:
        """被动健康：请求遇到连接级失败时立即摘除，不必等下一个探测周期。"""
        for group in self._provider.groups():
            for pool in (group.prefill_pool, group.decode_pool):
                if node.base_url not in pool.healthy:
                    continue
                if any(n.base_url == node.base_url for n in pool.nodes):
                    pool.healthy.discard(node.base_url)
                    logger.warning(
                        "node %s (%s) unreachable at request time: passively removed from rotation",
                        node.base_url, pool.role,
                    )
        self._sync_metrics()

    def _sync_metrics(self) -> None:
        if self._metrics is not None:
            self._metrics.sync_pools(self._provider)
