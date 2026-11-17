"""PD 分 metrics 汇总透传：抓取每个引擎实例的 /metrics（Prometheus 文本格式），
按角色改名（pd_{role}_ 前缀）并注入 pd_role / pd_instance 标签后与 Router 自有
指标合并输出。

不改名的话 P、D 会输出同名 metric family，Prometheus 抓取合并端点时会因
重复 family 报错；改名后 P、D 的同名指标各自成 family，实例维度靠标签区分。
"""

from __future__ import annotations

import asyncio
import logging
import re

import httpx

from ucm_router.domain.routing import ServingNode
from ucm_router.ports.node_provider import NodeProvider
from ucm_router.metrics import RouterMetrics

logger = logging.getLogger("ucm_router.metrics_proxy")

SCRAPE_TIMEOUT_SECONDS = 3.0

_SAMPLE_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{(.*)\})?\s+(.+)$")


def rewrite_scraped(text: str, role: str, instance: str) -> str:
    """把一个实例的 exposition 改名 + 注入标签。无法解析的行丢弃。"""
    injected = f'pd_role="{role}",pd_instance="{instance}"'
    out: list[str] = []
    for line in text.splitlines():
        if line.startswith("# HELP ") or line.startswith("# TYPE "):
            parts = line.split(" ", 3)
            if len(parts) >= 3:
                rest = parts[3] if len(parts) > 3 else ""
                out.append(f"{parts[0]} {parts[1]} pd_{role}_{parts[2]} {rest}".rstrip())
            continue
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE_RE.match(line)
        if match is None:
            continue
        name, labels, value = match.group(1), match.group(3), match.group(4)
        new_labels = f"{{{injected},{labels}}}" if labels else f"{{{injected}}}"
        out.append(f"pd_{role}_{name}{new_labels} {value}")
    return "\n".join(out) + "\n"


class MetricsProxy:
    def __init__(
        self, provider: NodeProvider, metrics: RouterMetrics
    ) -> None:
        self._provider = provider
        self._metrics = metrics
        self._client = httpx.AsyncClient()

    async def render(self) -> str:
        # Router 自有指标在前，PD 分改名合并的在后
        sections = [self._metrics.render()]
        nodes = [
            node
            for group in self._provider.groups()
            for pool in (group.prefill_pool, group.decode_pool)
            for node in pool.nodes
        ]
        results = await asyncio.gather(*(self._scrape(node) for node in nodes))
        sections.extend(r for r in results if r)
        return "\n".join(sections)

    async def _scrape(self, node: ServingNode) -> str | None:
        try:
            response = await self._client.get(
                f"{node.base_url}/metrics", timeout=SCRAPE_TIMEOUT_SECONDS)
        except httpx.HTTPError as exc:
            logger.warning("scrape %s failed: %r", node.base_url, exc)
            self._metrics.scrape_failures.labels(
                instance=node.base_url, role=node.role).inc()
            return None
        if response.status_code >= 400:
            logger.warning("scrape %s returned %d", node.base_url, response.status_code)
            self._metrics.scrape_failures.labels(
                instance=node.base_url, role=node.role).inc()
            return None
        return rewrite_scraped(response.text, node.role, node.base_url)

    async def aclose(self) -> None:
        await self._client.aclose()
