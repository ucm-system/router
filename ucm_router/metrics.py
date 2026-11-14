"""Router 自有指标：手写的极简 Prometheus 文本 exposition（无第三方依赖）。

边界：router 只报自己视角的事——选路、请求生命周期、耗时、错误。
KV 命中 / 等待 / 重算是 UCM Connector 的知识，这里一律不出现。
"""

from __future__ import annotations

from typing import Iterator

LabelKey = tuple[tuple[str, str], ...]


def _label_key(labels: dict[str, str] | None) -> LabelKey:
    return tuple(sorted((labels or {}).items()))


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _render_labels(labels: LabelKey) -> str:
    if not labels:
        return ""
    inner = ",".join(f'{k}="{_escape(v)}"' for k, v in labels)
    return f"{{{inner}}}"


class _Family:
    def __init__(self, name: str, help_text: str, metric_type: str) -> None:
        self.name = name
        self.help = help_text
        self.type = metric_type
        self.samples: dict[LabelKey, float] = {}

    def _render(self) -> Iterator[str]:
        yield f"# HELP {self.name} {self.help}"
        yield f"# TYPE {self.name} {self.type}"
        for key in sorted(self.samples):
            yield f"{self.name}{_render_labels(key)} {_format_value(self.samples[key])}"


def _format_value(value: float) -> str:
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return repr(value)


class Counter:
    def __init__(self, family: _Family) -> None:
        self._family = family

    def inc(self, labels: dict[str, str] | None = None, amount: float = 1.0) -> None:
        key = _label_key(labels)
        self._family.samples[key] = self._family.samples.get(key, 0.0) + amount


class Gauge:
    def __init__(self, family: _Family) -> None:
        self._family = family

    def set(self, labels: dict[str, str], value: float) -> None:
        self._family.samples[_label_key(labels)] = value

    def get(self, labels: dict[str, str]) -> float:
        return self._family.samples.get(_label_key(labels), 0.0)


class Timing:
    """sum/count 对：够用的耗时观测，不引入 histogram 分桶。"""

    def __init__(self, total: Counter, count: Counter) -> None:
        self._total = total
        self._count = count

    def observe(self, labels: dict[str, str], seconds: float) -> None:
        self._total.inc(labels, seconds)
        self._count.inc(labels)


class MetricsRegistry:
    def __init__(self) -> None:
        self._families: list[_Family] = []

    def _family(self, name: str, help_text: str, metric_type: str) -> _Family:
        family = _Family(name, help_text, metric_type)
        self._families.append(family)
        return family

    def counter(self, name: str, help_text: str) -> Counter:
        return Counter(self._family(name, help_text, "counter"))

    def gauge(self, name: str, help_text: str) -> Gauge:
        return Gauge(self._family(name, help_text, "gauge"))

    def timing(self, name: str, help_text: str) -> Timing:
        return Timing(
            self.counter(f"{name}_sum", help_text),
            self.counter(f"{name}_count", f"number of {name} observations"),
        )

    def render(self) -> str:
        lines: list[str] = []
        for family in self._families:
            lines.extend(family._render())
        return "\n".join(lines) + "\n"


class RouterMetrics:
    """Router 自有指标集合。"""

    def __init__(self, registry: MetricsRegistry) -> None:
        self.requests = registry.counter(
            "pd_requests_total", "requests by model/dispatch/outcome")
        self.failures = registry.counter(
            "pd_failures_total", "failed requests by error code")
        self.selections = registry.counter(
            "pd_node_selections_total", "node selections by role/instance")
        self.retries = registry.counter(
            "pd_retries_total", "in-request failover retries by role")
        self.recomputed = registry.counter(
            "pd_recomputed_total",
            "decode requests where the engine fell back to recompute "
            "(vLLM stop_reason=recomputed; D 侧 KV 等待超时/失败的关键观测点)")
        self.scrape_failures = registry.counter(
            "pd_scrape_failures_total", "backend /metrics scrape failures by role/instance")
        self.stream_bytes = registry.counter(
            "pd_stream_bytes_total", "bytes proxied from decode nodes by model")
        self.prefill_duration = registry.timing(
            "pd_prefill_duration_seconds", "prefill phase duration including failover retries")
        self.ttft = registry.timing(
            "pd_ttft_seconds", "router-observed time to first byte from decode (incl. prefill)")
        self.dispatch_lead = registry.timing(
            "pd_early_dispatch_lead_seconds",
            "early 模式下 decode 先于 prefill 完成下发的时长（等待被 P 计算掩盖的量）")
        self.sessions_active = registry.gauge(
            "pd_sessions_active", "in-flight sessions per node")
        self.pool_healthy = registry.gauge(
            "pd_pool_healthy", "healthy node count per pool")

    def sync_pools(self, provider) -> None:
        """把 NodePool 的实时状态刷进 gauge（在选路/释放/健康探测/增删节点后调用）。"""
        for group in provider.groups():
            for pool in (group.prefill_pool, group.decode_pool):
                for node in pool.nodes:
                    self.sessions_active.set(
                        {"role": pool.role, "instance": node.base_url},
                        pool.inflight.get(node.base_url, 0),
                    )
                self.pool_healthy.set(
                    {"model": group.model, "role": pool.role},
                    len(pool.healthy_nodes()),
                )
