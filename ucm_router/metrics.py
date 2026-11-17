"""Router 自有指标：prometheus_client 的 Counter / Gauge / Histogram。

不自己写 exposition。vLLM 的 requirements/common.txt 里就有
`prometheus_client >= 0.18.0`，UCM 的 connector 指标也是经 vLLM 的 Prometheus
桥导出的；Router 和引擎跑在同一套环境里，用同一个库，文本格式、标签转义、
分桶语义都跟着标准走，不必在这三处重复实现。

只用私有 CollectorRegistry，不用库默认的 REGISTRY：默认那个挂着 process_* /
python_* 一组采集器，而本项目的 /metrics 还要与 P/D 抓来的 exposition 合并，
塞一堆进程噪声进去没有意义（要的话 main.py 里换成默认 registry 即可）。

耗时类是 Histogram 而不是 sum/count 对：Prometheus 侧能直接出分位数。
"""

from __future__ import annotations

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# 秒级分桶：从毫秒级的建流等待铺到分钟级的长 prompt prefill。
# 上限取 300s，与 engine.decode_timeout_seconds 的默认值对齐。
DURATION_BUCKETS: tuple[float, ...] = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0,
    10.0, 30.0, 60.0, 120.0, 300.0,
)


class RouterMetrics:
    """Router 自有指标集合。

    标签名按字母序声明，exposition 里输出的顺序与声明顺序一致，
    方便和 README 里的指标表、既有看板对齐。
    """

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        reg = self.registry

        self.requests = Counter(
            "pd_requests_total",
            "requests by model/dispatch/outcome",
            ["dispatch", "model", "outcome"], registry=reg)
        self.failures = Counter(
            "pd_failures_total",
            "failed requests by error code",
            ["code"], registry=reg)
        self.selections = Counter(
            "pd_node_selections_total",
            "node selections by role/instance",
            ["instance", "role"], registry=reg)
        self.retries = Counter(
            "pd_retries_total",
            "in-request failover retries by role",
            ["role"], registry=reg)
        self.recomputed = Counter(
            "pd_recomputed_total",
            "decode requests where the engine fell back to recompute "
            "(vLLM stop_reason=recomputed; D 侧 KV 等待超时/失败的关键观测点)",
            ["model"], registry=reg)
        self.scrape_failures = Counter(
            "pd_scrape_failures_total",
            "backend /metrics scrape failures by role/instance",
            ["instance", "role"], registry=reg)
        self.stream_bytes = Counter(
            "pd_stream_bytes_total",
            "bytes proxied from decode nodes by model",
            ["model"], registry=reg)

        self.prefill_duration = Histogram(
            "pd_prefill_duration_seconds",
            "prefill phase duration including failover retries",
            ["model"], buckets=DURATION_BUCKETS, registry=reg)
        self.ttft = Histogram(
            "pd_ttft_seconds",
            "router-observed time to first byte from decode (incl. prefill)",
            ["model"], buckets=DURATION_BUCKETS, registry=reg)
        self.dispatch_lead = Histogram(
            "pd_early_dispatch_lead_seconds",
            "early 模式下从选路完成到 P 返回、D 建好响应流的时长（被 P 计算掩盖的等待上界）",
            ["model"], buckets=DURATION_BUCKETS, registry=reg)

        self.sessions_active = Gauge(
            "pd_sessions_active",
            "in-flight sessions per node",
            ["instance", "role"], registry=reg)
        self.pool_healthy = Gauge(
            "pd_pool_healthy",
            "healthy node count per pool",
            ["model", "role"], registry=reg)

    def render(self) -> str:
        """Router 自有指标的 exposition 文本（不含 P/D 抓来的部分）。"""
        return generate_latest(self.registry).decode()

    def sync_pools(self, provider) -> None:
        """把 NodePool 的实时状态刷进 gauge（在选路/释放/健康探测/增删节点后调用）。"""
        for group in provider.groups():
            for pool in (group.prefill_pool, group.decode_pool):
                for node in pool.nodes:
                    self.sessions_active.labels(
                        instance=node.base_url, role=pool.role,
                    ).set(pool.inflight.get(node.base_url, 0))
                self.pool_healthy.labels(
                    model=group.model, role=pool.role,
                ).set(len(pool.healthy_nodes()))


__all__ = ["DURATION_BUCKETS", "RouterMetrics"]
