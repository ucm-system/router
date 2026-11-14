"""Routing 域：v4 中 Router 保留的两个领域之一。

只回答"请求该由哪个 P 和哪个 D 处理"。ServingGroup 承载 KV 兼容性约束：
模型版本、Tokenizer、KV 数据类型、KV Layout、并行配置不一致的节点不能配成一对。

路由策略只根据节点健康状态、在途容量与会话标识（可选）做决策，不根据
KVCache 命中率做决策——命中判断属于 D 侧 UCM Connector，Router 完全不感知
Hash（v4 定稿）。会话亲和（HRW 哈希）是确定性放置：key 来自请求元数据
（user 字段 / X-Session-Id 头 / 会话结构），不需要 Tokenizer 或缓存状态。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

ROLE_PREFILL = "prefill"
ROLE_DECODE = "decode"


def affinity_score(key: str, base_url: str) -> bytes:
    """Rendezvous (HRW) 哈希：跨进程/重启稳定，仅当节点被移除时才重映射
    恰好落在该节点上的会话（vllm-ascend toy proxy 同款设计）。"""
    return hashlib.sha256(f"{key}\0{base_url}".encode()).digest()


@dataclass(frozen=True)
class ServingNode:
    """一个引擎实例。"""

    role: str            # "prefill" | "decode"
    base_url: str        # 例: http://127.0.0.1:7800（不带路径）

    @property
    def health_url(self) -> str:
        return f"{self.base_url}/health"


@dataclass
class NodePool:
    """同角色节点的池。

    healthy 集合由健康检查更新（启动探测 + 周期探测 + 请求失败时的被动摘除）；
    draining 集合中的节点不再接新请求，在途清零后由 try_finish_drain 移除；
    waiting 中的节点是 admin API 新增但尚未通过健康探测的（后台轮询就绪后激活，
    超过重试轮数放弃）；inflight 记录每节点在途请求数（背压上限的依据）；
    _cursor 是本池独立的 round-robin 游标（P、D 各自轮询，互不影响）。
    """

    role: str
    nodes: list[ServingNode]
    healthy: set[str] = field(default_factory=set)
    max_inflight: int = 0                     # 0 = 不限
    draining: set[str] = field(default_factory=set)
    inflight: dict[str, int] = field(default_factory=dict)
    waiting: dict[str, tuple[ServingNode, int]] = field(default_factory=dict)  # url -> (node, 已探测轮数)
    _cursor: int = 0

    def __post_init__(self) -> None:
        if not self.healthy:
            # 初值乐观：全部视为健康，由健康检查收敛（启动时会先探测一轮）
            self.healthy = {node.base_url for node in self.nodes}

    # --- 选路 -----------------------------------------------------------

    def healthy_nodes(self) -> list[ServingNode]:
        return [node for node in self.nodes if node.base_url in self.healthy]

    def _candidates(self, exclude: str | None) -> list[ServingNode]:
        return [
            node for node in self.nodes
            if node.base_url in self.healthy
            and node.base_url not in self.draining
            and node.base_url != exclude
            and (self.max_inflight <= 0
                 or self.inflight.get(node.base_url, 0) < self.max_inflight)
        ]

    def pick(self, key: str | None = None, exclude: str | None = None) -> ServingNode | None:
        """选一个可用节点；无可用节点返回 None。

        带 key 时用 Rendezvous (HRW) 哈希做会话亲和（确定性放置），
        否则 round-robin。可用 = 健康 + 非排空 + 非排除 + 在途未到上限。
        """
        candidates = self._candidates(exclude)
        if not candidates:
            return None
        if key:
            return max(candidates, key=lambda n: (affinity_score(key, n.base_url), n.base_url))
        node = candidates[self._cursor % len(candidates)]
        self._cursor += 1
        return node

    # --- 在途计数（orchestrator 在分配/释放时调用） -----------------------

    def acquire(self, url: str) -> None:
        self.inflight[url] = self.inflight.get(url, 0) + 1

    def release(self, url: str) -> None:
        count = self.inflight.get(url, 0) - 1
        if count > 0:
            self.inflight[url] = count
        else:
            self.inflight.pop(url, None)

    # --- 动态增删（admin API 调用） ---------------------------------------

    def has(self, url: str) -> bool:
        if any(node.base_url == url for node in self.nodes):
            return True
        return url in self.waiting

    def add_node(self, node: ServingNode) -> None:
        """加入并立即参与轮询（调用方须先探测健康）。"""
        self.nodes.append(node)
        self.healthy.add(node.base_url)

    def add_waiting(self, node: ServingNode) -> None:
        """admin 新增但探测未通过的节点：进等待队列，由健康检查轮询激活。"""
        self.waiting.setdefault(node.base_url, (node, 0))

    def waiting_nodes(self) -> list[ServingNode]:
        return [node for node, _ in self.waiting.values()]

    def bump_waiting(self, url: str) -> int:
        node, attempts = self.waiting[url]
        self.waiting[url] = (node, attempts + 1)
        return attempts + 1

    def activate_waiting(self, node: ServingNode) -> None:
        self.waiting.pop(node.base_url, None)
        self.add_node(node)

    def drop_waiting(self, url: str) -> None:
        self.waiting.pop(url, None)

    def remove_node(self, url: str) -> bool:
        before = len(self.nodes)
        self.nodes = [node for node in self.nodes if node.base_url != url]
        self.healthy.discard(url)
        self.draining.discard(url)
        self.inflight.pop(url, None)
        self.waiting.pop(url, None)
        return len(self.nodes) < before

    def mark_draining(self, url: str) -> None:
        """排空：停止接新请求；在途清零后由 try_finish_drain 移除。"""
        self.healthy.discard(url)
        self.draining.add(url)

    def try_finish_drain(self, url: str) -> bool:
        """排空完成（无在途）时真正移除节点，返回是否移除。"""
        if url in self.draining and self.inflight.get(url, 0) == 0:
            self.remove_node(url)
            return True
        return False

    def summary(self) -> dict:
        return {
            "total": len(self.nodes),
            "healthy": len(self.healthy_nodes()),
            "max_inflight": self.max_inflight,
            "nodes": [
                {
                    "url": node.base_url,
                    "healthy": node.base_url in self.healthy,
                    "draining": node.base_url in self.draining,
                    "inflight": self.inflight.get(node.base_url, 0),
                }
                for node in self.nodes
            ],
            "waiting": [
                {"url": url, "probe_attempts": attempts}
                for url, (_, attempts) in sorted(self.waiting.items())
            ],
        }


@dataclass(frozen=True)
class ServingGroup:
    """一组 KV 兼容的 P/D 实例，按模型名匹配请求。"""

    model: str
    prefill_pool: NodePool
    decode_pool: NodePool

    def pool_for(self, role: str) -> NodePool:
        if role == ROLE_PREFILL:
            return self.prefill_pool
        if role == ROLE_DECODE:
            return self.decode_pool
        raise ValueError(f"unknown role: {role!r}")


@dataclass(frozen=True)
class RouteAssignment:
    """一次冻结的 P/D 分配结果（failover 换节点时以 replace 生成新实例）。"""

    request_id: str
    group: ServingGroup
    prefill_node: ServingNode
    decode_node: ServingNode
