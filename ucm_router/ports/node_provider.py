"""Port: NodeProvider —— 获取节点。

v3/v4 定稿保留的三个 Port 之一。第一版实现为静态 YAML + admin API 动态增删；
K8s 服务发现是未来实现，不需要改 Domain 与 Application 层。
"""

from __future__ import annotations

from typing import Protocol

from ucm_router.config import RouterConfig, ServingGroupConfig
from ucm_router.domain.routing import NodePool, ServingGroup, ServingNode


class NodeProvider(Protocol):
    def groups(self) -> list[ServingGroup]: ...

    def find_group(self, model: str) -> ServingGroup | None: ...

    def pool_summary(self) -> list[dict]: ...


class StaticNodeProvider:
    """从 RouterConfig 构建静态 ServingGroup 列表。

    admin API 通过 NodePool 的 add_node/remove_node 在运行时增删节点；
    变更不持久化，重启后以 YAML 为准。
    """

    def __init__(self, config: RouterConfig) -> None:
        self._groups = [
            _build_group(group, config.routing.max_inflight_per_node)
            for group in config.routing.serving_groups
        ]

    def groups(self) -> list[ServingGroup]:
        return list(self._groups)

    def find_group(self, model: str) -> ServingGroup | None:
        for group in self._groups:
            if group.model == model:
                return group
        return None

    def pool_summary(self) -> list[dict]:
        return [
            {
                "model": group.model,
                "prefill": group.prefill_pool.summary(),
                "decode": group.decode_pool.summary(),
            }
            for group in self._groups
        ]


def _build_group(group: ServingGroupConfig, max_inflight: int) -> ServingGroup:
    prefill_nodes = [ServingNode(role="prefill", base_url=url) for url in group.prefill]
    decode_nodes = [ServingNode(role="decode", base_url=url) for url in group.decode]
    return ServingGroup(
        model=group.model,
        prefill_pool=NodePool(role="prefill", nodes=prefill_nodes, max_inflight=max_inflight),
        decode_pool=NodePool(role="decode", nodes=decode_nodes, max_inflight=max_inflight),
    )
