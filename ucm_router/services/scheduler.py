"""节点选择：每池独立 Round-Robin + 健康过滤 + 在途容量上限。

v1 定下的规则在 v4 依然成立：路由策略只根据节点健康/容量决策，不根据
KVCache 命中率决策——命中判断属于 UCM Connector，Router 不感知缓存。

池空（无健康节点）返回 503；有健康节点但全部在途到上限返回 429（客户端
可退避重试的过载语义）。
"""

from __future__ import annotations

from ucm_router.domain.request import RequestSession
from ucm_router.domain.routing import RouteAssignment, ServingGroup
from ucm_router.errors import (
    DecodeAtCapacity,
    DecodeServiceUnavailable,
    PrefillAtCapacity,
    PrefillServiceUnavailable,
)


class Scheduler:
    """无状态服务；轮询游标、容量与会话亲和状态在 NodePool 上。

    affinity_key 来自请求元数据（user 字段 / X-Session-Id 头 / 会话结构），
    是确定性放置而非缓存状态感知——不违反 v4 边界。
    """

    def assign(
        self, group: ServingGroup, session: RequestSession, affinity_key: str | None = None,
    ) -> RouteAssignment:
        prefill_node = self._pick(group.prefill_pool, group.model, affinity_key)
        decode_node = self._pick(group.decode_pool, group.model, affinity_key)
        return RouteAssignment(
            request_id=session.request_id,
            group=group,
            prefill_node=prefill_node,
            decode_node=decode_node,
        )

    @staticmethod
    def _pick(pool, model: str, affinity_key: str | None):
        node = pool.pick(key=affinity_key)
        if node is not None:
            return node
        if pool.healthy_nodes():
            raise (PrefillAtCapacity if pool.role == "prefill" else DecodeAtCapacity)(
                f"all healthy {pool.role} nodes for model {model!r} are at in-flight capacity"
            )
        raise (PrefillServiceUnavailable if pool.role == "prefill" else DecodeServiceUnavailable)(
            f"no healthy {pool.role} node for model {model!r}"
        )
