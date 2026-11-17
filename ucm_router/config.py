"""配置加载。

最基础形态是静态 YAML（K8s 服务发现留作 NodeProvider 的后续实现，
不影响 Domain 与 Application 层）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DECODE_DISPATCH_SEQUENTIAL = "sequential"
DECODE_DISPATCH_EARLY = "early"
_DEPATCH_MODES = (DECODE_DISPATCH_SEQUENTIAL, DECODE_DISPATCH_EARLY)


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 9000


@dataclass
class EngineConfig:
    """对后端引擎的 HTTP 超时与请求口径。"""

    connect_timeout_seconds: float = 5.0
    prefill_timeout_seconds: float = 120.0
    decode_timeout_seconds: float = 300.0
    # 在 kv_transfer_params.router 里声明本次编排（角色 / request_id / kv_load）。
    # 关掉即完全透传 decode body，用于和 connector 侧对齐前的回退。
    router_hint: bool = True


@dataclass
class HealthConfig:
    interval_seconds: float = 5.0
    timeout_seconds: float = 2.0
    waiting_max_retries: int = 12    # admin 动态添加的节点等待就绪的最大探测轮数


@dataclass
class ServingGroupConfig:
    model: str
    prefill: list[str] = field(default_factory=list)
    decode: list[str] = field(default_factory=list)


@dataclass
class RoutingConfig:
    serving_groups: list[ServingGroupConfig] = field(default_factory=list)
    max_inflight_per_node: int = 0    # 每节点在途请求上限；0 = 不限（超限返回 429）
    session_affinity: bool = False    # 会话亲和（HRW）：同会话稳定落到同一组 P/D


@dataclass
class RouterConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    decode_dispatch: str = DECODE_DISPATCH_SEQUENTIAL
    engine: EngineConfig = field(default_factory=EngineConfig)
    health: HealthConfig = field(default_factory=HealthConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)


class ConfigError(ValueError):
    pass


def load_config(path: str | Path) -> RouterConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError("config file must contain a YAML mapping")

    dispatch = raw.get("decode_dispatch", DECODE_DISPATCH_SEQUENTIAL)
    if dispatch not in _DEPATCH_MODES:
        raise ConfigError(f"decode_dispatch must be one of {_DEPATCH_MODES}, got {dispatch!r}")

    groups_raw = raw.get("routing", {}).get("serving_groups", [])
    if not groups_raw:
        raise ConfigError("routing.serving_groups must contain at least one group")

    groups: list[ServingGroupConfig] = []
    for i, group in enumerate(groups_raw):
        model = group.get("model")
        if not model or not isinstance(model, str):
            raise ConfigError(f"serving_groups[{i}].model must be a non-empty string")
        prefill = _endpoint_list(group, "prefill", i)
        decode = _endpoint_list(group, "decode", i)
        groups.append(ServingGroupConfig(model=model, prefill=prefill, decode=decode))

    routing_raw = raw.get("routing") or {}
    max_inflight = int(routing_raw.get("max_inflight_per_node", 0) or 0)
    if max_inflight < 0:
        raise ConfigError("routing.max_inflight_per_node must be >= 0")
    session_affinity = bool(routing_raw.get("session_affinity", False))

    return RouterConfig(
        server=_section(raw, "server", ServerConfig),
        decode_dispatch=dispatch,
        engine=_section(raw, "engine", EngineConfig),
        health=_section(raw, "health", HealthConfig),
        routing=RoutingConfig(
            serving_groups=groups,
            max_inflight_per_node=max_inflight,
            session_affinity=session_affinity,
        ),
    )


def _endpoint_list(group: dict[str, Any], role: str, index: int) -> list[str]:
    endpoints = group.get(role)
    if not endpoints or not isinstance(endpoints, list) or not all(isinstance(u, str) and u for u in endpoints):
        raise ConfigError(f"serving_groups[{index}].{role} must be a non-empty list of URLs")
    return endpoints


def _section(raw: dict[str, Any], name: str, dataclass_type: type) -> Any:
    """把 YAML 的子表合并到对应 dataclass 的默认值上。"""
    section = raw.get(name) or {}
    known = {f for f in dataclass_type.__dataclass_fields__}
    unknown = set(section) - known
    if unknown:
        raise ConfigError(f"unknown keys in '{name}': {sorted(unknown)}")
    return dataclass_type(**section)
