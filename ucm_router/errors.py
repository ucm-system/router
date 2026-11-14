"""错误模型。

失败语义是最基础的 fail-fast，不重试：重试意味着重复 prefill，涉及幂等性
设计，不属于"最基础"范围。错误码区分 P / D，客户端据此知道哪个池出了问题。
"""

from __future__ import annotations

NO_SERVING_GROUP = "NO_SERVING_GROUP"
INVALID_REQUEST = "INVALID_REQUEST"
PREFILL_SERVICE_UNAVAILABLE = "PREFILL_SERVICE_UNAVAILABLE"  # P 池为空（无健康节点）
DECODE_SERVICE_UNAVAILABLE = "DECODE_SERVICE_UNAVAILABLE"    # D 池为空（无健康节点）
PREFILL_AT_CAPACITY = "PREFILL_AT_CAPACITY"                  # P 池在途全部到上限
DECODE_AT_CAPACITY = "DECODE_AT_CAPACITY"                    # D 池在途全部到上限
PREFILL_REQUEST_FAILED = "PREFILL_REQUEST_FAILED"            # P 不可达 / 超时 / 非 2xx
DECODE_REQUEST_FAILED = "DECODE_REQUEST_FAILED"              # D 不可达 / 超时 / 非 2xx
DECODE_STREAM_INTERRUPTED = "DECODE_STREAM_INTERRUPTED"      # 流式响应中途断开
NODE_UNHEALTHY = "NODE_UNHEALTHY"                            # admin：待加节点探测失败
NODE_NOT_FOUND = "NODE_NOT_FOUND"                            # admin：节点不在池中
NODE_EXISTS = "NODE_EXISTS"                                  # admin：节点已在池中


class RouterError(Exception):
    """Router 对外错误的统一载体，由 main.py 的 exception handler 映射为 JSON 响应。"""

    http_status: int = 500
    code: str = "INTERNAL_ERROR"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def to_body(self) -> dict:
        return {"error": {"code": self.code, "message": self.message}}


class NoServingGroup(RouterError):
    http_status = 404
    code = NO_SERVING_GROUP


class InvalidRequest(RouterError):
    http_status = 400
    code = INVALID_REQUEST


class PrefillServiceUnavailable(RouterError):
    http_status = 503
    code = PREFILL_SERVICE_UNAVAILABLE


class DecodeServiceUnavailable(RouterError):
    http_status = 503
    code = DECODE_SERVICE_UNAVAILABLE


class PrefillAtCapacity(RouterError):
    http_status = 429
    code = PREFILL_AT_CAPACITY


class DecodeAtCapacity(RouterError):
    http_status = 429
    code = DECODE_AT_CAPACITY


class _BackendRequestFailed(RouterError):
    """P/D 请求失败的基类。

    node 标记出错节点；unreachable 表示连接级失败（供被动健康摘除——
    后端 5xx 说明节点活着，不能摘）；retryable 标记是否值得换节点重试——
    连接级失败或后端 5xx 可重试（KV 块按哈希内容寻址，重写幂等），
    超时不可重试（客户端已等待过久）。
    """

    def __init__(
        self, message: str, *, node=None, retryable: bool = False, unreachable: bool = False,
    ) -> None:
        super().__init__(message)
        self.node = node
        self.retryable = retryable
        self.unreachable = unreachable


class PrefillRequestFailed(_BackendRequestFailed):
    http_status = 502
    code = PREFILL_REQUEST_FAILED


class DecodeRequestFailed(_BackendRequestFailed):
    http_status = 503
    code = DECODE_REQUEST_FAILED


class NodeUnhealthy(RouterError):
    http_status = 503
    code = NODE_UNHEALTHY


class NodeNotFound(RouterError):
    http_status = 404
    code = NODE_NOT_FOUND


class NodeExists(RouterError):
    http_status = 409
    code = NODE_EXISTS
