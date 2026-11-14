"""端到端测试：真实 TCP 上验证两阶段编排、SSE 透传、就绪等待、失败语义与调度。"""

from __future__ import annotations

import asyncio

import httpx

from tests.conftest import (
    MODEL,
    ThreadServer,
    chat_payload,
    free_port,
    make_engine,
    make_router,
)


async def stats(http: httpx.AsyncClient, server) -> dict:
    return (await http.get(f"{server.url}/stats")).json()


# --- ① 单 P 单 D，sequential 模式（UCM toy proxy 同语义） ----------------------


async def test_sequential_end_to_end_stream(servers, store_dir, http):
    prefill = servers(make_engine(store_dir, "prefill"))
    decode = servers(make_engine(store_dir, "decode"))
    router = servers(make_router([(MODEL, [prefill.url], [decode.url])]))

    payload = chat_payload("one two three four five six seven eight", max_tokens=8)
    async with http.stream("POST", f"{router.url}/v1/chat/completions", json=payload) as resp:
        assert resp.status_code == 200
        assert resp.headers.get("x-request-id")
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = "".join([chunk async for chunk in resp.aiter_text()])

    assert body.rstrip().endswith("data: [DONE]")
    data_chunks = [line for line in body.splitlines()
                   if line.startswith("data: ") and line != "data: [DONE]"]
    assert len(data_chunks) == 8  # max_tokens 被尊重，逐 token 一个 chunk

    p, d = await stats(http, prefill), await stats(http, decode)
    assert p["prefill_requests"] == 1
    assert p["blocks_written"] == 2           # 8 tok / block 4 = 2 块，哈希文件已落盘
    assert d["decode_requests"] == 1
    assert d["blocks_waited"] == 0            # sequential：D 启动时 KV 全部就绪
    assert d["blocks_recomputed"] == 0


async def test_sequential_non_stream_json(servers, store_dir, http):
    prefill = servers(make_engine(store_dir, "prefill"))
    decode = servers(make_engine(store_dir, "decode"))
    router = servers(make_router([(MODEL, [prefill.url], [decode.url])]))

    payload = chat_payload("hello world from the router test", stream=False, max_tokens=4)
    resp = await http.post(f"{router.url}/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/json")
    body = resp.json()
    assert body["choices"][0]["message"]["content"].strip()
    assert resp.headers.get("x-request-id")


# --- ② early 模式：D 尽早收到请求，就绪等待发生在 D 侧 ------------------------


async def test_early_dispatch_d_side_wait(servers, store_dir, http):
    prefill = servers(make_engine(store_dir, "prefill",
                                  prefill_base_ms=400, ready_delay_ms=250))
    decode = servers(make_engine(store_dir, "decode"))
    router = servers(make_router([(MODEL, [prefill.url], [decode.url])],
                                 decode_dispatch="early"))

    payload = chat_payload("one two three four five six seven eight", max_tokens=4)
    async with http.stream("POST", f"{router.url}/v1/chat/completions", json=payload) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = "".join([chunk async for chunk in resp.aiter_text()])
    assert body.rstrip().endswith("data: [DONE]")

    d = await stats(http, decode)
    assert d["blocks_waited"] == 2        # D 到达时 KV 未就绪，逐块等待（前提①生效）
    assert d["blocks_recomputed"] == 0    # 等待在超时内成功，未回退重算


# --- ③ 失败语义：错误码区分 P / D，fail-fast 不重试 ---------------------------


async def test_no_serving_group_returns_404(servers, store_dir, http):
    prefill = servers(make_engine(store_dir, "prefill"))
    decode = servers(make_engine(store_dir, "decode"))
    router = servers(make_router([(MODEL, [prefill.url], [decode.url])]))

    payload = chat_payload("hi", model="unknown-model")
    resp = await http.post(f"{router.url}/v1/chat/completions", json=payload)
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "NO_SERVING_GROUP"


async def test_prefill_pool_empty_returns_503(servers, store_dir, http):
    """启动健康探测即发现 P 不可达 → 池空 → PREFILL_SERVICE_UNAVAILABLE。"""
    dead_prefill_port = free_port()
    decode = servers(make_engine(store_dir, "decode"))
    router = servers(make_router(
        [(MODEL, [f"http://127.0.0.1:{dead_prefill_port}"], [decode.url])]))

    resp = await http.post(f"{router.url}/v1/chat/completions",
                           json=chat_payload("hi", max_tokens=2))
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "PREFILL_SERVICE_UNAVAILABLE"


async def test_decode_pool_empty_returns_503(servers, store_dir, http):
    """D 池空在选路阶段即被拦下（调度先冻结 P/D 分配，再发起请求）。"""
    prefill = servers(make_engine(store_dir, "prefill"))
    dead_decode_port = free_port()
    router = servers(make_router(
        [(MODEL, [prefill.url], [f"http://127.0.0.1:{dead_decode_port}"])]))

    resp = await http.post(f"{router.url}/v1/chat/completions",
                           json=chat_payload("hi", max_tokens=2))
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "DECODE_SERVICE_UNAVAILABLE"
    p = await stats(http, prefill)
    assert p["prefill_requests"] == 0     # 请求根本没打到 P


async def test_prefill_down_midflight_returns_502(servers, store_dir, http):
    """节点启动时健康、请求前挂掉 → 连接失败 → 502 PREFILL_REQUEST_FAILED。"""
    prefill = servers(make_engine(store_dir, "prefill"))
    decode = servers(make_engine(store_dir, "decode"))
    router = servers(make_router([(MODEL, [prefill.url], [decode.url])]))
    prefill.stop()                        # 健康探测间隔被拉长，模拟探测盲区

    resp = await http.post(f"{router.url}/v1/chat/completions",
                           json=chat_payload("hi", max_tokens=2))
    assert resp.status_code == 502
    assert resp.json()["error"]["code"] == "PREFILL_REQUEST_FAILED"


async def test_decode_down_midflight_returns_503(servers, store_dir, http):
    prefill = servers(make_engine(store_dir, "prefill"))
    decode = servers(make_engine(store_dir, "decode"))
    router = servers(make_router([(MODEL, [prefill.url], [decode.url])]))
    decode.stop()

    resp = await http.post(f"{router.url}/v1/chat/completions",
                           json=chat_payload("hi", max_tokens=2))
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "DECODE_REQUEST_FAILED"


# --- ④ 多 P 多 D：每池独立 round-robin ---------------------------------------


async def test_round_robin_two_by_two(servers, store_dir, http):
    p1 = servers(make_engine(store_dir, "prefill"))
    p2 = servers(make_engine(store_dir, "prefill"))
    d1 = servers(make_engine(store_dir, "decode"))
    d2 = servers(make_engine(store_dir, "decode"))
    router = servers(make_router([(MODEL, [p1.url, p2.url], [d1.url, d2.url])]))

    for i in range(4):
        payload = chat_payload(f"request number {i} with some words", stream=False, max_tokens=2)
        resp = await http.post(f"{router.url}/v1/chat/completions", json=payload)
        assert resp.status_code == 200

    p1_stats, p2_stats = await stats(http, p1), await stats(http, p2)
    d1_stats, d2_stats = await stats(http, d1), await stats(http, d2)
    assert p1_stats["prefill_requests"] == 2 and p2_stats["prefill_requests"] == 2
    assert d1_stats["decode_requests"] == 2 and d2_stats["decode_requests"] == 2


# --- ⑤ 健康检查：坏节点移出轮询，恢复后加回 -----------------------------------


async def test_health_checker_removes_dead_decode_node(servers, store_dir, http):
    d_alive = servers(make_engine(store_dir, "decode"))
    d_dead = servers(make_engine(store_dir, "decode"))
    prefill = servers(make_engine(store_dir, "prefill"))
    router = servers(make_router(
        [(MODEL, [prefill.url], [d_alive.url, d_dead.url])],
        health_interval=0.3,
    ))
    d_dead.stop()

    # 等健康循环把死节点移出轮询（router /health 暴露池健康数）
    for _ in range(40):
        pools = (await http.get(f"{router.url}/health")).json()["pools"]
        if pools[0]["decode"]["healthy"] == 1:
            break
        await asyncio.sleep(0.2)
    assert pools[0]["decode"]["healthy"] == 1

    for i in range(3):
        payload = chat_payload(f"health check request {i}", stream=False, max_tokens=2)
        assert (await http.post(f"{router.url}/v1/chat/completions", json=payload)).status_code == 200

    alive_stats = await stats(http, d_alive)
    assert alive_stats["decode_requests"] == 3   # 流量全部落在幸存节点


# --- ⑥ OpenAI 兼容：/v1/completions 同链路 -----------------------------------


async def test_completions_endpoint_passthrough(servers, store_dir, http):
    prefill = servers(make_engine(store_dir, "prefill"))
    decode = servers(make_engine(store_dir, "decode"))
    router = servers(make_router([(MODEL, [prefill.url], [decode.url])]))

    payload = {"model": MODEL, "stream": False, "max_tokens": 4,
               "prompt": "raw completion prompt with words"}
    resp = await http.post(f"{router.url}/v1/completions", json=payload)
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["text"].strip()


# --- ⑦ metrics：自有指标 + PD 分改名合并透传 ---------------------------------


async def test_metrics_own_and_pd_passthrough(servers, store_dir, http):
    prefill = servers(make_engine(store_dir, "prefill"))
    decode = servers(make_engine(store_dir, "decode"))
    router = servers(make_router([(MODEL, [prefill.url], [decode.url])]))

    payload = chat_payload("metrics probe prompt with words", stream=False, max_tokens=2)
    assert (await http.post(f"{router.url}/v1/chat/completions", json=payload)).status_code == 200

    text = (await http.get(f"{router.url}/metrics")).text

    # 自有指标：请求成败、节点选择、在途、池健康、TTFT
    assert 'pd_requests_total{dispatch="sequential",model="mock-model",outcome="ok"} 1' in text
    assert f'pd_node_selections_total{{instance="{prefill.url}",role="prefill"}} 1' in text
    assert f'pd_node_selections_total{{instance="{decode.url}",role="decode"}} 1' in text
    assert f'pd_sessions_active{{instance="{prefill.url}",role="prefill"}} 0' in text
    assert 'pd_pool_healthy{model="mock-model",role="decode"} 1' in text
    assert "pd_ttft_seconds_count{model=" in text

    # PD 分改名合并：同族指标按角色拆成 pd_prefill_* / pd_decode_*，实例靠标签区分
    assert f'pd_prefill_mock_kv_blocks_written_total{{pd_role="prefill",pd_instance="{prefill.url}"}}' in text
    assert f'pd_decode_mock_kv_blocks_hit_total{{pd_role="decode",pd_instance="{decode.url}"}}' in text
    assert "# TYPE pd_prefill_mock_kv_blocks_written_total counter" in text


# --- ⑧ cached_tokens 双口径 --------------------------------------------------


LONG_PROMPT = "alpha beta gamma delta epsilon zeta eta theta"  # 8 tok → 2 块


async def test_cached_tokens_dual_caliber_non_stream(servers, store_dir, http):
    prefill = servers(make_engine(store_dir, "prefill"))
    decode = servers(make_engine(store_dir, "decode"))
    router = servers(make_router([(MODEL, [prefill.url], [decode.url])]))

    payload = chat_payload(LONG_PROMPT, stream=False, max_tokens=2)
    first = (await http.post(f"{router.url}/v1/chat/completions", json=payload)).json()
    assert first["pd"]["prefill"]["cached_tokens"] == 0      # 冷启动：P 无复用

    second = (await http.post(f"{router.url}/v1/chat/completions", json=payload)).json()
    assert second["usage"]["prompt_tokens_details"]["cached_tokens"] == 8  # P 口径（计费）
    assert second["pd"]["decode"]["cached_tokens"] == 8                    # D 原值保留
    assert second["pd"]["prefill"]["node"] == prefill.url
    assert second["pd"]["decode"]["node"] == decode.url


async def test_cached_tokens_stream_header(servers, store_dir, http):
    prefill = servers(make_engine(store_dir, "prefill"))
    decode = servers(make_engine(store_dir, "decode"))
    router = servers(make_router([(MODEL, [prefill.url], [decode.url])]))

    payload = chat_payload(LONG_PROMPT, stream=True, max_tokens=2)
    async with http.stream("POST", f"{router.url}/v1/chat/completions", json=payload) as resp:
        assert resp.status_code == 200
        async for _ in resp.aiter_text():
            pass
    # 第 2 次请求：P 侧块已就绪 → 复用值进响应头（流式不改写 usage chunk）
    async with http.stream("POST", f"{router.url}/v1/chat/completions", json=payload) as resp:
        assert resp.headers["x-pd-prefill-cached-tokens"] == "8"
        assert resp.headers["x-pd-prefill-node"] == prefill.url
        assert resp.headers["x-pd-decode-node"] == decode.url
        async for _ in resp.aiter_text():
            pass


# --- ⑨ admin API：动态增删节点 ------------------------------------------------


async def test_admin_add_remove_nodes(servers, store_dir, http):
    prefill = servers(make_engine(store_dir, "prefill"))
    d1 = servers(make_engine(store_dir, "decode"))
    d2 = servers(make_engine(store_dir, "decode"))
    router = servers(make_router([(MODEL, [prefill.url], [d1.url])]))

    resp = await http.post(f"{router.url}/admin/nodes",
                           json={"model": MODEL, "role": "decode", "url": d2.url})
    assert resp.status_code == 201
    listing = (await http.get(f"{router.url}/admin/nodes")).json()
    assert listing[0]["decode"]["total"] == 2

    # RR 立即覆盖新节点
    for i in range(2):
        payload = chat_payload(f"admin rr {i}", stream=False, max_tokens=2)
        assert (await http.post(f"{router.url}/v1/chat/completions", json=payload)).status_code == 200
    assert (await stats(http, d2))["decode_requests"] >= 1

    # 重复添加 → 409；加未就绪节点 → 202 进等待队列；未知模型 → 404；坏 role → 400
    resp = await http.post(f"{router.url}/admin/nodes",
                           json={"model": MODEL, "role": "decode", "url": d2.url})
    assert resp.status_code == 409 and resp.json()["error"]["code"] == "NODE_EXISTS"
    dead = free_port()
    resp = await http.post(f"{router.url}/admin/nodes",
                           json={"model": MODEL, "role": "decode", "url": f"http://127.0.0.1:{dead}"})
    assert resp.status_code == 202 and resp.json()["waiting"] is True
    resp = await http.delete(f"{router.url}/admin/nodes", params={
        "model": MODEL, "role": "decode", "url": f"http://127.0.0.1:{dead}"})
    assert resp.status_code == 200 and resp.json()["removed"] is True
    resp = await http.post(f"{router.url}/admin/nodes",
                           json={"model": "no-such", "role": "decode", "url": d2.url})
    assert resp.status_code == 404
    resp = await http.post(f"{router.url}/admin/nodes",
                           json={"model": MODEL, "role": "worker", "url": d2.url})
    assert resp.status_code == 400

    # 排空删除：无在途 → 立即移除；再删 → 404
    resp = await http.delete(f"{router.url}/admin/nodes", params={
        "model": MODEL, "role": "decode", "url": d2.url, "drain": "true"})
    assert resp.status_code == 200 and resp.json()["removed"] is True
    listing = (await http.get(f"{router.url}/admin/nodes")).json()
    assert listing[0]["decode"]["total"] == 1
    resp = await http.delete(f"{router.url}/admin/nodes", params={
        "model": MODEL, "role": "decode", "url": d2.url})
    assert resp.status_code == 404 and resp.json()["error"]["code"] == "NODE_NOT_FOUND"


async def test_admin_drain_waits_for_inflight(servers, store_dir, http):
    """drain=true 时节点停止接新请求，在途清零后才真正移除。"""
    prefill = servers(make_engine(store_dir, "prefill"))
    d1 = servers(make_engine(store_dir, "decode"))
    d2 = servers(make_engine(store_dir, "decode"))
    router = servers(make_router([(MODEL, [prefill.url], [d1.url, d2.url])]))

    resp = await http.delete(f"{router.url}/admin/nodes", params={
        "model": MODEL, "role": "decode", "url": d1.url, "drain": "true"})
    assert resp.status_code == 200 and resp.json()["removed"] is True  # 无在途 → 立即移除

    payload = chat_payload("post drain request", stream=False, max_tokens=2)
    assert (await http.post(f"{router.url}/v1/chat/completions", json=payload)).status_code == 200
    assert (await stats(http, d2))["decode_requests"] == 1


# --- ⑩ 容错增强 ---------------------------------------------------------------


async def test_prefill_failover_to_other_node(servers, store_dir, http):
    """P 连接失败 → 同池换节点重试一次（KV 哈希内容寻址，重试幂等）。"""
    p1 = servers(make_engine(store_dir, "prefill"))
    p2 = servers(make_engine(store_dir, "prefill"))
    decode = servers(make_engine(store_dir, "decode"))
    router = servers(make_router([(MODEL, [p1.url, p2.url], [decode.url])]))
    p1.stop()  # RR 首选 p1，连接失败后应 failover 到 p2

    payload = chat_payload("failover me please now", stream=False, max_tokens=2)
    resp = await http.post(f"{router.url}/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    assert resp.headers["x-pd-prefill-node"] == p2.url
    assert (await stats(http, p2))["prefill_requests"] == 1

    text = (await http.get(f"{router.url}/metrics")).text
    assert 'pd_retries_total{role="prefill"} 1' in text


async def test_passive_health_marks_unreachable_node(servers, store_dir, http):
    """请求时连接失败 → 立即被动摘除（不等探测周期），后续请求改为池空 503。"""
    prefill = servers(make_engine(store_dir, "prefill"))
    decode = servers(make_engine(store_dir, "decode"))
    router = servers(make_router([(MODEL, [prefill.url], [decode.url])]))
    payload = chat_payload("first ok", stream=False, max_tokens=2)
    assert (await http.post(f"{router.url}/v1/chat/completions", json=payload)).status_code == 200

    decode.stop()
    payload = chat_payload("second fails", stream=False, max_tokens=2)
    resp = await http.post(f"{router.url}/v1/chat/completions", json=payload)
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "DECODE_REQUEST_FAILED"

    # 被动摘除生效（health interval=3600，说明不是周期探测干的）
    pools = (await http.get(f"{router.url}/health")).json()["pools"]
    assert pools[0]["decode"]["healthy"] == 0

    resp = await http.post(f"{router.url}/v1/chat/completions",
                           json=chat_payload("third", stream=False, max_tokens=2))
    assert resp.json()["error"]["code"] == "DECODE_SERVICE_UNAVAILABLE"


async def test_inflight_limit_returns_429(servers, store_dir, http):
    """每节点在途上限=1 时，并发第二个请求被 429 拒绝而非排队。"""
    prefill = servers(make_engine(store_dir, "prefill"))
    decode = servers(make_engine(store_dir, "decode"))
    router = servers(make_router([(MODEL, [prefill.url], [decode.url])],
                                 max_inflight_per_node=1))

    async def send():
        payload = chat_payload("limit probe words here", stream=True, max_tokens=15)
        async with http.stream("POST", f"{router.url}/v1/chat/completions", json=payload) as resp:
            if resp.status_code == 200:
                async for _ in resp.aiter_text():
                    pass
                return resp.status_code, None
            return resp.status_code, (await resp.aread())

    results = await asyncio.gather(send(), send())
    codes = {status for status, _ in results}
    assert codes == {200, 429}
    for status, body in results:
        if status == 429:
            assert body and b"PREFILL_AT_CAPACITY" in body


async def test_mid_stream_interrupt_is_recorded(servers, store_dir, http):
    """D 流中途断开：客户端收到截断流（无 [DONE]），router 记录失败指标。"""
    from dataclasses import replace as dc_replace
    from tests.mock_ucm_engine import EngineSettings, build_app

    prefill = servers(make_engine(store_dir, "prefill"))
    decode_settings = dc_replace(EngineSettings(role="decode", model=MODEL, store=store_dir,
                                                decode_token_delay_ms=200))
    decode = servers(ThreadServer(build_app(decode_settings), graceful_shutdown=1.0))
    router = servers(make_router([(MODEL, [prefill.url], [decode.url])]))

    payload = chat_payload("stream will die mid way", stream=True, max_tokens=30)  # 约 6s 流
    chunks: list[str] = []
    interrupted = False
    async with http.stream("POST", f"{router.url}/v1/chat/completions", json=payload) as resp:
        assert resp.status_code == 200
        reader = resp.aiter_text()
        chunks.append(await reader.__anext__())   # 流已建立并吐出第一个 chunk
        decode.stop()                             # 服务端在流中途强制断开（优雅期 1s）
        try:
            async for text in reader:
                chunks.append(text)
        except httpx.HTTPError:
            interrupted = True
    assert interrupted
    assert "[DONE]" not in "".join(chunks)

    text = (await http.get(f"{router.url}/metrics")).text
    assert 'pd_failures_total{code="DECODE_STREAM_INTERRUPTED"} 1' in text


# --- ⑪ 对标 vllm-ascend toy proxy 吸收的行为 ---------------------------------


def test_prefill_rewrite_forces_single_token():
    """prefill 改写必须封死所有产出口径（min_tokens / max_completion_tokens）。"""
    from ucm_router.ports.engine_client import rewrite_for_prefill

    body = rewrite_for_prefill({
        "model": "m", "stream": True, "max_tokens": 16, "min_tokens": 4,
        "max_completion_tokens": 32, "stream_options": {"include_usage": True},
    })
    assert body["stream"] is False
    assert body["max_tokens"] == 1
    assert body["min_tokens"] == 1                 # 否则 min_tokens=4 与 max_tokens=1 冲突
    assert body["max_completion_tokens"] == 1      # OpenAI 新口径，引擎可能优先用它
    assert "stream_options" not in body


async def test_session_affinity_sticky(servers, store_dir, http):
    """同一会话（X-Session-Id）稳定落到同一组 P/D；不同会话互不影响。"""
    p1 = servers(make_engine(store_dir, "prefill"))
    p2 = servers(make_engine(store_dir, "prefill"))
    d1 = servers(make_engine(store_dir, "decode"))
    d2 = servers(make_engine(store_dir, "decode"))
    router = servers(make_router([(MODEL, [p1.url, p2.url], [d1.url, d2.url])],
                                 session_affinity=True))

    async def send(session_id: str, tag: str) -> tuple[str, str]:
        payload = chat_payload(f"{tag} turn with words", stream=False, max_tokens=2)
        resp = await http.post(f"{router.url}/v1/chat/completions",
                               json=payload, headers={"X-Session-Id": session_id})
        assert resp.status_code == 200
        return resp.headers["x-pd-prefill-node"], resp.headers["x-pd-decode-node"]

    s1 = await asyncio.gather(*(send("sess-1", f"turn {i}") for i in range(3)))
    s2 = await asyncio.gather(*(send("sess-2", f"turn {i}") for i in range(3)))

    assert len({p for p, _ in s1}) == 1 and len({d for _, d in s1}) == 1
    assert len({p for p, _ in s2}) == 1 and len({d for _, d in s2}) == 1


async def test_admin_add_waits_for_readiness(servers, store_dir, http):
    """探测未通过的节点进 waiting 队列，就绪后自动激活（ascend waiting_nodes 语义）。"""
    prefill = servers(make_engine(store_dir, "prefill"))
    d1 = servers(make_engine(store_dir, "decode"))
    d2 = make_engine(store_dir, "decode")          # 先不启动，端口已保留
    router = servers(make_router([(MODEL, [prefill.url], [d1.url])],
                                 health_interval=0.3))

    resp = await http.post(f"{router.url}/admin/nodes",
                           json={"model": MODEL, "role": "decode", "url": d2.url})
    assert resp.status_code == 202
    assert resp.json()["waiting"] is True

    servers(d2)                                     # 现在启动同一实例
    for _ in range(30):
        listing = (await http.get(f"{router.url}/admin/nodes")).json()
        if listing[0]["decode"]["total"] == 2 and not listing[0]["decode"]["waiting"]:
            break
        await asyncio.sleep(0.3)
    assert listing[0]["decode"]["total"] == 2
    assert listing[0]["decode"]["waiting"] == []

    # 等待中的节点也可以直接删除
    d3 = make_engine(store_dir, "decode")
    resp = await http.post(f"{router.url}/admin/nodes",
                           json={"model": MODEL, "role": "decode", "url": d3.url})
    assert resp.status_code == 202
    resp = await http.delete(f"{router.url}/admin/nodes", params={
        "model": MODEL, "role": "decode", "url": d3.url})
    assert resp.status_code == 200 and resp.json()["removed"] is True


async def test_decode_fail_before_first_chunk_returns_real_status(servers, store_dir, http):
    """preopen：提交 200 头之前 D 就死掉 → 客户端拿到真实 503，而不是空 200 截断流。"""
    prefill = servers(make_engine(store_dir, "prefill"))
    from dataclasses import replace as dc_replace
    from tests.mock_ucm_engine import EngineSettings, build_app

    decode_settings = dc_replace(EngineSettings(role="decode", model=MODEL, store=store_dir,
                                                first_token_delay_ms=8000))
    decode = servers(ThreadServer(build_app(decode_settings), graceful_shutdown=0.5))
    router = servers(make_router([(MODEL, [prefill.url], [decode.url])]))

    async def send():
        payload = chat_payload("preopen probe words here", stream=True, max_tokens=5)
        async with http.stream("POST", f"{router.url}/v1/chat/completions", json=payload) as resp:
            if resp.status_code == 200:
                async for chunk in resp.aiter_bytes():
                    pass
                return resp.status_code, b""
            return resp.status_code, await resp.aread()

    task = asyncio.create_task(send())
    await asyncio.sleep(2)      # prefill 已完成、D 已连接、router 正阻塞在首 chunk（8s 后才来）
    decode.stop()               # 服务端强制断开 → preopen 失败
    status, body = await asyncio.wait_for(task, timeout=15)
    assert status == 503
    assert b"DECODE_REQUEST_FAILED" in body


async def test_recomputed_detection(servers, store_dir, http):
    """D 侧重算兜底（stop_reason=recomputed）被计数与标记——前提①③失败率的观测点。"""
    prefill = servers(make_engine(store_dir, "prefill"))
    decode = servers(make_engine(store_dir, "decode", recompute_stop_reason=True))
    router = servers(make_router([(MODEL, [prefill.url], [decode.url])]))

    payload = chat_payload("recomputed probe words here", stream=False, max_tokens=2)
    body = (await http.post(f"{router.url}/v1/chat/completions", json=payload)).json()
    assert body["pd"]["decode"]["recomputed"] is True

    payload = chat_payload("recomputed probe words here", stream=True, max_tokens=2)
    async with http.stream("POST", f"{router.url}/v1/chat/completions", json=payload) as resp:
        assert resp.status_code == 200
        async for _ in resp.aiter_text():
            pass

    text = (await http.get(f"{router.url}/metrics")).text
    assert 'pd_recomputed_total{model="mock-model"} 2' in text


async def test_v1_models_and_reset_prefix_cache(servers, store_dir, http):
    prefill = servers(make_engine(store_dir, "prefill"))
    decode = servers(make_engine(store_dir, "decode"))
    router = servers(make_router([(MODEL, [prefill.url], [decode.url])]))

    resp = await http.get(f"{router.url}/v1/models")
    assert resp.status_code == 200
    assert MODEL in resp.json()["data"][0]["id"]

    resp = await http.post(f"{router.url}/reset_prefix_cache")
    assert resp.status_code == 200
    body = resp.json()
    assert body["reset"] == 2 and body["failed"] == []


