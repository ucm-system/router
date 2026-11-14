"""模拟引擎：扮演 vLLM + UCM Connector 的 P / D 角色（仅供 router 验证使用）。

模拟口径来自设计定稿的三条事实：
1. KVCache 以哈希值做文件名，落在共享文件系统上（P/D 通过 --store 指向同一目录）；
2. 去中心化，无 master server（本文件不含任何中心化组件）；
3. Prefetch 无独立 RPC，由 Decode 请求到达触发 D 侧逐块 lookup。

并模拟 v4 依赖的 D 侧就绪等待（待验证前提①③的 mock 版）：
- P 写 KV：先写数据文件，延迟 --ready-delay-ms 后补 .ready 标记，
  模拟"哈希文件可见 ≠ KV 可完整读取"的时间差；
- D 逐块检查 .ready，未就绪则轮询等待（Connector 就绪等待能力），
  等待超时则回退本地重算（兜底路径，优化路径失效不失败）。

用法：
    python tests/mock_ucm_engine.py --role prefill --port 7810 --store DIR --model M
    python tests/mock_ucm_engine.py --role decode --port 7820 --store DIR --model M

/tokens 相关数值都是示意，未校准；本引擎不追求推理正确性，只让 router 的
两阶段编排、SSE 透传、TTFT 差异、就绪等待可被端到端观察。
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response, StreamingResponse

POLL_INTERVAL_SECONDS = 0.02
WORD_BANK = (
    "the quick brown fox jumps over the lazy dog while stars drift "
    "across the observable universe in silence".split()
)


@dataclass
class EngineSettings:
    role: str                      # "prefill" | "decode"
    model: str
    store: Path
    block_tokens: int = 4          # mock block 大小（token 数），故意取小便于观察
    prefill_base_ms: int = 200     # P：prefill 固定开销
    prefill_per_token_ms: int = 3  # P：每 token 计算开销
    ready_delay_ms: int = 150      # P：数据文件可见到 .ready 标记出现的时间差
    kv_wait_timeout_ms: int = 5000     # D：单块就绪等待上限（前提①的就绪等待）
    recompute_base_ms: int = 300       # D：就绪等待失败后本地重算的兜底开销
    decode_token_delay_ms: int = 20    # D：每个输出 token 间隔
    first_token_delay_ms: int = 0      # D：首个 chunk 前的延迟（测 preopen 用）
    recompute_stop_reason: bool = False  # D：模拟 vLLM 的 stop_reason=recomputed（KV 等待回退重算）
    max_output_tokens: int = 64        # D：单请求输出 token 上限


@dataclass
class Stats:
    prefill_requests: int = 0
    decode_requests: int = 0
    blocks_written: int = 0
    blocks_hit: int = 0        # 首次检查即就绪
    blocks_waited: int = 0     # 经轮询等待后才就绪（D 侧 Connector 等待生效）
    blocks_recomputed: int = 0  # 等待超时，回退本地重算
    last_ttft_ms: float = 0.0


class SharedStore:
    """KV 共享存储的 mock：目录 + 哈希文件名 + ready 标记。无 master，纯文件系统。"""

    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def data_path(self, block_hash: str) -> Path:
        return self.root / f"{block_hash}.kv"

    def ready_path(self, block_hash: str) -> Path:
        return self.root / f"{block_hash}.kv.ready"

    def is_ready(self, block_hash: str) -> bool:
        return self.ready_path(block_hash).exists()

    async def write_block(self, block_hash: str, ready_delay_seconds: float) -> None:
        """先写数据文件，延迟后补 .ready——制造"可见但未就绪"窗口（前提③）。"""
        self.data_path(block_hash).write_bytes(f"kv:{block_hash}".encode())
        await asyncio.sleep(ready_delay_seconds)
        self.ready_path(block_hash).write_bytes(b"")

    async def wait_ready(self, block_hash: str, timeout_seconds: float) -> bool:
        """D 侧 Connector 就绪等待：逐块轮询，直到就绪或超时。"""
        deadline = time.monotonic() + timeout_seconds
        while not self.is_ready(block_hash):
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
        return True


def block_hashes(model: str, prompt: str, block_tokens: int) -> list[str]:
    """mock 的块哈希：前缀链式 sha1。这是引擎/UCM 侧知识，Router 不持有。"""
    tokens = prompt.split() if prompt.strip() else ["<empty>"]
    namespace = f"ucm://{model}"
    hashes: list[str] = []
    prev = hashlib.sha1(namespace.encode()).hexdigest()
    for i in range(0, len(tokens), block_tokens):
        chunk = " ".join(tokens[i : i + block_tokens])
        prev = hashlib.sha1(f"{prev}|{i}|{chunk}".encode()).hexdigest()
        hashes.append(prev)
    return hashes


def extract_prompt(payload: dict, chat: bool) -> str:
    if chat:
        messages = payload.get("messages", [])
        return " ".join(
            m.get("content", "") for m in messages
            if isinstance(m, dict) and isinstance(m.get("content"), str)
        )
    prompt = payload.get("prompt", "")
    if isinstance(prompt, list):
        prompt = " ".join(str(p) for p in prompt)
    return str(prompt)


def output_tokens(max_tokens: int | None) -> list[str]:
    n = max_tokens if isinstance(max_tokens, int) and max_tokens > 0 else 16
    n = min(n, 256)
    return [WORD_BANK[i % len(WORD_BANK)] for i in range(n)]


def build_app(settings: EngineSettings) -> FastAPI:
    store = SharedStore(settings.store)
    stats = Stats()
    app = FastAPI(title=f"mock-{settings.role}-engine")

    # --- 管理端点 -------------------------------------------------------
    started_at = time.monotonic()

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "role": settings.role}

    @app.get("/stats")
    async def get_stats() -> dict:
        return {
            "role": settings.role,
            "model": settings.model,
            **{k: v for k, v in stats.__dict__.items()},
        }

    @app.get("/metrics")
    async def prometheus_metrics() -> Response:
        """极简 Prometheus exposition，供 router 的 /metrics 改名合并透传。"""
        lines = [
            "# HELP mock_kv_blocks_written_total KV block files written by this mock engine",
            "# TYPE mock_kv_blocks_written_total counter",
            f"mock_kv_blocks_written_total {stats.blocks_written}",
            "# HELP mock_kv_blocks_hit_total KV block lookups ready at first check",
            "# TYPE mock_kv_blocks_hit_total counter",
            f"mock_kv_blocks_hit_total {stats.blocks_hit}",
            "# HELP mock_process_uptime_seconds mock engine uptime",
            "# TYPE mock_process_uptime_seconds gauge",
            f"mock_process_uptime_seconds {time.monotonic() - started_at:.3f}",
        ]
        return Response(
            content="\n".join(lines) + "\n",
            media_type="text/plain; version=0.0.4",
        )

    @app.get("/v1/models")
    async def list_models() -> dict:
        return {
            "object": "list",
            "data": [{"id": settings.model, "object": "model", "owned_by": "mock"}],
        }

    @app.post("/reset_prefix_cache")
    async def reset_prefix_cache() -> Response:
        # mock：只应答不清缓存（共享存储的块是"别处写的"，语义上不属于本实例）
        return Response(status_code=200)

    # --- OpenAI 兼容端点 -------------------------------------------------
    @app.post("/v1/chat/completions")
    async def chat_completions(payload: dict) -> Response:
        return await generate(payload, chat=True)

    @app.post("/v1/completions")
    async def completions(payload: dict) -> Response:
        return await generate(payload, chat=False)

    async def generate(payload: dict, chat: bool) -> Response:
        prompt = extract_prompt(payload, chat)
        hashes = block_hashes(settings.model, prompt, settings.block_tokens)
        if settings.role == "prefill":
            body = await prefill(prompt, hashes)
        else:
            body = await decode(payload, prompt, hashes, chat)
        return body

    async def prefill(prompt: str, hashes: list[str]) -> JSONResponse:
        """P 角色：计算 prefill → KV 块落盘 → 返回单 token。

        cached_tokens 口径：写入前就已 ready 的块数（模拟 P 侧前缀缓存复用）。
        """
        stats.prefill_requests += 1
        token_count = max(1, len(prompt.split()))
        cost = (settings.prefill_base_ms + token_count * settings.prefill_per_token_ms) / 1000
        await asyncio.sleep(cost)
        pre_cached_blocks = sum(1 for h in hashes if store.is_ready(h))
        for h in hashes:
            if not store.is_ready(h):
                stats.blocks_written += 1
                await store.write_block(h, settings.ready_delay_ms / 1000)
        return JSONResponse({
            "id": f"chatcmpl-mock-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "model": settings.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": WORD_BANK[0]},
                "finish_reason": "length",
            }],
            "usage": {
                "prompt_tokens": token_count,
                "completion_tokens": 1,
                "prompt_tokens_details": {
                    "cached_tokens": pre_cached_blocks * settings.block_tokens,
                },
            },
        })

    async def decode(payload: dict, prompt: str, hashes: list[str], chat: bool) -> Response:
        """D 角色：Connector 逐块 lookup → 就绪等待（超时则重算兜底）→ 生成。

        cached_tokens 口径：首次检查即就绪的块数（D 本地/存储命中，未等待）。
        """
        stats.decode_requests += 1
        started = time.monotonic()
        waited = recomputed = hit_count = 0
        for h in hashes:
            ready_at_first_check = store.is_ready(h)
            ok = await store.wait_ready(h, settings.kv_wait_timeout_ms / 1000)
            if ready_at_first_check:
                hit_count += 1
                stats.blocks_hit += 1
            elif ok:
                waited += 1
                stats.blocks_waited += 1
            else:
                recomputed += 1
                stats.blocks_recomputed += 1
                await asyncio.sleep(settings.recompute_base_ms / 1000)
        first_delay = time.monotonic() - started
        stats.last_ttft_ms = first_delay * 1000
        print(
            f"[mock-decode] prompt={len(prompt.split())}tok blocks={len(hashes)} "
            f"hit={hit_count} wait={waited} "
            f"recompute={recomputed} lookup_time={first_delay * 1000:.0f}ms",
            flush=True,
        )

        tokens = output_tokens(payload.get("max_tokens"))
        stop_reason = "recomputed" if settings.recompute_stop_reason else None

        if payload.get("stream"):
            return StreamingResponse(
                sse_chunks(payload, tokens, chat, stop_reason=stop_reason),
                media_type="text/event-stream",
            )
        return JSONResponse(
            non_stream_body(payload, tokens, chat, finish="stop",
                            cached_tokens=hit_count * settings.block_tokens,
                            stop_reason=stop_reason)
        )

    async def sse_chunks(payload: dict, tokens: list[str], chat: bool, stop_reason: str | None = None):
        """OpenAI 流式格式：逐 token 一个 chunk，末尾 [DONE]。"""
        if settings.first_token_delay_ms > 0:
            await asyncio.sleep(settings.first_token_delay_ms / 1000)
        completion_id = f"chatcmpl-mock-{uuid.uuid4().hex[:8]}"
        created = int(time.time())
        for i, token in enumerate(tokens):
            last = i == len(tokens) - 1
            if chat:
                delta = {"content": f"{token} "} if not last else {}
                choice = {"index": 0, "delta": delta,
                          "finish_reason": "stop" if last else None}
            else:
                choice = {"index": 0, "text": f"{token} " if not last else "",
                          "finish_reason": "stop" if last else None}
            if last and stop_reason:
                choice["stop_reason"] = stop_reason
            chunk = {
                "id": completion_id, "object": "chat.completion.chunk" if chat else "text_completion.chunk",
                "created": created, "model": payload.get("model", settings.model),
                "choices": [choice],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode()
            await asyncio.sleep(settings.decode_token_delay_ms / 1000)
        yield b"data: [DONE]\n\n"

    def non_stream_body(
        payload: dict, tokens: list[str], chat: bool, finish: str,
        cached_tokens: int = 0, stop_reason: str | None = None,
    ) -> dict:
        text = " ".join(tokens) + " "
        base = {
            "id": f"chatcmpl-mock-{uuid.uuid4().hex[:8]}",
            "created": int(time.time()),
            "model": payload.get("model", settings.model),
            "usage": {
                "prompt_tokens": len(prompt_tokens(payload, chat)),
                "completion_tokens": len(tokens),
                "prompt_tokens_details": {"cached_tokens": cached_tokens},
            },
        }
        if chat:
            choice = {"index": 0,
                      "message": {"role": "assistant", "content": text},
                      "finish_reason": finish}
            base["object"] = "chat.completion"
        else:
            choice = {"index": 0, "text": text, "finish_reason": finish}
            base["object"] = "text_completion"
        if stop_reason:
            choice["stop_reason"] = stop_reason
        base["choices"] = [choice]
        return base

    def prompt_tokens(payload: dict, chat: bool) -> list[str]:
        return extract_prompt(payload, chat).split() or ["<empty>"]

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="mock vLLM+UCM engine (prefill/decode)")
    parser.add_argument("--role", choices=("prefill", "decode"), required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--store", required=True, help="shared KV store directory")
    parser.add_argument("--model", default="mock-model")
    parser.add_argument("--block-tokens", type=int, default=4)
    parser.add_argument("--prefill-base-ms", type=int, default=200)
    parser.add_argument("--prefill-per-token-ms", type=int, default=3)
    parser.add_argument("--ready-delay-ms", type=int, default=150)
    parser.add_argument("--kv-wait-timeout-ms", type=int, default=5000)
    parser.add_argument("--recompute-base-ms", type=int, default=300)
    parser.add_argument("--decode-token-delay-ms", type=int, default=20)
    parser.add_argument("--first-token-delay-ms", type=int, default=0)
    parser.add_argument("--recompute-stop-reason", action="store_true",
                        help="decode 模拟 vLLM stop_reason=recomputed（KV 等待回退重算）")
    return parser.parse_args()


def main() -> None:
    import uvicorn

    args = parse_args()
    settings = EngineSettings(
        role=args.role,
        model=args.model,
        store=Path(args.store),
        block_tokens=args.block_tokens,
        prefill_base_ms=args.prefill_base_ms,
        prefill_per_token_ms=args.prefill_per_token_ms,
        ready_delay_ms=args.ready_delay_ms,
        kv_wait_timeout_ms=args.kv_wait_timeout_ms,
        recompute_base_ms=args.recompute_base_ms,
        decode_token_delay_ms=args.decode_token_delay_ms,
        first_token_delay_ms=args.first_token_delay_ms,
        recompute_stop_reason=args.recompute_stop_reason,
    )
    print(f"[mock-{args.role}] serving on {args.host}:{args.port}, store={args.store}", flush=True)
    uvicorn.run(build_app(settings), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
