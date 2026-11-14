# ucm-router

UCM（[Unified Cache Manager](https://github.com/ModelEngine-Group/unified-cache-management)）PD 分离部署下**最基础的 LLM Router**：对客户端暴露 OpenAI 兼容接口，对内维护 Prefill（P）池与 Decode（D）池，编排一次请求的 PD 两段调用。

设计定稿见[《UCM PD Router 设计五易其稿》](https://keriko-blog.pages.dev/posts/037-ucm-pd-router-evolution)（v4）。三句话分工：

> **Router 决定请求去哪里，UCM 决定 KV 如何准备好，Engine 决定什么时候执行计算。**

## v4 边界（本仓库的实现范围）

Router 只做两件事：

1. **节点选择**：按 ServingGroup 匹配模型 → P、D 池各自独立 Round-Robin（健康过滤）
2. **请求生命周期**：生成 `X-Request-Id` 贯穿 P/D 两段调用 → 提交 prefill → 提交 decode → 字节级透传响应 → 清理

Router **完全不感知** KVCache：不持有 Tokenizer / Block Hash 逻辑，不做缓存查询，不编排预取。命中判断、SSD→DRAM 提升、等待 P 未写入的 KV，全部是 D 侧 UCM Connector 的职责。进程内只有 **Routing** 与 **Request** 两个领域、三个 Port（NodeProvider / EngineClient / SessionRepository）。

同样**不包含**（留作后续演进）：K8s 服务发现、prefix/缓存感知调度、重试/熔断、鉴权、metrics 上报。失败语义是最基础的 fail-fast，不重试。

## 架构

```
                        ┌─────────────────────────────────────────┐
 Client ──── SSE ──────▶│  Router (FastAPI + httpx)               │
 (OpenAI SDK)           │  ├─ Scheduler: P池RR / D池RR(健康过滤)   │
                        │  ├─ Orchestrator: 两阶段调用             │
                        │  └─ HealthChecker: 周期 GET /health     │
                        └───────────┬─────────────────┬───────────┘
                             ①请求 │                 │ ②请求
                                   ▼                 ▼
                            P instances        D instances
                          (vLLM + UCM        (vLLM + UCM
                           Connector          Connector
                           kv_both)           kv_both)
                                   │                 ▲
                                   ▼     lookup/wait  │
                            ┌──────────────────────────┐
                            │  共享 KV 存储池 (UCM store)│
                            └──────────────────────────┘
```

P 与 D 之间没有直接网络传输：P 把 KV 块按哈希文件名落盘到共享存储（去中心化，无 master），D 收到请求后由 UCM Connector 逐块 lookup，就绪等待后放行 decode。

## 请求生命周期（`orchestrator.py`）

```
Client          Router                          P                     D
  │  POST          │                             │                     │
  │───────────────▶│ 选P、选D（冻结 RouteAssignment）│                    │
  │                │── 改写请求: stream=False,    │                    │
  │                │   max_tokens=1 ────────────▶│ prefill + KV 落盘   │
  │                │◀─ 200 (sequential 模式到此) ─│                     │
  │                │── 原始请求 ──────────────────────────────────────▶│ Connector lookup/wait
  │◀━━ SSE 字节透传 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━│
```

decode 请求**原样透传**，不带任何缓存/传输参数（UCM 靠块哈希前缀发现 KV，不需要）。

## 容错语义

| 场景 | 行为 |
| --- | --- |
| 选路时池空（无健康节点） | 503 `PREFILL/DECODE_SERVICE_UNAVAILABLE`，请求不发出 |
| 全部健康节点在途到上限 | 429 `PREFILL/DECODE_AT_CAPACITY`（客户端退避重试） |
| P 连接失败 / 后端 5xx | **同池换节点重试一次**（KV 块哈希内容寻址，重复 prefill 幂等）；仍失败 → 502 `PREFILL_REQUEST_FAILED` |
| D 连接失败 / 后端 5xx | 同上换 D 重试一次；仍失败 → 503 `DECODE_REQUEST_FAILED` |
| 请求超时 | 不重试（客户端已等待过久） |
| 连接级失败 | **被动健康**：立即把节点移出轮询，不等探测周期 |
| D 首字节前失败 | **preopen**（提交 200 头前先读 D 首个 chunk）→ 客户端拿到真实 503，而非空 200 截断流 |
| 流中失败（首字节后） | 客户端看到截断流（无 `[DONE]`）；router 记 `DECODE_STREAM_INTERRUPTED` 失败指标 |
| D 侧等不到 KV 自行重算 | 引擎报 `stop_reason=recomputed`；router 计 `pd_recomputed_total` 指标（**博文前提①③失败率的观测点**），不编排重试——重算兜底属引擎/UCM 职责 |

说明：UCM 集中式模式下 KV 块按哈希做文件名，重试 prefill 会写出一模一样的块文件，因此换节点重试是幂等安全的。early 模式下 P 失败仍会中止已建立的 D 请求。

## 会话亲和（可选，默认关）

`routing.session_affinity: true` 后，同一会话稳定落到同一组 P/D（提升多轮对话的 KV 复用）：

- key 派生优先级：请求 `user` 字段 → `X-Session-Id` 请求头 → `model + 前置 system/developer 消息 + 第一条 user 消息`（设计来自 vllm-ascend toy proxy 的 `conversation_affinity_key`，不哈希持续增长的历史）
- 算法 Rendezvous (HRW) 哈希：跨进程/重启稳定，仅在节点移除时重映射恰好落在其上的会话
- **确定性放置，不感知 KV 状态**——不需要 Tokenizer/块哈希，在 v4 边界内；真正的缓存命中判断仍在 UCM Connector

## cached_tokens 双口径

PD 分离下客户端只能看到 D 的响应，而 prefill 侧复用发生在 P。Router 的处理：

- **非流式**：响应 JSON 中 `usage.prompt_tokens_details.cached_tokens` 用 **P 的值**（计费口径：prefill 侧复用），D 的原值保留在 `pd.decode.cached_tokens`；同时附 `pd.prefill/decode` 两个节点的信息
- **流式**：SSE chunk 保持字节级透传不改写，P 的 cached_tokens 放在响应头 `X-PD-Prefill-Cached-Tokens`（两种模式 router 都在 P 完成后才返回响应，头部发出前已知）
- 注意：UCM 外部加载的块数（"到底省了多少重算"）属于 UCM/引擎侧观测，Router 不感知（v4 边界）；该字段仅作账务/透传，**不参与任何路由决策**

## Metrics

`GET /metrics`（Prometheus 文本格式）合并输出两部分：

**Router 自有指标**（只报选路与生命周期，不报 KV 状态——那是 UCM 的事）：

| 指标 | 说明 |
| --- | --- |
| `pd_requests_total{model,dispatch,outcome}` | 请求量与成败 |
| `pd_failures_total{code}` | 失败细分（含 `DECODE_STREAM_INTERRUPTED`） |
| `pd_node_selections_total{role,instance}` | RR 分布公平性 |
| `pd_retries_total{role}` | 换节点 failover 次数 |
| `pd_recomputed_total{model}` | D 侧 KV 等待回退重算次数（vLLM `stop_reason=recomputed`） |
| `pd_prefill_duration_seconds_{sum,count}{model}` | P 段耗时 |
| `pd_ttft_seconds_{sum,count}{model}` | 客户端视角 TTFT（含 prefill） |
| `pd_early_dispatch_lead_seconds_{sum,count}{model}` | early 模式：decode 先于 prefill 完成下发的时长（等待被 P 计算掩盖的量） |
| `pd_sessions_active{role,instance}` | 节点在途请求 |
| `pd_pool_healthy{model,role}` | 池健康数 |
| `pd_stream_bytes_total{model}` | 透传字节数 |
| `pd_scrape_failures_total{role,instance}` | 抓取实例 /metrics 失败次数 |

**PD 分透传**：并发抓取每个实例的 `/metrics`，metric family 改名加 `pd_{role}_` 前缀并注入 `pd_role` / `pd_instance` 标签后合并（不改名的话 P、D 同名 family 会让 Prometheus 拒绝该端点）。

## 动态增删节点（admin API）

```bash
# 添加：健康则立即入池（201）；探测未通过则进等待队列（202），
# 由健康检查按 interval 轮询，就绪自动激活，超过 waiting_max_retries 轮放弃。
# 不持久化，重启以 YAML 为准。
curl -X POST http://127.0.0.1:9000/admin/nodes \
  -d '{"model":"Qwen/Qwen2.5-7B-Instruct","role":"decode","url":"http://10.0.0.5:7801"}'

# 查询（含每节点 healthy/draining/inflight 与 waiting 列表）
curl http://127.0.0.1:9000/admin/nodes

# 删除：drain=true 先停止接新请求，在途清零后自动移除
curl -X DELETE "http://127.0.0.1:9000/admin/nodes?model=Qwen/Qwen2.5-7B-Instruct&role=decode&url=http://10.0.0.5:7801&drain=true"
```

其他端点：`GET /v1/models`（代理到任一健康实例）、`POST /reset_prefix_cache`（广播到全部健康实例）。

admin API 未做鉴权，生产部署不要把 router 端口暴露到不可信网络。

## Decode 下发时序（`decode_dispatch`）

| 模式 | 行为 | 适用 |
| --- | --- | --- |
| `sequential`（默认） | P 的 prefill 完成（HTTP 响应返回）后再发 D。与 UCM 官方 `toy_proxy_server.py` 同语义 | 不依赖 D 侧就绪等待能力，正确性最稳妥 |
| `early` | 选完 P/D 后尽快发 D，不等 P 完成；P 失败则中止 D 请求 | v4 理想路径，"D 尽早收到请求，等待发生在 D 侧" |

`early` 模式是博文[待验证前提](https://keriko-blog.pages.dev/posts/037-ucm-pd-router-evolution)的验证载体：

1. D 侧 UCM Connector 具备请求级 KV 就绪判断和等待能力；
2. Connector 的等待不占住引擎 Decode 槽位造成队头阻塞（需要准入控制）；
3. "哈希文件可见"与"KV 可完整读取"的时间差由 Connector 就绪语义兜住。

任何一条不成立时，Router 仍可不感知 Hash，但需在 D 侧补足等待与准入机制，而不是把判断逻辑搬回 Router。

## 快速开始

```bash
pip install -e ".[dev]"

# 跑测试（11 个端到端用例，全部跑在真实 TCP 上）
pytest

# 一键演示：2 mock P + 2 mock D + router，观察 KV 等待→命中与 Round-Robin
bash scripts/demo.sh        # Ctrl-C 退出并清理
```

demo（early 模式）预期观察点：

- 第 1 次请求：mock-D 日志出现 `wait=2`（D 到达时 KV 未就绪，逐块等待）
- 第 2 次请求（相同 prompt）：`hit=2`、`lookup_time` 变短（块已在共享存储）
- 两次请求落在不同的 P/D 节点（每池 Round-Robin）

### 接入真实 vLLM + UCM

P、D 启动配置相同（`kv_role: kv_both`，集中式共享存储模式），仅端口不同：

```bash
export ENABLE_UCM_PATCH=1
vllm serve /path/to/Qwen2.5-7B-Instruct \
  --max-model-len 20000 --tensor-parallel-size 1 \
  --gpu_memory_utilization 0.87 --trust-remote-code \
  --port 7800 --block-size 128 --enforce-eager \
  --kv-transfer-config '{
      "kv_connector": "UCMConnector",
      "kv_role": "kv_both",
      "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
      "kv_connector_extra_config": {"UCM_CONFIG_FILE": "/etc/ucm/pd.yaml"}
  }'
```

UCM 存储配置 `/etc/ucm/pd.yaml`（P/D 共用，`/mnt/ucm-shared` 为共享文件系统挂载）：

```yaml
ucm_connectors:
  - ucm_connector_name: "UcmPipelineStore"
    ucm_connector_config:
      store_pipeline: "Cache|Posix"
      storage_backends: "/mnt/ucm-shared"
enable_event_sync: true
use_layerwise: false
```

然后配置 `config.yaml`（参考 `config.example.yaml`）把两组实例的地址填进 `serving_groups`，启动：

```bash
python -m ucm_router --config config.yaml
```

## 配置项

| 配置 | 说明 |
| --- | --- |
| `routing.serving_groups` | 模型 → P/D 地址列表。KV 兼容（同权重/Tokenizer/KV Layout/并行配置）的节点才能配成一组 |
| `routing.max_inflight_per_node` | 每节点在途上限；0 = 不限。全部到上限返回 429 |
| `routing.session_affinity` | 会话亲和（HRW 哈希），见上文，默认关 |
| `decode_dispatch` | `sequential` / `early`，见上表 |
| `engine.*_timeout_seconds` | 连接 / prefill / decode 流空闲超时 |
| `health.interval_seconds` | 健康探测周期；不健康节点移出轮询，恢复后加回。Router 启动时先探测一轮再放流量；请求时连接级失败会触发被动摘除 |

## 目录结构

```
ucm_router/
├── main.py                  # FastAPI app：/v1/chat/completions, /v1/completions, /health
├── config.py                # YAML 配置加载
├── errors.py                # 错误码区分 P/D（fail-fast，不重试）
├── domain/
│   ├── routing.py           # Routing域: ServingNode / NodePool / ServingGroup / RouteAssignment
│   └── request.py           # Request域: RequestSession 生命周期
├── ports/
│   ├── node_provider.py     # Port ①: 静态 YAML（K8s 是未来实现）
│   ├── engine_client.py     # Port ②: OpenAI 兼容 HTTP（prefill 改写 / decode 透传 / SSE 字节流）
│   └── session_repository.py# Port ③: 内存实现
└── services/
    ├── scheduler.py         # 每池 RR + 健康过滤（不看缓存命中率——v1 规则在 v4 依然成立）
    ├── orchestrator.py      # 两阶段编排（sequential / early）
    └── health.py            # 周期探测 /health
tests/
├── mock_ucm_engine.py       # mock 引擎：哈希块文件 + .ready 标记 + D 侧就绪等待/重算兜底
└── test_e2e.py              # 端到端：透传/等待/失败语义/RR/健康检查
```

## 设计来源

- 设计定稿与推导过程：[UCM PD Router 设计五易其稿](https://keriko-blog.pages.dev/posts/037-ucm-pd-router-evolution)
- UCM 项目：[ModelEngine-Group/unified-cache-management](https://github.com/ModelEngine-Group/unified-cache-management) · [文档](https://ucm.readthedocs.io/en/latest/) · [PD 部署](https://ucm.readthedocs.io/en/latest/user-guide/capabilities/pd-disaggregation/centralized/) · [官方 toy proxy（sequential 模式的语义参照）](https://github.com/ModelEngine-Group/unified-cache-management/blob/main/ucm/pd/toy_proxy_server.py)
- 主要对标实现：[vllm-ascend toy proxy](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py)（prefill 改写口径、preopen、waiting 节点、亲和 key 设计、recomputed 兜底语义的参照，本仓库以干净的模块结构适配 UCM 模式）与 [vllm-project/router](https://github.com/vllm-project/router)（sgl-router fork，两阶段 + connector 参数编排的印证）；参照过的还有 Mooncake toy proxy、llm-d、production-stack orchestrated router

## License

MIT
