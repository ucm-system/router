# ucm-router

## 项目简介

ucm-router 是 [UCM（Unified Cache Manager）](https://github.com/ModelEngine-Group/unified-cache-management) PD 分离部署下的 LLM Router。它对客户端提供 OpenAI 兼容接口，对内维护 Prefill（P）池和 Decode（D）池，把一次请求拆成 prefill、decode 两段提交给引擎实例，再把 D 的响应原样送回客户端。

三个角色的分工：Router 决定请求去哪里，UCM 决定 KV 什么时候准备好，引擎决定计算什么时候执行。所以 Router 不持有 Tokenizer，不计算 block hash，不查询缓存，也不编排预取；它会在请求里声明自己编排的是哪条腿、期望哪种 KV 加载意图，但块范围、地址和时机都由 Connector 自己定。命中判断、SSD 到 DRAM 的提升、等待 P 尚未写完的 KV，全部在 D 侧 Connector 里完成。

进程内只有两个领域（Routing、Request）和三个 Port（NodeProvider、EngineClient、SessionRepository），没有数据库、消息队列等外部依赖，单进程即可运行。运行依赖只有 FastAPI、uvicorn、httpx、PyYAML 和 `prometheus_client`，后者 vLLM 自己就带着。

已实现：OpenAI 兼容入口与 SSE 字节透传、每池独立的 Round-Robin 选路、健康探测与被动摘除、在途容量上限、同池换节点 failover、preopen、可选会话亲和、发给两腿的 router 编排声明、Prometheus 指标汇总、动态增删节点。

未实现，也不在当前版本计划内：K8s 服务发现（NodeProvider 只有静态 YAML 实现）、prefix 或缓存感知调度、鉴权、TLS、熔断与退避、请求排队、跨进程共享状态。过载和失败一律立即返回错误码，退避由客户端负责。

设计推导过程记录在[《UCM PD Router 设计五易其稿》](https://keriko-blog.pages.dev/posts/037-ucm-pd-router-evolution)，本仓库是该文 v4 定稿的实现，sequential 模式与 UCM 官方 [toy_proxy_server.py](https://github.com/ModelEngine-Group/unified-cache-management/blob/main/ucm/pd/toy_proxy_server.py) 同语义。

## 架构设计

```
                        ┌─────────────────────────────────────────┐
 Client ──── SSE ───────▶│  Router (FastAPI + httpx)               │
 (OpenAI SDK)            │  ├─ Scheduler      每池 RR / 健康过滤   │
                         │  ├─ Orchestrator   两阶段提交           │
                         │  └─ HealthChecker  周期 GET /health     │
                         └───────────┬─────────────────┬──────────┘
                          1. prefill │       2. decode │
                                     ▼                 ▼
                              P instances        D instances
                            (vLLM + UCM        (vLLM + UCM
                             Connector           Connector
                             kv_both)            kv_both)
                                     │                 ▲
                                     ▼   block lookup  │
                            ┌───────────────────────────┐
                            │  共享 KV 存储（UCM store） │
                            └───────────────────────────┘
```

P 与 D 之间没有直接网络传输。P 把 KV 块以哈希文件名写入共享存储（去中心化，没有 master 节点），D 收到请求后由 UCM Connector 逐块 lookup，就绪后放行 decode。Router 看不见这个过程，它只知道"先向 P 提交、再向 D 提交"。

```
ucm_router/
├── main.py                    FastAPI 应用与所有 HTTP 端点（/v1、/health、/metrics、/admin）
├── config.py                  YAML 配置加载与启动期校验
├── errors.py                  错误类型、错误码与 HTTP 状态码映射
├── metrics.py                 Router 自有指标（prometheus_client 的 counter / gauge / histogram）
├── domain/
│   ├── routing.py             ServingNode / NodePool / ServingGroup / RouteAssignment
│   └── request.py             RequestSession 的阶段与时间戳
├── ports/
│   ├── node_provider.py       Port：节点来源（静态 YAML + admin 运行时增删）
│   ├── engine_client.py       Port：引擎调用（prefill 改写、router 声明、SSE 字节流）
│   └── session_repository.py  Port：请求上下文（进程内字典）
└── services/
    ├── scheduler.py           选路：候选过滤、RR/HRW、容量判断
    ├── orchestrator.py        请求生命周期编排与响应代理
    ├── health.py              启动探测、周期探测、被动摘除、waiting 队列轮询
    └── metrics_proxy.py       抓取实例 /metrics，改名后与自有指标合并
tests/
├── mock_ucm_engine.py         mock 引擎：哈希块文件 + .ready 标记 + D 侧等待与重算兜底 + 消费 router 声明
└── test_e2e.py                29 个用例（端到端 + 请求改写单元）
```

domain 只回答"谁可以被选"和"这次请求处在哪个阶段"，services 负责编排，ports 是与外部系统（引擎 HTTP、节点来源、请求状态）的边界。每个 Port 目前只有一个实现，换成 K8s 服务发现或持久化的会话状态都不需要改 domain 和 services。

一个 ServingGroup 里的 P 和 D 必须权重、Tokenizer、KV 数据类型、block size、KV layout、并行配置一致，否则 D 读不到 P 写出的块。请求按 `model` 字段精确匹配到 group，匹配不到直接返回 404。

## 路由请求：选路、示例与时机

### 选路

Scheduler 在 P 池和 D 池各取一个节点，两个池的轮询游标互相独立。候选节点要同时满足：健康、不在排空（draining）中、在途请求数小于 `max_inflight_per_node`（0 表示不限）。没有候选时分两种情况：池内没有任何健康节点返回 503，有健康节点但都在途到上限返回 429。

选路结果 `RouteAssignment` 是冻结的 dataclass，换节点重试时用 `replace` 生成新实例。整个判断只依赖节点健康、容量和（可选的）会话标识，不读取任何 KV 或命中率信息，也不解析 prompt。

### 一次请求的完整过程

```
Client          Router                          P                     D
  │  POST          │                             │                     │
  │───────────────▶│ 选 P、选 D，冻结分配          │                     │
  │                │── 改写后的请求 ─────────────▶│ prefill + KV 落盘    │
  │                │◀─ 200 + usage ──────────────│                     │
  │                │── 原始请求 ──────────────────────────────────────▶│ Connector lookup / 等待
  │◀━━ SSE 字节透传 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━│
```

客户端请求：

```bash
curl -N http://127.0.0.1:9000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-7B-Instruct","stream":true,"max_tokens":20,
       "messages":[{"role":"user","content":"describe the UCM PD router"}]}'
```

Router 发给 P 的请求由 `rewrite_for_prefill` 生成：`stream` 改为 false，`max_tokens` 和 `min_tokens` 置 1（存在 OpenAI 新口径 `max_completion_tokens` 时同步置 1），去掉只在流式下合法的 `stream_options`，其余字段保留，再叠加 router 声明（见下文「发给引擎的 router 声明」），并附 `X-Request-Id` 头。改成阻塞单 token 是为了拿到 P 的响应，其 `usage` 用于下文的 cached_tokens 口径。

Router 发给 D 的是客户端的 JSON 外加一条 router 声明（见下文「发给引擎的 router 声明」），其余字段一字不改，另附 `X-Request-Id` 头。声明里只有 Router 自己知道的事：这条请求是本次编排的 decode 腿、编排标识、期望的 KV 加载意图。缓存状态、块哈希、对端地址一概不写。

透传有这一处例外，值得从契约上讲清：`kv_transfer_params` 这个袋子客户端自己也能传，Router 只接管其中的 `router` 子键（客户端伪造的值被 Router 写的覆盖），袋子里属于别人的键继续原样送达 P 和 D。UCM 目前不读这些键，所以没有副作用；换成会真按 `remote_host`/`remote_port` 建连的 connector（Mooncake、NIXL 一类）时，客户端可控的传输参数就成了风险面，那时需要在入口一并剥掉。

响应形态：`stream=true` 时 D 的 SSE 按字节透传，Router 不解析 chunk 内容；`stream=false` 时解析 D 的 JSON，合并 cached_tokens 口径后返回。两种模式都带同样的响应头：

```
X-Request-Id: 73235e58cc4d4cef90b332102de4e6aa
X-PD-Prefill-Node: http://10.0.0.3:7800
X-PD-Decode-Node:  http://10.0.0.5:7801
X-PD-Prefill-Cached-Tokens: 128
```

`X-Request-Id` 同时作为请求头发给 P 和 D。引擎是否会把它打进自己的日志要在引擎侧确认（仓库里的 mock 引擎就不读它，只读 body），vLLM 若不透传这个头，就只能靠时间与 body 内容对账；需要引擎侧稳定读到编排标识时，用下面这个 body 字段。

### 发给引擎的 router 声明

`kv_both` 形态下 P 和 D 的启动命令完全相同（只有端口不同，见接真实 vLLM 那小节），实例没有任何"我是 D"的静态依据。目前它唯一的线索是请求形状：`max_tokens=1` 且非流式的那条就是 prefill 腿。那是碰巧，不是声明，Connector 要按角色决定加载策略时不能依赖它，所以这个信息得由 Router 显式给出。

放 body 不放 header：vLLM 的 connector 回调拿到的是 `SamplingParams.kv_transfer_params`，HTTP header 停在 API server 层，连接器读不到。Router 因此在 `kv_transfer_params.router` 下写一个自己拥有的子键：

```json
"kv_transfer_params": {
  "router": {
    "execution_role": "decode",
    "request_id": "4521163032634863939e42fb2ff831c1",
    "kv_load": "prefetch_only"
  }
}
```

prefill 腿的两个值是 `execution_role: "prefill"` 和 `kv_load: "store_only"`，`request_id` 两腿相同、与响应头 `X-Request-Id` 同源。三个字段都只描述 Router 自己的决策：这条请求被分到哪条腿、属于哪次编排、期望的 KV 加载意图。`kv_load` 把"预取但不推"写成显式意图，而不是让 connector 从角色名反推；以后要加别的加载策略就多一个取值，语义不用改。

契约的几条边界：

- 所有权：`router` 子键由 Router 独占，每次覆盖，客户端伪造的值无效。这个名字就此定死，UCM 侧不应再在 `kv_transfer_params` 里占用同名键。
- 仍然不碰 KV：声明里没有命中数、没有块哈希、没有就绪水位、没有对端地址。Router 说的是"我怎么编排的"，不是"缓存里有什么"，v4 的分工不变。
- fail-open：不认识这个袋子的 connector 忽略它，旧引擎不会因为升级 Router 而改变行为。灰度是否生效由消费方自己计数，经实例 `/metrics` 暴露、由 Router 改名透传成 `pd_{role}_...`，Router 不需要去解析响应字节。仓库里的 mock 引擎是这条路的参照实现：`mock_router_hint_seen_total` 记录读到几条声明，`mock_router_hint_mismatch_total` 记录声明的角色与实例实际承担的角色不符（正常恒为 0，非 0 说明 Router 写错了腿）。UCM Connector 采纳后按同样口径加两条 counter 就能观测。
- 回退开关：`engine.router_hint: false` 停止注入，decode 腿回到完全原样透传。注意这只关掉 Router 的写入，不拦客户端：客户端自己伪造的 `router` 值照样直达两腿（测试里用 mismatch 计数把这条钉住）。
- 与官方 toy proxy 的关系：toy proxy 发的是原始 prompt，加了声明之后两者不再是同一份字节。UCM 忽略未知键，行为仍然等价，"同语义"按这个口径理解。
- 换节点重试不受影响：failover 只换地址，`execution_role` 和 `kv_load` 前后不变。

### cached_tokens 的两个口径

PD 分离下客户端只能看到 D 的 usage，而 prefill 侧复用发生在 P。Router 的处理方式是：

非流式响应里 `usage.prompt_tokens_details.cached_tokens` 取 P 的值（计费口径，表示 prefill 侧复用了多少），D 的原值和两端节点信息一起放进 `pd` 字段（数值示意，block size 128 的场景）：

```json
{
  "choices": [ ... ],
  "usage": {"prompt_tokens": 130, "prompt_tokens_details": {"cached_tokens": 128}},
  "pd": {
    "prefill": {"node": "http://10.0.0.3:7800", "cached_tokens": 128, "prompt_tokens": 130},
    "decode": {"node": "http://10.0.0.5:7801", "cached_tokens": 0, "recomputed": false}
  }
}
```

流式响应保持字节级透传，不改写 usage chunk，P 的口径放在 `X-PD-Prefill-Cached-Tokens` 头里。两种模式下 Router 都在 P 返回之后才向客户端发出响应，所以头部值在写出前已经确定。

UCM 从共享存储实际加载了多少块，是引擎与 UCM 侧的观测，Router 不感知。上面这些数值只用于账务和观测，不参与路由决策。

### Decode 下发时机

| `decode_dispatch` | 行为 | 依赖 |
| --- | --- | --- |
| `sequential`（默认） | P 的 HTTP 响应返回后再把请求发给 D | 无，不依赖 D 侧的就绪等待能力 |
| `early` | 选完 P/D 后立刻向 D 提交请求，不等 P；P 失败则中止已建立的 D 请求 | D 侧 Connector 需要具备三项能力，见下 |

`early` 把 KV 等待从 Router 挪到了 D，收益是客户端 TTFT 少一段串行等待。它的前提是：

1. D 侧 UCM Connector 具备请求级的 KV 就绪判断和等待能力；
2. Connector 的等待不占用引擎的 decode 槽位，否则会造成队头阻塞（需要准入控制）；
3. "哈希文件已可见"与"KV 可完整读取"之间的时间差由 Connector 的就绪语义兜住。

UCM 文档同样提醒：prefill 的 HTTP 响应只是时序点，不代表存储已经持久可读。这三条中任何一条不成立时，正确做法是在 D 侧补齐等待与准入机制，而不是把判断逻辑搬回 Router。

`pd_early_dispatch_lead_seconds` 记录从选路完成到 "P 返回、D 的响应流建立好" 之间的时长，也就是这段被 P 的计算掩盖掉的等待。由于 D 的响应头通常也要等 KV 就绪才到达，这个值和 `pd_prefill_duration_seconds` 很接近，把它当作提前量的上界来看更合适。

### 会话亲和（默认关闭）

打开 `routing.session_affinity` 后，同一会话稳定落到同一组 P/D，多轮对话的 KV 复用更好。

key 的派生优先级：请求的 `user` 字段，其次 `X-Session-Id` 请求头，最后用 `model` 加上前置 system/developer 消息和第一条 user 消息。第三种口径来自 vllm-ascend toy proxy 的 `conversation_affinity_key`，作用是避免哈希持续增长的历史导致 key 每轮都变。

放置算法是 Rendezvous（HRW）哈希，取 `sha256(key + node url)` 最大的节点。它在跨进程和重启后保持稳定，节点增减时只重映射恰好落在被移除节点上的会话。亲和是尽力而为的：目标节点不健康、在排空或在途到上限时，仍会从剩余候选里按同一规则选一个，不会因此失败。

这是确定性放置而不是缓存感知调度：输入是请求元数据，不需要 Tokenizer，也不读取 KV 状态，仍在 v4 边界内。真正的命中判断在 UCM Connector。

## 错误处理

错误响应统一为下面的结构，由 `errors.py` 里的类型定义、`main.py` 的 exception handler 序列化：

```json
{"error": {"code": "PREFILL_REQUEST_FAILED",
           "message": "prefill node http://10.0.0.3:7800 returned 500: ..."}}
```

| 场景 | HTTP | code | 说明 |
| --- | --- | --- | --- |
| 请求体不是 JSON 对象、缺 `model`；admin 请求体缺字段或 role 非法 | 400 | `INVALID_REQUEST` | 未进入编排 |
| `model` 没有对应的 ServingGroup | 404 | `NO_SERVING_GROUP` | 未进入编排 |
| 选路时某池无健康节点 | 503 | `PREFILL_SERVICE_UNAVAILABLE` / `DECODE_SERVICE_UNAVAILABLE` | 请求不发出，另一侧也不会被打到 |
| 某池健康节点全部在途到上限 | 429 | `PREFILL_AT_CAPACITY` / `DECODE_AT_CAPACITY` | 不排队，客户端自行退避 |
| P 连接失败或后端 5xx | 502 | `PREFILL_REQUEST_FAILED` | 先在池内换一个节点重试，没有可换的节点才返回 |
| D 连接失败或后端 5xx | 503 | `DECODE_REQUEST_FAILED` | 同上，换 D 重试 |
| 后端返回 4xx（例如 prompt 超长） | 502 / 503 | `*_REQUEST_FAILED` | 4xx 不重试，后端的响应片段进 message |
| P 或 D 读超时 | 502 / 503 | `*_REQUEST_FAILED` | 不重试（客户端已经等满 `*_timeout_seconds`），也不触发被动摘除 |
| 流式下 D 首字节前失败 | 503 | `DECODE_REQUEST_FAILED` | preopen 保证是真实状态码，不是空 200 加截断流 |
| 流式下首字节之后失败 | 截断流 | 无（只记指标） | 客户端看到缺少 `[DONE]` 的流，Router 记 `DECODE_STREAM_INTERRUPTED` |
| D 侧等不到 KV 自行重算 | 200 | 无（只记指标） | 引擎报 `stop_reason=recomputed`，响应本身成功，Router 计 `pd_recomputed_total` |
| admin 新增节点已在池中 / 删除不存在的节点 | 409 / 404 | `NODE_EXISTS` / `NODE_NOT_FOUND` | |

换节点重试为什么安全：UCM 集中式模式下 KV 块以内容哈希为文件名，同一个 prompt 换到另一个 P 重做 prefill，写出的是完全相同的块文件，重复执行幂等。重试只在同池内进行，每次排除刚失败的那个节点，另一侧的分配保持不变。

这段循环有个已知的边界问题。它只靠"排除上一个失败节点"推进，没有 hop 数上限；节点要等错误冒泡回入口之后才会被被动摘除。于是当池里两个以上节点同时死亡、而周期探测还没跟上时，循环会在它们之间来回绕。实测 `health.interval_seconds=3600`、两台 P 都已死掉的情况下，一个请求在 20 秒内换了 11285 次节点，客户端始终拿不到响应；探测间隔取秒级时，探测会在很短时间内把节点摘掉，请求随即返回 502。调小探测间隔是当前的缓解办法，根治需要让重试记住已经试过的 URL。

哪些失败算节点不可达：连接级错误（连接失败、连接超时、读错误、连接池超时、协议错误）以及空 body 的 5xx 视为不可达。后者不会来自推理引擎本身，通常是网关或目标端口已死。不可达会触发被动健康摘除，节点立即移出轮询，不等下一个探测周期；后端返回 5xx 且有响应体说明进程还活着，不摘除。

`early` 模式下如果 P 在重试后仍失败，Router 会取消 prefill 任务并关闭已经建立的 D 请求，客户端拿到的是 P 侧的 502。重算兜底属于引擎和 UCM 的职责，Router 不因为 D 重算而重试或报错。

所有端点（包括 `/admin/nodes`）都没有鉴权，也没有 TLS 和全局限流。不要把 Router 端口暴露到不可信网络，需要鉴权与限流请放在 Router 前面的网关上。

## 可观测性

### 日志

`python -m ucm_router` 安装 root handler，格式 `%(asctime)s %(levelname)s %(name)s %(message)s`，级别由 `--log-level` 控制（同时传给 uvicorn）。logger 名字分别是 `ucm_router`、`ucm_router.orchestrator`、`ucm_router.health`、`ucm_router.metrics_proxy`。

一次流式请求留下的四行（demo 环境的实际输出）：

```
2026-11-16 20:01:38,988 INFO ucm_router.orchestrator req=73235e58cc4d4cef90b332102de4e6aa model=Qwen/Qwen2.5-7B-Instruct p=http://127.0.0.1:7810 d=http://127.0.0.1:7820 dispatch=early affinity=off
2026-11-16 20:01:39,686 INFO ucm_router.orchestrator req=73235e58cc4d4cef90b332102de4e6aa prefill ok on http://127.0.0.1:7810 in 0.695s
2026-11-16 20:01:39,732 INFO ucm_router.orchestrator req=73235e58cc4d4cef90b332102de4e6aa first byte from http://127.0.0.1:7820, ttft=0.743s
2026-11-16 20:01:40,143 INFO ucm_router.orchestrator req=73235e58cc4d4cef90b332102de4e6aa stream complete in 1.155s
```

选路那行同时给出这次用的 P、D 地址、下发模式与亲和开关，出问题时最先看它。异常与状态变化的行如下（省略时间戳，id 略写）：

```
WARN  ucm_router  req failed: DECODE_REQUEST_FAILED decode node ... failed before first byte: ...
WARN  ucm_router.orchestrator  req=... prefill node http://10.0.0.3:7800 failed (...), failing over to http://10.0.0.4:7800
WARN  ucm_router.orchestrator  req=... decode side reported recomputed: KV wait fell back to recompute
ERROR ucm_router.orchestrator  req=... stream interrupted from http://10.0.0.5:7801 after 3.221s: ...
WARN  ucm_router.health  node http://10.0.0.3:7800 (prefill) is down: removed from rotation
INFO  ucm_router.health  node http://10.0.0.3:7800 (prefill) is back: added to rotation
INFO  ucm_router.health  waiting node http://10.0.0.9:7801 (decode) is ready: added to rotation
INFO  ucm_router.orchestrator  drained node http://10.0.0.5:7801 (decode) removed from pool
```

`req=` 的值就是响应头 `X-Request-Id`，也是发给 P 和 D 的同一个请求头，router 内部两段调用靠它串；能不能再往下串到引擎日志，取决于引擎是否记录这个头。

INFO 级别下 httpx 会为每一次引擎调用和每一次健康探测各打一行，探测周期短的时候这部分会盖过 router 自己的日志。只想看 router 的行就 `--log-level warning`，或者把 `httpx` 这个 logger 单独调高级别。

### 状态端点

`GET /health` 返回 Router 自身与池的即时状态，`active_sessions` 是进程内在途会话数：

```json
{"status": "ok", "active_sessions": 0,
 "pools": [{"model": "Qwen/Qwen2.5-7B-Instruct",
           "prefill": {"total": 2, "healthy": 2, "max_inflight": 0,
                       "nodes": [{"url": "http://127.0.0.1:7810", "healthy": true,
                                  "draining": false, "inflight": 0}],
                       "waiting": []},
           "decode": {"total": 2, "healthy": 2, "max_inflight": 0,
                      "nodes": [ ... ], "waiting": []}}]}
```

`nodes[]` 里每个实例给出 healthy、draining、inflight 三个状态，`waiting[]` 给出 admin 新增但还没探测通过的节点及其 `probe_attempts`。同一份数据也可以通过 `GET /admin/nodes` 拿到。

### GET /metrics

Prometheus 文本格式，由 `prometheus_client` 渲染。Router 自有指标只覆盖选路与请求生命周期：

| 指标 | 类型 | 标签 | 含义 |
| --- | --- | --- | --- |
| `pd_requests_total` | counter | dispatch, model, outcome | 请求量与成败（outcome: ok / failed） |
| `pd_failures_total` | counter | code | 按错误码细分的失败，含 `DECODE_STREAM_INTERRUPTED` |
| `pd_node_selections_total` | counter | instance, role | 各节点被选中的次数，用来检查 RR 是否公平 |
| `pd_retries_total` | counter | role | 同池换节点 failover 次数 |
| `pd_recomputed_total` | counter | model | D 侧等待 KV 回退重算的次数 |
| `pd_prefill_duration_seconds` | histogram | model | P 段耗时（含重试） |
| `pd_ttft_seconds` | histogram | model | Router 视角的 TTFT，含 prefill |
| `pd_early_dispatch_lead_seconds` | histogram | model | early 模式下从选路到 P 完成、D 已建流的时长 |
| `pd_sessions_active` | gauge | instance, role | 每节点在途请求数 |
| `pd_pool_healthy` | gauge | model, role | 每池健康节点数 |
| `pd_stream_bytes_total` | counter | model | 从 D 透传出去的字节数 |
| `pd_scrape_failures_total` | counter | instance, role | 抓取实例 `/metrics` 失败的次数 |

三个耗时指标是带分桶的 histogram，桶边界见 `metrics.DURATION_BUCKETS`（5ms 到 300s，上限与 `engine.decode_timeout_seconds` 的默认值对齐），分位数直接可算：

```promql
histogram_quantile(0.95, sum by (le) (rate(pd_ttft_seconds_bucket[5m])))
```

counter 和 histogram 各自附带一条 `_created` 时间序列，是 `prometheus_client` 的标准输出。

用库而不是自己拼 exposition，理由是这个 Router 只会和 vLLM 跑在同一套环境里，而 vLLM 的 `requirements/common.txt` 本身就带着 `prometheus_client >= 0.18`；UCM 的 connector 指标也是经 vLLM 的 Prometheus 桥导出的，同一套格式与语义不必重写一遍，顺带还拿到了 histogram。只使用私有 CollectorRegistry，不碰库的默认 registry，`/metrics` 里因此没有 `process_*` / `python_*` 那组采集器。

实例侧指标由 `metrics_proxy.py` 并发抓取每个实例的 `/metrics`（3s 超时），把 metric family 改名加上 `pd_{role}_` 前缀并注入 `pd_role`、`pd_instance` 标签后合并输出。这一段是文本级改写，没走库的 parser：解析再渲染会把 `_created`、无类型家族这类条目降级成独立的 untyped 家族，而代理该做的是原样搬、只改名字与标签。改名是必须的：P 和 D 的同名 family 直接合并会让 Prometheus 拒绝整个端点。抓取失败只累加 `pd_scrape_failures_total`，不影响其余部分。

`pd_recomputed_total` 是唯一与 KV 相关的指标，但它的来源不是 Router 的判断，而是引擎响应里的 `stop_reason=recomputed`：非流式路径解析 JSON 精确匹配，流式路径在字节流上做子串门检（同一个 chunk 同时包含 `"stop_reason"` 和 `recomputed`）。误报需要模型正文恰好生成这段 JSON 字面量，可以忽略。

## 使用方式

### 安装与启动

需要 Python 3.10 以上，依赖是 fastapi、uvicorn、httpx、pyyaml 和 prometheus_client（最后这个在 vLLM 环境里已经装好了）。

```bash
pip install -e ".[dev]"

cp config.example.yaml config.yaml   # 改 routing.serving_groups 里的实例地址
python -m ucm_router --config config.yaml --log-level info
```

配置在启动期校验，不合法的输入直接抛错而不是静默忽略：`decode_dispatch` 只能取两个值之一，`serving_groups` 至少一组且每组的 `prefill`、`decode` 都是非空 URL 列表，`max_inflight_per_node` 不能为负，配置段落里出现未知 key 也会报错。

### 配置项

| 配置 | 默认值 | 说明 |
| --- | --- | --- |
| `server.host` / `server.port` | `0.0.0.0` / `9000` | 监听地址 |
| `decode_dispatch` | `sequential` | `sequential` 或 `early`，见上文 |
| `routing.serving_groups` | 无 | 模型名到 P/D 地址列表，KV 兼容的实例才能配成一组 |
| `routing.max_inflight_per_node` | `0` | 每节点在途上限，0 表示不限；全部到上限返回 429 |
| `routing.session_affinity` | `false` | HRW 会话亲和 |
| `engine.connect_timeout_seconds` | `5` | 与引擎建连的超时 |
| `engine.prefill_timeout_seconds` | `120` | 等 P 返回响应的读超时 |
| `engine.decode_timeout_seconds` | `300` | D 的响应流上相邻字节间的空闲超时 |
| `engine.router_hint` | `true` | 是否在两腿请求里写 `kv_transfer_params.router`，false 时 decode body 完全透传 |
| `health.interval_seconds` | `5` | 健康探测周期 |
| `health.timeout_seconds` | `2` | 单次探测超时 |
| `health.waiting_max_retries` | `12` | admin 新增但未就绪的节点最多探测多少轮，超上限放弃 |

Router 启动时会先完整探测一轮再放流量，之后按 `health.interval_seconds` 周期探测，请求期间的连接级失败则走被动摘除。

### 客户端调用

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:9000/v1", api_key="unused")
stream = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "describe the UCM PD router"}],
    max_tokens=20,
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
```

Router 不做鉴权，`api_key` 只是为了让 SDK 构造出客户端。

### 本地演示（不需要 GPU）

```bash
bash scripts/demo.sh        # Ctrl-C 退出（只杀进程）
```

解释器按 `PYTHON` → `python3` → `python` 的顺序取，依赖装在哪个环境里就指哪个，例如 `PYTHON=.venv/bin/python bash scripts/demo.sh`。

它起 2 个 mock P（7810、7811）、2 个 mock D（7820、7821）和一个 early 模式的 router（9000），共享存储目录在 `${TMPDIR:-/tmp}/ucm-router-demo-store`（每次运行先清空，保证冷 KV），配置文件写在仓库根目录的 `demo-router.generated.yaml`（每次覆写，已被 .gitignore 收录）。随后发两次完全相同的流式请求，依次打印两腿收到的 router 声明、4 个实例的 `/stats`、router 的 `/health`、`/metrics` 节选和 `/admin/nodes`，然后停在 `tail -f /dev/null` 等 Ctrl-C。

一次实际运行的输出：

```
[mock-decode] prompt=9tok blocks=3 hit=0 wait=3 recompute=0 lookup_time=694ms router_hint=decode/prefetch_only
[mock-decode] prompt=9tok blocks=3 hit=3 wait=0 recompute=0 lookup_time=0ms router_hint=decode/prefetch_only

[demo] router 写给两腿的声明（kv_transfer_params.router，从引擎侧收到的 body 里读回）：
  :7810 {"execution_role": "prefill", "request_id": "bf884a1c00e74716a8226b6689f0f506", "kv_load": "store_only"}
  :7820 {"execution_role": "decode",  "request_id": "bf884a1c00e74716a8226b6689f0f506", "kv_load": "prefetch_only"}
```

第一次请求时 D 比 P 早到，3 个块全都在轮询等待（P 写完 `.kv` 后要再等 150ms 才补 `.ready`，制造"文件可见但 KV 未就绪"的窗口）；第二次相同 prompt 落到另一台 D，3 个块首查即就绪，`lookup_time` 从 694ms 掉到 0ms。router 侧对应的 TTFT 是 0.734s 与 0.241s，两次的选路行分别是 `p=7810 d=7820` 和 `p=7811 d=7821`，即每池各自轮询。4 个实例的 `/stats` 也对得上：每台各处理 1 次请求，`router_hints_seen` 各 1、`router_hint_mismatch` 各 0。

`/metrics` 节选里能看到声明这条路走完了：mock 引擎自己计数 `mock_router_hint_seen_total`，router 抓取后按角色改名成 `pd_decode_mock_router_hint_seen_total{pd_instance=...}`，所以灰度是否被吃到不需要 Router 解析响应字节就能观测。

mock 引擎的 token 数与耗时都是示意值，它不追求推理正确性，只让两阶段编排、字节透传、TTFT 差异和就绪等待能被端到端看到。这几个延时都可以调：`--ready-delay-ms`、`--prefill-base-ms`、`--block-tokens`、`--kv-wait-timeout-ms` 等参数直接传给 `tests/mock_ucm_engine.py`，手工起实例就能观察不同的等待形态（`scripts/demo.sh` 用的是默认值）。

### 接真实 vLLM + UCM

P、D 两组的启动配置相同，都是 `kv_role: kv_both` 的集中式共享存储模式，只有端口不同：

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

UCM 存储配置 `/etc/ucm/pd.yaml`，P、D 共用同一份：

```yaml
ucm_connectors:
  - ucm_connector_name: "UcmPipelineStore"
    ucm_connector_config:
      store_pipeline: "Cache|Posix"
      storage_backends: "/mnt/ucm-shared"
enable_event_sync: true
use_layerwise: false
```

`/mnt/ucm-shared` 必须在两侧指向同一个共享文件系统挂载，两台机器上各自同名的本地目录不构成共享存储。模型目录名、dtype、block size 等要一致，起步阶段并行配置也保持相同。UCM 侧还有一个 `cache_buffer_capacity_gb`（官方 1p1d 例子取 32），那是 DRAM 缓冲容量，与 Router 无关。更完整的部署与验证步骤见 UCM 文档的[共享存储 PD 部署](https://ucm.readthedocs.io/en/latest/user-guide/capabilities/pd-disaggregation/centralized/)。

之后把实例地址填进 `config.yaml` 的 `serving_groups` 并启动 Router。

### 动态增删节点

```bash
# 添加：探测通过立即入池（201）；未通过进等待队列（202），
# 由健康检查按 interval 轮询，就绪后自动激活，超过 waiting_max_retries 轮放弃。
curl -X POST http://127.0.0.1:9000/admin/nodes \
  -d '{"model":"Qwen/Qwen2.5-7B-Instruct","role":"decode","url":"http://10.0.0.5:7801"}'

# 查看每节点的 healthy / draining / inflight 以及 waiting 列表
curl http://127.0.0.1:9000/admin/nodes

# 删除：drain=true（默认）先停止接新请求，在途清零后移除；drain=false 立即摘除
curl -X DELETE "http://127.0.0.1:9000/admin/nodes?model=Qwen/Qwen2.5-7B-Instruct&role=decode&url=http://10.0.0.5:7801&drain=true"
```

变更不持久化，重启后以 YAML 配置为准。另外两个运维端点：`GET /v1/models` 依次问任一健康实例，取第一个成功的响应，全部失败返回 503 `NO_BACKEND_AVAILABLE`；`POST /reset_prefix_cache` 广播到全部健康实例，返回 `{"reset": 2, "failed": ["http://...:7801"]}` 这样的成功数与失败地址列表（mock 引擎会返回 200 但什么都不清，它没有 prefix cache）。

### 测试

```bash
pytest        # 29 个用例，约 42s
```

29 个用例里 27 个会在后台线程上用 uvicorn 起真实 TCP 服务（端口临时分配），Router 和 mock 引擎是彼此独立的 server，走真实网络路径而不是进程内 ASGI；剩下 2 个是纯函数断言（`rewrite_for_prefill`、`stamp_router_hint`），不起服务。整套跑完启动 88 个 server 实例，耗时主要在服务起停，本机约 42 秒。

有断言覆盖的行为：

- sequential 与 early 下的两阶段编排，流式（校验 8 个 `data:` chunk 与 `[DONE]`）与非流式透传，`/v1/completions` 入口
- early 模式下 D 侧逐块等待 KV（`blocks_waited == 2`）
- 错误码：404 无匹配 group、503 池空（并断言请求没打到 P）、502 P 不可达、503 D 不可达、429 在途上限
- P 侧同池换节点 failover（`pd_retries_total{role="prefill"}`）、被动健康摘除、preopen 首字节前失败返回真实 503
- 流中断后 `pd_failures_total{code="DECODE_STREAM_INTERRUPTED"}`、cached_tokens 双口径（JSON 字段与响应头各一条）、`stop_reason=recomputed` 的两条识别路径（非流式精确解析、流式子串门检）、HRW 会话亲和、每池 RR 分布
- admin 的 201/202/400/404/409 与 drain 删除、waiting 队列自动激活、`/metrics` 自有指标与 `pd_{role}_` 改名透传、`/v1/models` 与 `/reset_prefix_cache`
- router 声明：两腿各自收到的 `execution_role`/`kv_load`、`request_id` 与响应头同源、客户端伪造的 `router` 值被覆盖、同袋其他键继续透传、引擎侧计数经 `pd_{role}_` 改名出现在 router `/metrics`、`router_hint=false` 时 decode body 与客户端 JSON 完全相等且伪造值直达 P（被 `router_hint_mismatch` 抓到）。mock 引擎用 `/last_request` 回吐它收到的 body，断言打在真实网络上而不是打在 router 代码里

下面这些行为代码里有实现，但目前没有对应断言，改动时请手工验证：D 侧的换节点重试与 D 侧 429、请求超时不重试、不健康节点探测恢复后加回轮询、early 模式下 P 失败中止已建立的 D 请求。

## 设计来源

- 设计定稿与推导过程：[UCM PD Router 设计五易其稿](https://keriko-blog.pages.dev/posts/037-ucm-pd-router-evolution)
- UCM 项目：[代码仓库](https://github.com/ModelEngine-Group/unified-cache-management)、[文档](https://ucm.readthedocs.io/en/latest/)、[PD 部署](https://ucm.readthedocs.io/en/latest/user-guide/capabilities/pd-disaggregation/centralized/)、[官方 toy proxy（sequential 模式的语义参照）](https://github.com/ModelEngine-Group/unified-cache-management/blob/main/ucm/pd/toy_proxy_server.py)
- 主要对标实现：[vllm-ascend toy proxy](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py)，prefill 改写口径、preopen、waiting 节点、亲和 key、recomputed 兜底语义都取自它，本仓库按 UCM 的集中式共享存储模式重新组织；[vllm-project/router](https://github.com/vllm-project/router)（sgl-router fork）同样是两阶段编排加 connector 参数这条路。另外参考过 Mooncake toy proxy、llm-d 和 production-stack 的 orchestrated router。

## License

MIT
