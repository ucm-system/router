#!/usr/bin/env bash
# UCM PD Router 一键演示：2 P + 2 D (mock 引擎) + router。
#
# 演示内容（decode_dispatch=early，即 v4 理想路径）：
#   第 1 次请求：D 到达时 KV 未就绪 → D 侧 Connector 逐块等待（观察 mock-decode 的 wait=N）
#   第 2 次请求：相同 prompt 的 KV 块已全部就绪 → 直接命中（观察 hit=N、lookup_time 变短）
#
# 用法：bash scripts/demo.sh     （Ctrl-C 退出，进程自动清理）
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python}"
MODEL="Qwen/Qwen2.5-7B-Instruct"
ROUTER_PORT=9000
STORE="${TMPDIR:-/tmp}/ucm-router-demo-store"
CONFIG="demo-router.generated.yaml"

PIDS=()
cleanup() {
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

start() { # start <名字> <命令...>
  local name="$1"; shift
  echo "[demo] start $name"
  "$@" &
  PIDS+=("$!")
}

wait_port() { # wait_port <端口>
  local port="$1"
  for _ in $(seq 1 60); do
    if "$PY" -c "import socket,sys;s=socket.socket();s.settimeout(1);sys.exit(0 if s.connect_ex(('127.0.0.1',$port))==0 else 1)"; then
      return 0
    fi
    sleep 0.3
  done
  echo "[demo] ERROR: port $port not reachable" >&2
  return 1
}

rm -rf "$STORE"
mkdir -p "$STORE"

start "mock prefill #1" "$PY" tests/mock_ucm_engine.py --role prefill --port 7810 --store "$STORE" --model "$MODEL"
start "mock prefill #2" "$PY" tests/mock_ucm_engine.py --role prefill --port 7811 --store "$STORE" --model "$MODEL"
start "mock decode  #1" "$PY" tests/mock_ucm_engine.py --role decode  --port 7820 --store "$STORE" --model "$MODEL"
start "mock decode  #2" "$PY" tests/mock_ucm_engine.py --role decode  --port 7821 --store "$STORE" --model "$MODEL"

# 演示用 router 配置：early 下发（D 尽早收到请求，等待发生在 D 侧 Connector）
cat > "$CONFIG" <<EOF
server:
  host: 127.0.0.1
  port: ${ROUTER_PORT}

decode_dispatch: early

routing:
  serving_groups:
    - model: "${MODEL}"
      prefill:
        - "http://127.0.0.1:7810"
        - "http://127.0.0.1:7811"
      decode:
        - "http://127.0.0.1:7820"
        - "http://127.0.0.1:7821"

engine:
  connect_timeout_seconds: 5
  prefill_timeout_seconds: 120
  decode_timeout_seconds: 300

health:
  interval_seconds: 5
  timeout_seconds: 2
EOF

start "router" "$PY" -m ucm_router --config "$CONFIG"

for port in 7810 7811 7820 7821 "${ROUTER_PORT}"; do
  wait_port "$port"
done

echo
echo "======================================================================"
echo " 第 1 次请求：KV 未就绪 → D 侧逐块等待（看终端里 mock-decode 的 wait=2）"
echo "======================================================================"
curl -sN "http://127.0.0.1:${ROUTER_PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${MODEL}\",\"stream\":true,\"max_tokens\":20,\"messages\":[{\"role\":\"user\",\"content\":\"describe the UCM PD router in one short sentence\"}]}" \
  | while IFS= read -r line; do
      case "$line" in
        "data: "[{]*) printf "." ;;
        "data: [DONE]") echo "  [stream done]" ;;
      esac
    done

echo
echo "======================================================================"
echo " 第 2 次请求：KV 块已就绪 → 直接命中（观察 hit、lookup_time 变短）"
echo "======================================================================"
curl -sN "http://127.0.0.1:${ROUTER_PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${MODEL}\",\"stream\":true,\"max_tokens\":20,\"messages\":[{\"role\":\"user\",\"content\":\"describe the UCM PD router in one short sentence\"}]}" \
  | while IFS= read -r line; do
      case "$line" in
        "data: "[{]*) printf "." ;;
        "data: [DONE]") echo "  [stream done]" ;;
      esac
    done

echo
echo "[demo] 引擎计数（每个池各处理了一半请求；decode 的 hit/wait/recompute 反映 KV 状态）："
for port in 7810 7811 7820 7821; do
  curl -s "http://127.0.0.1:${port}/stats"
  echo
done

echo
echo "[demo] router 池健康："
curl -s "http://127.0.0.1:${ROUTER_PORT}/health"
echo
echo
echo "[demo] /metrics（自有指标 + PD 分改名合并，节选）："
curl -s "http://127.0.0.1:${ROUTER_PORT}/metrics" | grep -E "^pd_(requests_total|ttft_seconds_count|early_dispatch_lead_seconds_count|retries_total|prefill_duration_seconds_count|pool_healthy)" | head -12
echo
echo "[demo] /admin/nodes（动态增删节点的查询入口）："
curl -s "http://127.0.0.1:${ROUTER_PORT}/admin/nodes"
echo
echo "[demo] 完成。再跑一次同样的请求会全部命中；按 Ctrl-C 退出。"
tail -f /dev/null
