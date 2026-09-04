#!/bin/bash
# ============================================================================
# start-deepseek.sh - start DeepSeek-V4-Flash-Vision distributed across 2 Macs
#
# Every setting below is annotated with the measurement that justifies it.
# See ../README.md for the numbers and ../../shared/ for background.
#
#   ./start-deepseek.sh              start (with prefix-cache warmup)
#   ./start-deepseek.sh --no-warmup  start without warmup
#   ./start-deepseek.sh --status     show status
#   ./start-deepseek.sh --stop       stop server and worker
#
# Configure via environment variables or edit the defaults below.
# ============================================================================
set -u

# ----------------------------------------------------------------- configure
WRK_SSH=${WRK_SSH:-user@worker.local}          # SSH target of the worker machine
WRK_RPC=${WRK_RPC:-192.168.0.2:50053}          # THUNDERBOLT address of the worker.
                                               # Using the LAN address instead works
                                               # and silently routes every token
                                               # over Ethernet. Don't.

BUILD=${BUILD:-$HOME/llama.cpp-rpc/build}      # patched build, server side
WRK_BUILD=${WRK_BUILD:-$HOME/llama.cpp-rpc/build}  # patched build, worker side
                                               # MUST be the same commit+flags on
                                               # both machines, else you get
                                               # 'send failed bytes_sent=0'

MODEL=${MODEL:-/path/to/models/deepseek-vision/UD-Q3_K_XL/DeepSeek-V4-Flash-Vision-Exp-UD-Q3_K_XL-00001-of-00004.gguf}
MMPROJ=${MMPROJ:-/path/to/models/deepseek-vision/mmproj-F16.gguf}
LOGDIR=${LOGDIR:-$HOME/llama-logs}
PORT=${PORT:-8241}
ALIAS=${ALIAS:-DeepSeek}

PAR=${PAR:-4}                  # 4 slots: 33.67 tok/s aggregate (3 slots: 31.51)
CTX=${CTX:-1048576}            # 4 x 262k. Context is nearly free: 25k->90k costs 7.6%
SPLIT=${SPLIT:-72,90}          # layers RPC,local - tune to your memory

mkdir -p "$LOGDIR"

# --------------------------------------------------------------- status/stop
if [ "${1:-}" = "--status" ]; then
  echo "=== server ==="
  pgrep -f llama-server >/dev/null && echo "  llama-server running" || echo "  llama-server DOWN"
  curl -s -m 3 "http://127.0.0.1:$PORT/health" 2>/dev/null || echo "  /health unreachable"
  echo
  echo "=== worker ==="
  ssh -o BatchMode=yes "$WRK_SSH" 'pgrep -f ggml-rpc-server >/dev/null && echo "  rpc-server running" || echo "  rpc-server DOWN"' </dev/null
  exit 0
fi

if [ "${1:-}" = "--stop" ]; then
  pkill -f llama-server 2>/dev/null
  screen -ls | grep -q dsprod && screen -S dsprod -X quit
  ssh -o BatchMode=yes "$WRK_SSH" 'pkill -f ggml-rpc-server' </dev/null
  echo "stopped"
  exit 0
fi

echo "=== 1/5  cleanup ==="
# CRITICAL: never launch llama-server via nohup, and never nest the launch chain.
# ssh -> screen -> script -> screen produces 'invalid device: RPC0' because macOS
# Local Network Privacy grants network access based on the launch chain.
pkill -f llama-server 2>/dev/null
for s in dsprod prodsrv moesrv sweep slotsrv; do
  screen -ls | grep -q "$s" && screen -S "$s" -X quit
done
sleep 3

echo "=== 2/5  restart worker ==="
# ggml-rpc-server accepts EXACTLY ONE connection per process lifetime.
# Probing the port with nc/telnet/lsof consumes it -> always start it fresh.
# 'script -q /dev/null' forces line-buffered output, otherwise the log stays empty.
# Build the worker with -DGGML_BLAS=OFF: BLAS measurably slows RPC decode.
ssh -o BatchMode=yes "$WRK_SSH" "
  pkill -f ggml-rpc-server; sleep 5
  mkdir -p $LOGDIR; rm -f $LOGDIR/worker.log
  nohup script -q /dev/null $WRK_BUILD/bin/ggml-rpc-server \
    --host 0.0.0.0 --port 50053 > $LOGDIR/worker.log 2>&1 &
  echo '  worker started'" </dev/null
sleep 20

echo "=== 3/5  start server ==="
# Parameter rationale - all measured, none assumed:
#   --cache-type-k/v q8_0  MLA KV is ~45 KiB/token; q8_0 makes 4x262k fit
#   --cache-prompt         prefix cache: 2nd session 8.9 s instead of 97.9 s (11x)
#   --cache-ram 0          REQUIRED with RPC + parallel>1 (llama.cpp #26128)
#   --tensor-split         119 GiB model across 2 nodes; one has only ~112 GiB
#   --load-mode none       no mlock - model streams from external SSD
#   --flash-attn on        required for MLA
#   -b/-ub 2048            larger batches improve MoE throughput
#   NO --cache-reuse       MLA rejects it ("not supported by this context")
#   NO draft model         acceptance only 0.27-0.46 -> 22-32% SLOWER
#   NO --slot-save-path    save/restore is a no-op with distributed KV:
#                          n_restored=0, full prefill still runs (measured)
# The build must contain the Metal MoE patch (ne21>=8 instead of >=32):
#   4 slots went from 12.6 tok/s to 33.67 tok/s, no more stalled slots.
rm -f "$LOGDIR/server.log"
screen -dmS dsprod /bin/bash -c "$BUILD/bin/llama-server \
  --model $MODEL \
  --mmproj $MMPROJ \
  --rpc $WRK_RPC \
  --host 0.0.0.0 --port $PORT \
  --ctx-size $CTX --parallel $PAR \
  --device RPC0,MTL0 --n-gpu-layers 999 --tensor-split $SPLIT \
  --flash-attn on --batch-size 2048 --ubatch-size 2048 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --load-mode none --cache-prompt --jinja --cache-ram 0 \
  --alias $ALIAS --temp 1.0 --top-p 1.0 --min-p 0.01 \
  > $LOGDIR/server.log 2>&1"

sleep 6
pgrep -f llama-server >/dev/null || {
  echo "  ERROR: server exited immediately"; head -20 "$LOGDIR/server.log"; exit 1; }

echo "=== 4/5  wait for load ==="
# /health returns exit code 0 even on HTTP 503 -> grep the body for "ok".
READY=0
for i in $(seq 1 60); do
  sleep 10
  pgrep -f llama-server >/dev/null || {
    echo "  ERROR: server crashed"
    grep -aiE "error|failed|invalid" "$LOGDIR/server.log" | head -5; exit 1; }
  if curl -s -m 3 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"'; then
    echo "  ready after $((i*10))s"; READY=1; break
  fi
done
[ $READY -eq 1 ] || { echo "  ERROR: timeout while loading"; exit 1; }

if [ "${1:-}" = "--no-warmup" ]; then
  echo "=== 5/5  warmup skipped ==="
else
  echo "=== 5/5  warm the prefix cache ==="
  # The first request after startup costs ~95 s of prompt processing (24k tokens).
  # After that the RAM prefix cache carries it: every further session ~9 s.
  # Slot save/restore could in principle persist this across restarts, but it
  # does not work with distributed KV (n_restored=0), so warm it instead - the
  # wait then happens BEFORE the first real request rather than during it.
  # Replace the prompt below with your own standard system prompt.
  python3 - "$PORT" "$ALIAS" <<'PY'
import json, sys, time, urllib.request
port, alias = sys.argv[1], sys.argv[2]
unit = ("[ctx] def handle(req, ctx):\n"
        "    res = pipeline.run(req.payload, timeout=30)\n"
        "    if res.status != 'ok':\n"
        "        return retry(req, backoff=2.0)\n"
        "    return Response(res.data)\n")
prompt = "You are an autonomous coding agent.\nProject state:\n" + unit * 500
body = {"model": alias,
        "messages": [{"role": "user", "content": prompt + "\nReply with: ready"}],
        "max_tokens": 8, "temperature": 1.0, "cache_prompt": True, "stream": False}
r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                           data=json.dumps(body).encode(),
                           headers={"Content-Type": "application/json"})
t0 = time.time()
try:
    with urllib.request.urlopen(r, timeout=900) as f:
        d = json.load(f)
    u = d.get("usage", {})
    print(f"  warmed: {u.get('prompt_tokens', 0)} tokens in {time.time()-t0:.0f}s")
    print("  -> subsequent sessions start in ~9 s instead of ~95 s")
except Exception as e:
    print(f"  warmup failed (non-critical): {e}")
PY
fi

echo
echo "=== DONE ==="
echo "  endpoint : http://0.0.0.0:$PORT/v1  (alias: $ALIAS)"
echo "  slots    : $PAR"
echo "  logs     : $LOGDIR/server.log | $LOGDIR/worker.log"
echo "  expect   : ~8.4 tok/s per slot with 4 concurrent (33.7 aggregate)"
