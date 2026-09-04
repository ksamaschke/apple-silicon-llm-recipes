# DeepSeek-V4-Flash-Vision distributed across two Macs

> Serve a 119 GiB MoE model that fits on neither machine alone, using llama.cpp RPC over Thunderbolt

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Instructions](#instructions)
- [Measured results](#measured-results)
- [What does not work](#what-does-not-work)
- [Troubleshooting](#troubleshooting)

---

## Overview

### Basic idea

DeepSeek-V4-Flash-Vision-Exp at `UD-Q3_K_XL` is 119 GiB. A 128 GB Mac Studio has roughly 112 GiB usable for GPU work after `iogpu.wired_limit_mb`, and the KV cache for a useful context adds tens of GiB on top. The model does not fit.

llama.cpp's RPC backend splits the model's layers across two machines. One Mac runs `llama-server` and holds part of the weights; the other runs `ggml-rpc-server` and holds the rest. Over Thunderbolt 5 with RDMA, the interconnect is fast enough that this is practical rather than merely possible.

This recipe covers the working configuration, the Metal patch that more than doubled throughput, and the settings that turned out to be measured losses.

### What you'll accomplish

- A 119 GiB vision-capable MoE model serving an OpenAI-compatible API
- 4 concurrent slots at 262k context each
- **33.7 tok/s aggregate** decode throughput (8.4 tok/s per slot with 4 agents running)
- Context length that costs almost nothing: 25k → 90k is a 7.6% throughput drop

### What to know before starting

- Comfortable with the macOS command line, `ssh`, and `screen`
- Basic understanding of llama.cpp server flags
- Familiarity with GGUF quantization names

## Prerequisites

**Hardware**

- 2× Apple Silicon Mac with 128 GB unified memory (measured on M3 Ultra Mac Studio)
- **Thunderbolt cable directly between the two machines** — this matters enormously; see [rpc-setup.md](../shared/rpc-setup.md)
- ~120 GiB free disk on the machine hosting the model files (external NVMe is fine)

**Software**

- macOS 26.x on both machines
- Xcode command line tools, CMake 3.14+
- `hf` CLI (`pip install huggingface_hub[cli]`) with a token — unauthenticated downloads stall (see Troubleshooting)
- Passwordless SSH from the server machine to the worker machine

**Memory limit on both machines:**

```shell
sudo sysctl iogpu.wired_limit_mb=114688
```

This is not persistent across reboots. Add it to a launch daemon if you want it to survive.

## Time & risk

- **Estimated time:** ~45 minutes plus model download (119 GiB)
- **Risk level:** Low — nothing is installed system-wide; the patch applies to your own clone
- **Rollback:** Delete the llama.cpp clone and the model directory
- **Last updated:** 2026-09-04

## Instructions

### Step 1. Clone and patch llama.cpp

Clone upstream and apply the Metal MoE patch. Without it, throughput at 4 concurrent slots collapses — see [Measured results](#measured-results).

```shell
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp-rpc
cd ~/llama.cpp-rpc
```

Edit `ggml/src/ggml-metal/ggml-metal-common.cpp`:

```diff
 bool ggml_metal_op_mul_mat_id_use_mm(const struct ggml_tensor * op, bool has_simdgroup_mm) {
     const int64_t ne00 = op->src[0]->ne[0];
     const int64_t ne21 = op->src[2]->ne[1];

-    return has_simdgroup_mm && ne00 >= 64 && ne21 >= 32;
+    return has_simdgroup_mm && ne00 >= 64 && ne21 >= 8;
 }
```

**Why:** `ne21` is the number of tokens routed to an expert in one batch. With 4 concurrent slots, that number is typically 4–8. The upstream threshold of 32 sends those batches to the slow vector kernel; lowering it to 8 lets them reach the simdgroup matrix kernel. The CUDA backend made the analogous change in [ggml-org#27342](https://github.com/ggml-org/llama.cpp/pull/27342).

Full patch set including MLA/lightning-indexer fixes: [shared/metal-patches.md](../shared/metal-patches.md).

### Step 2. Build on both machines

Build the **same commit with the same flags** on both machines. Mismatched builds fail at runtime with `send failed bytes_sent=0`.

```shell
cmake -B build -DGGML_METAL=ON -DGGML_RPC=ON -DGGML_BLAS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
```

> [!IMPORTANT]
> `-DGGML_BLAS=OFF` on the worker. With BLAS enabled, RPC decode throughput drops substantially.

### Step 3. Download the model

```shell
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
hf download unsloth/DeepSeek-V4-Flash-Vision-Exp-GGUF \
  --include "UD-Q3_K_XL/*" \
  --local-dir /path/to/models/deepseek-vision
```

Set `HF_TOKEN`. Unauthenticated downloads from CloudFront routinely stall with sockets stuck in `CLOSE_WAIT` — the process stays alive and transfers nothing.

### Step 4. Start the RPC worker

On the **worker** machine:

```shell
pkill -f ggml-rpc-server; sleep 5
script -q /dev/null ~/llama.cpp-rpc/build/bin/ggml-rpc-server \
  --host 0.0.0.0 --port 50053 > ~/worker.log 2>&1 &
```

> [!WARNING]
> `ggml-rpc-server` accepts **exactly one connection per process lifetime**. Testing the port with `nc`, `telnet`, or `lsof` consumes it, and the real server then cannot connect. Always restart the worker immediately before starting the server.

`script -q /dev/null` forces line-buffered output; without it the log stays empty.

### Step 5. Start the server

On the **server** machine, in a single flat `screen` session:

```shell
screen -dmS dsprod /bin/bash -c "~/llama.cpp-rpc/build/bin/llama-server \
  --model /path/to/models/deepseek-vision/UD-Q3_K_XL/DeepSeek-V4-Flash-Vision-Exp-UD-Q3_K_XL-00001-of-00004.gguf \
  --mmproj /path/to/models/deepseek-vision/mmproj-F16.gguf \
  --rpc 192.168.0.2:50053 \
  --host 0.0.0.0 --port 8241 \
  --ctx-size 1048576 --parallel 4 \
  --device RPC0,MTL0 --n-gpu-layers 999 --tensor-split 72,90 \
  --flash-attn on --batch-size 2048 --ubatch-size 2048 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --load-mode none --cache-prompt --jinja --cache-ram 0 \
  --alias DeepSeek --temp 1.0 --top-p 1.0 --min-p 0.01 \
  > ~/server.log 2>&1"
```

> [!CAUTION]
> **Do not nest the launch.** `ssh → screen → script → screen` produces `invalid device: RPC0`. macOS Local Network Privacy grants network permission based on the process launch chain; a nested launch loses it. One `screen -dmS` with `bash -c` directly over SSH works.

**Key parameters:**

- `--rpc 192.168.0.2:50053` — the **Thunderbolt** address of the worker, not its LAN address
- `--tensor-split 72,90` — 72 layers on the RPC worker, 90 locally; tune to your memory
- `--cache-ram 0` — **required** with RPC and `--parallel > 1` ([ggml-org#26128](https://github.com/ggml-org/llama.cpp/issues/26128))
- `--cache-type-k/v q8_0` — MLA KV is ~45 KiB/token; q8_0 makes 4×262k fit
- `--load-mode none` — no mlock; the model streams from external SSD

### Step 6. Wait for readiness

```shell
until curl -s http://127.0.0.1:8241/health | grep -q '"ok"'; do sleep 10; done
```

Check the **body** for `"ok"`. `curl` returns exit code 0 even on HTTP 503 while the model is still loading.

Loading takes ~100 s from a warm page cache.

### Step 7. Warm the prefix cache

The first request pays the full prompt-processing cost. For a 24k-token system prompt that is **~95 seconds**. Every subsequent session with the same prefix costs **~9 seconds**.

Send one throwaway request with your standard system prompt right after startup, so that cost lands before the first real request rather than in the middle of an agent run:

```shell
curl -s -X POST http://127.0.0.1:8241/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"DeepSeek","messages":[{"role":"user","content":"<your system prompt>"}],
       "max_tokens":8,"cache_prompt":true}' > /dev/null
```

A complete start script that does all of this is in [scripts/start-deepseek.sh](scripts/start-deepseek.sh).

## Measured results

All numbers: 4 shards `UD-Q3_K_XL`, q8_0 KV, no draft model, barrier-synchronized clients so every slot decodes in the same window.

**Throughput by concurrency (45k context):**

- **1 slot:** 18.98 tok/s
- **2 slots:** 30.51 tok/s aggregate (15.25 per slot)
- **3 slots:** 31.51 tok/s aggregate (10.50 per slot)
- **4 slots:** 33.67 tok/s aggregate (8.42 per slot)

The curve is not a clean saturation: 2→3 adds only 3%, but 3→4 adds 7%. Even batch sizes map better onto the simdgroup kernels.

**Throughput by context (4 slots):**

- **25k:** 34.35 tok/s (8.59 per slot)
- **45k:** 33.67 tok/s (8.42 per slot)
- **90k:** 31.73 tok/s (7.93 per slot)

**Context is nearly free.** More than tripling it costs 7.6%. Plan slots by memory, not by speed.

**Effect of the Metal patch (4 slots, 45k):**

- **Without patch:** 12.6 tok/s aggregate — one slot stuck at 3.14 tok/s dragging the rest
- **With patch:** 33.67 tok/s aggregate

**Prompt processing:** ~217–292 tok/s, roughly flat with prompt size. A cold 24k prompt takes ~95 s; with the prefix cache warm, a new session with the same prefix takes ~9 s.

## What does not work

Measured, not assumed:

**Speculative decoding (DSpark / draft model).** Acceptance rate was 0.27–0.46 across configurations; break-even needs roughly 0.8. Result: 14.36 tok/s vs. 18.47 without at 1 slot. With `--spec-draft-n-max 3` instead of 5, still 16.17 vs. 19.40. **22–32% slower.** Do not use it with this model.

**`--cache-reuse`.** The MLA context rejects it: `cache_reuse is not supported by this context, it will be disabled`. Harmless but pointless.

**`--slot-save-path` for persisting the prefix cache.** Save and restore both *report success* — and do nothing. `n_saved: 0`, `n_restored: 0`, a 12 MB file for 24k tokens, and the next request re-processes all 24,024 tokens (95.2 s cold vs. 107.5 s "after restore"). The distributed MLA KV is not captured. Warm the cache instead.

**LAN instead of Thunderbolt.** Not benchmarked head-to-head here, but the Thunderbolt bridge is the entire basis of these numbers — RDMA over Thunderbolt is what makes layer-split decode viable.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `invalid device: RPC0` | Nested launch chain broke macOS Local Network Privacy | Launch with one `screen -dmS ... bash -c` directly over SSH; never `ssh → screen → script → screen` |
| `Failed to connect to <ip>:50053` | Worker's single connection already consumed, or wrong address | Restart the worker; never probe the port with `nc`/`telnet`; use the Thunderbolt IP |
| `send failed bytes_sent=0` | Server and worker built from different commits/flags | Rebuild both from the same commit with identical CMake flags |
| Worker log stays empty | Block buffering | Wrap in `script -q /dev/null` |
| One slot at ~3 tok/s while others are fine | Metal MoE threshold not patched | Apply the `ne21 >= 8` patch and rebuild both sides |
| Throughput far below expectation with `--parallel > 1` | `--cache-ram` not zero | Set `--cache-ram 0` (required with RPC) |
| Download stalls, process alive, zero bytes | CloudFront closed the sockets (`CLOSE_WAIT`), client never times out | Set `HF_TOKEN` and restart the download; it resumes |
| `curl` reports success but model not ready | `/health` returns 503 with exit code 0 | Grep the body for `"ok"` |
| First request takes ~95 s | Cold prefix cache | Warm it at startup (Step 7) |

> [!NOTE]
> `iogpu.wired_limit_mb` resets on reboot. If the server suddenly fails to allocate, check it first.
