# GLM-5.3-Flash distributed across two Macs

> A hybrid KDA/MLA model with a small KV cache — and the Metal kernel patch without which it does not run at all

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Instructions](#instructions)
- [Measured results](#measured-results)
- [Choosing a quantization](#choosing-a-quantization)
- [Troubleshooting](#troubleshooting)

---

## Overview

### Basic idea

GLM-5.3-Flash has an unusual attention layout: of its 46 layers, only **12 use MLA** (where the KV cache grows with context). The other **34 are KDA** — a linear-attention variant with a **constant-size state** regardless of context length.

Per-token KV cost is therefore about a sixth of a conventional MLA model:

- **DeepSeek-V4-Flash-Vision:** ~45 KiB/token (43 MLA layers)
- **GLM-5.3-Flash:** ~7.2 KiB/token (12 MLA layers)

This recipe runs it **distributed across two Macs** over Thunderbolt RPC, the same topology as the [DeepSeek recipe](../deepseek-v4-distributed/). The small KV is what makes long contexts cheap here; the weights still want two machines at usable quantizations.

**The critical part is a Metal kernel patch.** Without it GLM-5.3-Flash does not run on Metal at all — it aborts with `unsupported op 'LIGHTNING_INDEXER'`.

### What you'll accomplish

- GLM-5.3-Flash serving an OpenAI-compatible API across two Macs
- **43.88 tok/s aggregate** at 4 concurrent slots (measured)
- A working lightning-indexer patch that keeps DeepSeek working too

### What to know before starting

- Comfortable with the macOS command line, `ssh`, `screen`, and CMake
- Read [shared/rpc-setup.md](../shared/rpc-setup.md) first — the RPC failure modes are the same as for DeepSeek

## Prerequisites

**Hardware**

- 2× Apple Silicon Mac with 128 GB unified memory (measured on M4 Max)
- Thunderbolt cable directly between the machines
- Disk for the GGUF — 112 GiB for IQ3_XXS

**Software**

- macOS 26.x, Xcode command line tools, CMake 3.14+
- `hf` CLI with `HF_TOKEN` set
- Passwordless SSH from server to worker

**Memory limit on both machines** (not persistent across reboots):

```shell
sudo sysctl iogpu.wired_limit_mb=114688
```

## Time & risk

- **Estimated time:** ~45 minutes plus model download
- **Risk level:** Low — the patch applies to your own clone
- **Rollback:** Delete the clone and model directory
- **Last updated:** 2026-09-04

## Instructions

### Step 1. Patch the lightning indexer

This is not optional. Upstream hardcodes the lightning-indexer head count as a compile-time constant. DeepSeek-V4 has 64 heads, GLM-5.3-Flash has 32 — with the constant fixed at one value, the other model fails `supports_op` and the run aborts.

The patch has **four parts**, and missing any one of them still fails:

1. `supports_op`: test divisibility instead of equality (`ne[1] % NHPTG == 0`)
2. Kernel `fa.metal`: loop bound from `args.n_head` at runtime instead of `constexpr NH`
3. Host: add `n_head` to the kargs struct and populate it
4. **`ggml-metal-ops.cpp`: a second hard `GGML_ASSERT` on the head count** — easy to miss, and the reason a first attempt still crashed

Full diffs: [shared/metal-patches.md](../shared/metal-patches.md), patch 2.

Apply the `ne21 >= 8` MoE patch (patch 1) as well — it is what makes concurrency worthwhile on this hardware.

### Step 2. Build on both machines

Same commit, same flags, both machines. Mismatched builds fail at runtime with `send failed bytes_sent=0`.

```shell
cmake -B build -DGGML_METAL=ON -DGGML_RPC=ON -DGGML_BLAS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
```

Verify the model loads without `unsupported op`:

```shell
grep -a "LIGHTNING_INDEXER" ~/glm.log
```

### Step 3. Download the model

```shell
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
hf download unsloth/GLM-5.3-Flash-GGUF \
  --include "UD-IQ3_XXS/*" \
  --local-dir /path/to/models/glm53-flash
```

Set `HF_TOKEN` — unauthenticated downloads stall with sockets in `CLOSE_WAIT` while the process stays alive and transfers nothing.

### Step 4. Start the RPC worker

On the **worker** machine:

```shell
pkill -f ggml-rpc-server; sleep 5
script -q /dev/null ~/llama.cpp-glm/build/bin/ggml-rpc-server \
  --host 0.0.0.0 --port 50053 > ~/worker.log 2>&1 &
```

> [!WARNING]
> `ggml-rpc-server` accepts **one connection per process lifetime**. Probing the port with `nc`/`telnet`/`lsof` consumes it. Always restart the worker immediately before starting the server.

### Step 5. Start the server

```shell
screen -dmS glm /bin/bash -c "~/llama.cpp-glm/build/bin/llama-server \
  --model /path/to/models/glm53-flash/UD-IQ3_XXS/GLM-5.3-Flash-UD-IQ3_XXS-00001-of-00004.gguf \
  --mmproj /path/to/models/glm53-flash/mmproj-F16.gguf \
  --rpc 192.168.0.2:50053 \
  --host 0.0.0.0 --port 8241 \
  --ctx-size 262144 --parallel 4 \
  --device RPC0,MTL0 --n-gpu-layers 999 --tensor-split 50,50 \
  --flash-attn on --batch-size 2048 --ubatch-size 2048 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --cache-prompt --jinja --cache-ram 0 \
  --alias GLM-5.3-Flash --temp 1.0 --top-p 1.0 --min-p 0.01 \
  > ~/glm.log 2>&1"
```

> [!CAUTION]
> Do not nest the launch chain. `ssh → screen → script → screen` yields `invalid device: RPC0` because macOS Local Network Privacy grants network access based on the launch chain. One flat `screen -dmS` with `bash -c` over SSH works.

`--cache-ram 0` is required with RPC and `--parallel > 1` ([ggml-org#26128](https://github.com/ggml-org/llama.cpp/issues/26128)).

### Step 6. Wait for readiness and warm the cache

```shell
until curl -s http://127.0.0.1:8241/health | grep -q '"ok"'; do sleep 10; done
```

Grep the **body** for `"ok"` — `curl` exits 0 on HTTP 503 while loading.

Then send one throwaway request with your standard system prompt, so the cold prompt-processing cost lands before the first real request. See [DeepSeek recipe, Step 7](../deepseek-v4-distributed/README.md#step-7-warm-the-prefix-cache).

## Measured results

Measured on the reference hardware, IQ3_XXS, distributed across two Macs.

**Without speculative decoding:**

- **1 slot:** 10.99 tok/s
- **4 slots:** **43.88 tok/s aggregate**, 107.7 s wall for the parallel run
- **Prompt processing:** 241.8 tok/s best, 168.2 tok/s at 24.8k tokens
- **Prefix cache hit from turn 2:** 79.4%

**With the DFlash2 draft model:**

- **1 slot:** 9.19 tok/s — *slower*
- **4 slots:** 37.88 tok/s aggregate — *slower*
- **Prompt processing:** 218.8 tok/s best, 104.0 tok/s at 24.8k
- **Draft acceptance:** 0.177 overall

**Do not use DFlash2 speculation with this model.** At 0.177 acceptance the drafting overhead outweighs the savings: 16% lower single-slot throughput, 14% lower aggregate. Break-even needs roughly 0.8.

**For comparison** (same hardware, same two-machine topology): DeepSeek-V4-Flash-Vision at Q3_K_XL reaches 33.67 tok/s aggregate at 4 slots. GLM at IQ3_XXS reaches 43.88 — about 30% higher, at a smaller quantization.

## Choosing a quantization

Unsloth's KL-divergence figures against the unquantized model (higher is closer):

- **Q3_K_XL:** 86.3%
- **IQ3_XXS:** 81.6%
- **IQ2_XXS:** 76.3%

GLM-5.3-Flash is **not** QAT-trained, so quality degrades gradually rather than falling off a cliff — unlike DeepSeek-V4, which is MXFP4 QAT and where sub-4-bit is risky.

**File sizes:**

- **IQ2_XXS** — 48 GiB
- **UD-Q2_K_XL** — 101 GiB
- **IQ3_XXS** — 112 GiB (measured configuration)
- **Q3_K_XL** — 137 GiB
- **IQ4_XS** — 146 GiB
- **Q4_K_XL** — 186 GiB

With two 128 GB machines, IQ3_XXS through Q4_K_XL are all reachable. The measurements above use IQ3_XXS.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `unsupported op 'LIGHTNING_INDEXER'`, run aborts | Head-count patch missing or incomplete | Apply **all four parts** of patch 2 — including the second `GGML_ASSERT` in `ggml-metal-ops.cpp` |
| Crash after patching, still on the indexer | Only the first assert was patched | Patch `ggml-metal-ops.cpp` too |
| `invalid device: RPC0` | Nested launch chain broke Local Network Privacy | One flat `screen -dmS ... bash -c` over SSH |
| `Failed to connect to <ip>:50053` | Worker's single connection consumed | Restart the worker; never probe the port |
| `send failed bytes_sent=0` | Build mismatch between machines | Rebuild both from the same commit and flags |
| `Remote RPC server crashed or returned malformed response` | Worker died — check its log | Restart worker; verify both builds match |
| Slower with DFlash2 than without | Acceptance 0.177, far below break-even | Drop the draft model |
| Poor throughput with `--parallel > 1` | `ne21` MoE threshold not patched, or `--cache-ram` not zero | Apply patch 1; set `--cache-ram 0` |
| Download stalls at zero bytes, process alive | Unauthenticated HF download, `CLOSE_WAIT` | Set `HF_TOKEN` and restart — it resumes |
