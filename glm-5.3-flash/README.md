# GLM-5.3-Flash on a single Mac

> A hybrid KDA/MLA model whose KV cache is small enough that long contexts and multiple slots actually fit in 128 GB

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Instructions](#instructions)
- [Choosing a quantization](#choosing-a-quantization)
- [Troubleshooting](#troubleshooting)

---

## Overview

### Basic idea

GLM-5.3-Flash is a MoE model with an unusual attention layout: of its 46 layers, only **12 use MLA** (multi-head latent attention, where the KV cache grows with context). The other **34 are KDA** — a linear-attention variant with a **constant-size state** regardless of context length.

The practical consequence is large. Per-token KV cost:

- **DeepSeek-V4-Flash-Vision:** ~45 KiB/token (43 MLA layers)
- **GLM-5.3-Flash:** ~7.2 KiB/token (12 MLA layers)

That is roughly a sixth. Where DeepSeek needs two machines to hold weights plus KV, GLM fits a useful quantization plus four long-context slots on a single 128 GB Mac.

### What you'll accomplish

- GLM-5.3-Flash serving an OpenAI-compatible API on one machine
- 4 concurrent slots at 128k context each
- A model that stays usable at aggressive quantizations (see [Choosing a quantization](#choosing-a-quantization))

### What to know before starting

- Comfortable with the macOS command line and CMake
- Basic understanding of llama.cpp server flags

## Prerequisites

**Hardware**

- Apple Silicon Mac with 128 GB unified memory (measured on M3 Ultra Mac Studio)
- Disk for the GGUF — 48 GiB (IQ2_XXS) to 186 GiB (Q4_K_XL) depending on quantization

**Software**

- macOS 26.x, Xcode command line tools, CMake 3.14+
- `hf` CLI with `HF_TOKEN` set

**Memory limit** (not persistent across reboots):

```shell
sudo sysctl iogpu.wired_limit_mb=114688
```

## Time & risk

- **Estimated time:** ~30 minutes plus model download
- **Risk level:** Low
- **Rollback:** Delete the clone and model directory
- **Last updated:** 2026-09-04

## Instructions

### Step 1. Clone and patch llama.cpp

GLM-5.3-Flash needs the **lightning-indexer head-count patch** to use the Metal path. Upstream hardcodes the head count; DeepSeek-V4 has 64 heads, GLM-5.3-Flash has 32, and only one of them passes the support check.

```shell
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp-glm
cd ~/llama.cpp-glm
```

Apply patches 1 and 2 from [shared/metal-patches.md](../shared/metal-patches.md):

- **`ne21 >= 8`** — MoE expert-batch threshold, matters as soon as you run more than one slot
- **Runtime head count for the lightning indexer** — required for GLM to run the op on Metal at all

### Step 2. Build

```shell
cmake -B build -DGGML_METAL=ON -DGGML_RPC=ON -DGGML_BLAS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
```

### Step 3. Download a quantization

```shell
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
hf download unsloth/GLM-5.3-Flash-GGUF \
  --include "UD-Q2_K_XL/*" \
  --local-dir /path/to/models/glm53-flash
```

Set `HF_TOKEN` — unauthenticated downloads stall with sockets in `CLOSE_WAIT` while the process stays alive.

See [Choosing a quantization](#choosing-a-quantization) for which one to pick.

### Step 4. Start the server

```shell
screen -dmS glm /bin/bash -c "~/llama.cpp-glm/build/bin/llama-server \
  --model /path/to/models/glm53-flash/UD-Q2_K_XL/GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00003.gguf \
  --mmproj /path/to/models/glm53-flash/mmproj-F16.gguf \
  --host 0.0.0.0 --port 8242 \
  --ctx-size 524288 --parallel 4 \
  --n-gpu-layers 999 \
  --flash-attn on --batch-size 2048 --ubatch-size 2048 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --cache-prompt --jinja \
  --alias GLM --temp 1.0 --top-p 1.0 --min-p 0.01 \
  > ~/glm.log 2>&1"
```

`--ctx-size 524288` with `--parallel 4` gives four 128k slots.

**Memory arithmetic** (Q2_K_XL, 128 GB machine with 112 GiB usable):

- Weights: 101 GiB
- KV: 4 slots × 128k tokens × ~7.2 KiB ≈ 3.6 GiB at q8_0
- Total: ~105 GiB — fits

The same arithmetic for DeepSeek-V4 would need ~22 GiB of KV for the same slot configuration, on top of a larger model.

### Step 5. Wait and verify

```shell
until curl -s http://127.0.0.1:8242/health | grep -q '"ok"'; do sleep 10; done

curl -X POST http://127.0.0.1:8242/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"GLM","messages":[{"role":"user","content":"Explain KDA attention in two sentences."}],"max_tokens":200}'
```

Grep the health **body** for `"ok"` — `curl` exits 0 on HTTP 503 while loading.

### Step 6. Warm the prefix cache

The first request with a long system prompt pays full prompt-processing cost. Send one throwaway request with your standard prompt after startup so that cost does not land in the middle of real work. See the [DeepSeek recipe, Step 7](../deepseek-v4-distributed/README.md#step-7-warm-the-prefix-cache) for the pattern.

## Choosing a quantization

Unsloth publishes KL-divergence figures against the unquantized model — higher means closer to the original:

- **Q3_K_XL:** 86.3%
- **IQ3_XXS:** 81.6%
- **IQ2_XXS:** 76.3%

GLM-5.3-Flash is **not** QAT-trained, so quality degrades gradually rather than falling off a cliff. Community reports describe IQ3 and even IQ2 as still useful for agentic and coding work — but note that "useful" here is a subjective report, not a benchmark.

**File sizes and what fits on a 128 GB Mac** (112 GiB usable, 4 slots × 128k at q8_0 KV ≈ 3.6 GiB):

- **IQ2_XXS** — 48 GiB — fits comfortably, room for far more context
- **UD-Q2_K_XL** — 101 GiB — fits (~105 GiB total)
- **IQ3_XXS** — 112 GiB — does **not** fit with 4 long slots
- **Q3_K_XL** — 137 GiB — does not fit on one machine
- **IQ4_XS** — 146 GiB — two machines
- **Q4_K_XL** — 186 GiB — two machines

For a single 128 GB Mac, **Q2_K_XL is the largest that fits with four long-context slots**. If you want IQ3 or better, either reduce slots/context or go distributed — see the [DeepSeek recipe](../deepseek-v4-distributed/) for the two-machine pattern, which applies unchanged to GLM.

> [!NOTE]
> Throughput numbers for GLM on this setup are not yet included here. The DeepSeek recipe's numbers were measured; adding unmeasured GLM figures would undermine the point of these recipes. They will be added once measured.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `GGML_OP_LIGHTNING_INDEXER` unsupported / falls back to CPU | Head-count patch not applied | Apply patch 2 from [shared/metal-patches.md](../shared/metal-patches.md) and rebuild |
| Poor throughput with `--parallel > 1` | MoE threshold not patched | Apply the `ne21 >= 8` patch |
| Out of memory on load | Quantization too large for the slot configuration | Use Q2_K_XL or reduce `--ctx-size` / `--parallel` |
| Download stalls at zero bytes, process alive | Unauthenticated HF download, sockets in `CLOSE_WAIT` | Set `HF_TOKEN`, restart — it resumes |
| `curl` succeeds but model not ready | `/health` returns 503 with exit code 0 | Grep the body for `"ok"` |
| First request very slow | Cold prefix cache | Warm it at startup (Step 6) |
