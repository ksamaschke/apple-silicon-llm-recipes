# Qwen3.8-Flash-Next with MTP speculative decoding

> The fastest of the three models measured here — and the one where speculative decoding actually pays

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Instructions](#instructions)
- [Measured results](#measured-results)
- [Deciding whether speculation helps](#deciding-whether-speculation-helps)
- [Troubleshooting](#troubleshooting)

---

## Overview

### Basic idea

Qwen3.8-Flash-Next ships with **MTP (multi-token prediction)** weights — a draft head trained as part of the model rather than a separate draft model. llama.cpp can use it to propose several tokens per step and verify them in one batch.

On this hardware it is the fastest configuration measured: **84.97 tok/s aggregate** at 4 concurrent slots, and prompt processing in the 500–600 tok/s range — roughly 2.5× what the DeepSeek and GLM setups reach.

Unlike the separate draft models tried with DeepSeek and GLM, the MTP head genuinely wins here. This recipe shows the configuration and the method for checking that on your own workload.

### What you'll accomplish

- Qwen3.8-Flash-Next serving an OpenAI-compatible API across two Macs
- 4 concurrent slots with MTP speculative decoding
- **84.97 tok/s aggregate**, up from 70.27 without speculation

### What to know before starting

- Comfortable with the macOS command line, `ssh`, `screen`, and CMake
- Read [shared/rpc-setup.md](../shared/rpc-setup.md) for the RPC failure modes

## Prerequisites

**Hardware**

- 2× Apple Silicon Mac with 128 GB unified memory (measured on M3 Ultra Mac Studio)
- Thunderbolt cable directly between the machines
- ~104 GiB disk for Q4_K_XL plus 23 GiB for the MTP weights

**Software**

- macOS 26.x, Xcode command line tools, CMake 3.14+
- A llama.cpp build with MTP support: `llama-server --help | grep spec-type`
- `hf` CLI with `HF_TOKEN` set

**Memory limit on both machines** (not persistent):

```shell
sudo sysctl iogpu.wired_limit_mb=114688
```

## Time & risk

- **Estimated time:** ~40 minutes plus download
- **Risk level:** Low
- **Rollback:** Delete the clone and model directory
- **Last updated:** 2026-09-04

## Instructions

### Step 1. Build llama.cpp

```shell
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp-qwen
cd ~/llama.cpp-qwen
cmake -B build -DGGML_METAL=ON -DGGML_RPC=ON -DGGML_BLAS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
./build/bin/llama-server --help | grep spec-type
```

Apply the `ne21 >= 8` MoE patch from [shared/metal-patches.md](../shared/metal-patches.md) — it matters for concurrency and for speculative verify batches, which are exactly the small expert batches the upstream threshold sends to the slow kernel.

Build both machines from the same commit with identical flags.

### Step 2. Download model and MTP weights

```shell
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
hf download unsloth/Qwen3.8-Flash-Next-GGUF \
  --include "UD-Q4_K_XL/*" --include "MTP/*" \
  --local-dir /path/to/models/qwen38-flash-next
```

> [!NOTE]
> The MTP file is a **draft head without its own `token_embd.weight`**. Loading it as a standalone model fails with `borrow_shared_tensor: this model is a draft head without its own 'token_embd.weight'`. It must be passed as `--model-draft` alongside its target.

### Step 3. Start the RPC worker

```shell
pkill -f ggml-rpc-server; sleep 5
script -q /dev/null ~/llama.cpp-qwen/build/bin/ggml-rpc-server \
  --host 0.0.0.0 --port 50053 > ~/worker.log 2>&1 &
```

One connection per process lifetime — restart it immediately before every server start, and never probe the port.

### Step 4. Start the server with MTP

```shell
screen -dmS qwen /bin/bash -c "~/llama.cpp-qwen/build/bin/llama-server \
  --model /path/to/models/qwen38-flash-next/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --model-draft /path/to/models/qwen38-flash-next/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
  --rpc 192.168.0.2:50053 \
  --host 0.0.0.0 --port 8241 \
  --ctx-size 262144 --parallel 4 \
  --device RPC0,MTL0 --n-gpu-layers 999 --tensor-split 50,50 \
  --flash-attn on --batch-size 2048 --ubatch-size 512 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --cache-prompt --jinja --cache-ram 0 --slots \
  --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.75 \
  --alias Qwen3.8-Flash-Next --temp 1.0 --top-p 1.0 --min-p 0.01 \
  > ~/qwen.log 2>&1"
```

**Speculation parameters:**

- `--spec-type draft-mtp` — use the MTP head as drafter
- `--spec-draft-n-max 2` — draft 2 tokens per step. Larger is not better; every rejected token is wasted work
- `--spec-draft-p-min 0.75` — only draft when confident, which keeps acceptance up
- `--slots` — exposes `/slots`, needed for the acceptance check below

Do not nest the launch chain; see the RPC guide.

### Step 5. Verify and check acceptance

```shell
until curl -s http://127.0.0.1:8241/health | grep -q '"ok"'; do sleep 10; done

curl -X POST http://127.0.0.1:8241/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3.8-Flash-Next","messages":[{"role":"user","content":"Write a Python function that merges two sorted lists."}],"max_tokens":400}'

grep -a "draft acceptance" ~/qwen.log | tail -5
```

You will see lines like:

```
draft acceptance = 0.88889 (32 accepted / 36 generated), mean len = 3.67
```

### Step 6. Warm the prefix cache

Send one throwaway request with your standard system prompt after startup, so the cold prompt-processing cost lands before the first real request rather than during it.

## Measured results

Measured on the reference hardware, Q4_K_XL, distributed across two Macs.

**Without speculation:**

- **1 slot:** 17.44 tok/s
- **4 slots:** 70.27 tok/s aggregate
- **Prompt processing:** 603.5 tok/s best
- **Prefix cache hit from turn 2:** 79.4%

**With MTP speculation:**

- **1 slot:** **21.29 tok/s** — 22% faster
- **4 slots:** **84.97 tok/s aggregate** — 21% faster
- **Prompt processing:** 511.9 tok/s best
- **Draft acceptance:** 0.398 overall, mean draft length 2.18; individual tasks reached 0.889

**Comparison across all three models on the same two-machine setup** (4 concurrent slots, aggregate):

- **Qwen3.8-Flash-Next Q4_K_XL + MTP:** 84.97 tok/s
- **Qwen3.8-Flash-Next Q4_K_XL, no speculation:** 70.27 tok/s
- **GLM-5.3-Flash IQ3_XXS:** 43.88 tok/s
- **DeepSeek-V4-Flash-Vision Q3_K_XL:** 33.67 tok/s

Qwen is roughly 2.5× the DeepSeek configuration — at a *higher* quantization. Prompt processing shows the same ordering: ~600 tok/s versus ~240 (GLM) and ~290 (DeepSeek).

## Deciding whether speculation helps

Speculative decoding is a bet: extra compute drafting tokens, repaid only if the drafts are accepted. Same hardware, same harness, three models:

- **Qwen3.8-Flash-Next + MTP head:** acceptance 0.398 → **21% faster**
- **GLM-5.3-Flash + DFlash2 draft model:** acceptance 0.32 → **14% slower**
- **DeepSeek-V4-Flash-Vision + DSpark draft model:** acceptance 0.27–0.46 → **22–32% slower**

Acceptance rate alone does not decide it — Qwen at 0.398 wins while DeepSeek at up to 0.46 loses. What differs is the **cost of drafting**: an MTP head sharing the target model's weights is far cheaper to run than a separate draft model that must be loaded and executed independently. A cheap drafter pays off at a much lower acceptance rate.

**Practical rule:** with an MTP head, try it and measure. With a separate draft model, expect to need acceptance well above 0.5 before it pays. Either way, run your own workload and compare timed runs with and without `--spec-type` rather than trusting the acceptance number in isolation.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `borrow_shared_tensor: this model is a draft head without its own 'token_embd.weight'` | MTP file loaded as `--model` | Pass it as `--model-draft` alongside the target model |
| `--spec-type` not recognized | Build predates MTP support | Update llama.cpp and rebuild |
| `draft acceptance` absent from the log | Speculation not active | Confirm `--spec-type draft-mtp` and that the MTP weights loaded |
| Slower with speculation | Acceptance below break-even for your drafter's cost | Lower `--spec-draft-n-max`, raise `--spec-draft-p-min`, or disable |
| `Remote RPC server crashed or returned malformed response` | Worker died | Restart worker; verify both builds match exactly |
| `invalid device: RPC0` | Nested launch chain | One flat `screen -dmS ... bash -c` over SSH |
| `Failed to connect to <ip>:50053` | Worker's one connection consumed | Restart worker; never probe the port |
| `curl` succeeds but model not ready | `/health` returns 503 with exit code 0 | Grep the body for `"ok"` |
