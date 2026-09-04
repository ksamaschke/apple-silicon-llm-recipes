# Qwen3.6-27B with MTP speculative decoding

> Where speculative decoding pays off — and how to tell before you commit to it

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Instructions](#instructions)
- [When speculative decoding helps](#when-speculative-decoding-helps)
- [Troubleshooting](#troubleshooting)

---

## Overview

### Basic idea

Qwen3.6-27B is a dense model small enough to fit comfortably on a single Apple Silicon Mac, and it ships with **MTP (multi-token prediction)** weights. MTP lets llama.cpp draft several tokens per step and verify them in one batch — a real speedup when the drafts are usually accepted.

That last condition is the whole story. This recipe covers the working configuration and, just as importantly, how to check whether speculative decoding is actually helping you rather than costing you.

### What you'll accomplish

- Qwen3.6-27B serving an OpenAI-compatible API with MTP speculative decoding
- 4 concurrent slots
- A method for deciding whether to keep speculative decoding on

### What to know before starting

- Comfortable with the macOS command line and CMake
- Basic understanding of llama.cpp server flags

## Prerequisites

**Hardware**

- Apple Silicon Mac; 64 GB is enough for Q4/Q6 with generous context
- Disk for the GGUF and MTP weights

**Software**

- macOS 26.x, Xcode command line tools, CMake 3.14+
- `hf` CLI with `HF_TOKEN` set

## Time & risk

- **Estimated time:** ~25 minutes plus download
- **Risk level:** Low
- **Rollback:** Delete the clone and model directory
- **Last updated:** 2026-09-04

## Instructions

### Step 1. Build llama.cpp

MTP support needs a build that includes it; check that your commit has `--spec-type draft-mtp`.

```shell
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp-qwen
cd ~/llama.cpp-qwen
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
./build/bin/llama-server --help | grep spec-type
```

If you also run this model with several concurrent slots, apply the `ne21 >= 8` MoE patch from [shared/metal-patches.md](../shared/metal-patches.md). It is a no-op for dense models but harmless, and matters if you later serve an MoE from the same build.

### Step 2. Download the model

```shell
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
hf download unsloth/Qwen3.6-27B-MTP-GGUF \
  --include "UD-Q6_K_XL/*" \
  --local-dir /path/to/models/qwen36-mtp
```

### Step 3. Start the server with MTP

```shell
screen -dmS qwen /bin/bash -c "~/llama.cpp-qwen/build/bin/llama-server \
  --model /path/to/models/qwen36-mtp/UD-Q6_K_XL/Qwen3.6-27B-UD-Q6_K_XL-00001-of-00002.gguf \
  --host 0.0.0.0 --port 8221 \
  --alias qwen36-mtp \
  --ctx-size 524288 --parallel 4 \
  --n-gpu-layers 99 \
  --flash-attn on \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --batch-size 2048 --ubatch-size 512 \
  --cache-prompt --cache-ram 16384 \
  --kv-unified --slot-prompt-similarity 0.1 \
  --slots \
  --spec-type draft-mtp \
  --spec-draft-n-max 2 \
  --spec-draft-p-min 0.75 \
  > ~/qwen.log 2>&1"
```

**Speculative decoding parameters:**

- `--spec-type draft-mtp` — use the model's MTP head as the drafter
- `--spec-draft-n-max 2` — draft 2 tokens per step. Larger is not better; upstream documentation notes larger values are often slower, and each rejected token is wasted work
- `--spec-draft-p-min 0.75` — only draft when the head is confident, which keeps the acceptance rate up

**Other notable flags:**

- `--kv-unified` — shared KV allocation across slots
- `--slot-prompt-similarity 0.1` — routes a request to a slot whose prompt already matches, improving cache hits
- `--slots` — exposes `/slots` for inspection, which you need for the check below

### Step 4. Verify and check the acceptance rate

```shell
until curl -s http://127.0.0.1:8221/health | grep -q '"ok"'; do sleep 5; done

curl -X POST http://127.0.0.1:8221/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen36-mtp","messages":[{"role":"user","content":"Write a Python function that merges two sorted lists."}],"max_tokens":400}'
```

Then look for the acceptance rate in the server log:

```shell
grep -a "draft acceptance" ~/qwen.log | tail -5
```

You will see lines like:

```
draft acceptance = 0.78261 (144/184)  mean len 2.61
```

**This number decides everything.** See below.

## When speculative decoding helps

Speculative decoding is a bet: you spend extra compute drafting tokens, and win only if most drafts are accepted. The break-even acceptance rate is roughly **0.8** — below that, the overhead of drafting and verifying rejected tokens outweighs the savings.

**A worked counter-example.** On DeepSeek-V4-Flash-Vision with a separate draft model, measured acceptance was **0.27–0.46** depending on configuration. The result:

- 1 slot, no speculation: **18.47 tok/s**
- 1 slot, speculation: **14.36 tok/s** — 22% slower
- With `--spec-draft-n-max 3` instead of 5: 16.17 vs. 19.40 — still 17% slower
- 4 slots, no speculation: 6.54 tok/s aggregate
- 4 slots, speculation: 5.76 tok/s aggregate

Speculative decoding was switched off for that model. Not because the feature is bad, but because the measurement said so.

**What to do:** run your own workload, grep the acceptance rate, and compare a timed run with and without `--spec-type`. If acceptance sits below ~0.8, turn it off. MTP heads on the model's own architecture (as here) generally do much better than a separate draft model, which is exactly why this recipe uses MTP.

> [!NOTE]
> Throughput figures for this Qwen configuration are not included here because they have not been measured on the reference hardware. The DeepSeek numbers above are measured and are included to illustrate the decision method, not as a claim about Qwen.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `--spec-type` not recognized | Build predates MTP support | Update llama.cpp and rebuild |
| Slower with speculation than without | Acceptance rate below break-even | Check `draft acceptance` in the log; lower `--spec-draft-n-max`, raise `--spec-draft-p-min`, or disable |
| `draft acceptance` absent from the log | Speculation not actually active | Confirm `--spec-type draft-mtp` and that the GGUF includes MTP weights |
| Out of memory with large `--ctx-size` | KV cache too large | Lower `--ctx-size`/`--parallel`, or use q4_0 KV as shown |
| `curl` succeeds but model not ready | `/health` returns 503 with exit code 0 | Grep the body for `"ok"` |
