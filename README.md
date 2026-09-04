# Apple Silicon LLM Recipes

> Step-by-step recipes for serving large GGUF models **distributed across two Apple Silicon Macs** over Thunderbolt RPC.

Modeled on the [NVIDIA DGX Spark Playbooks](https://github.com/NVIDIA/dgx-spark-playbooks) format: every recipe is a single README with prerequisites, numbered steps, measured expectations, and a troubleshooting table.

## About

These recipes cover what the llama.cpp docs leave out for Apple Silicon:

- Splitting a model across two Macs via `ggml-rpc-server` when it does not fit on one
- Metal kernel patches — one that doubles MoE throughput, one without which GLM does not run at all
- Which settings are measured wins and which are measured losses
- Failure modes specific to macOS (Local Network Privacy, RPC connection lifecycle)

**Every performance number here was measured on real hardware.** Where something was not measured, it says so.

## Available Recipes

- **[Qwen3.8-Flash-Next with MTP speculative decoding](qwen3.8-flash-next-mtp/)** — fastest measured: 84.97 tok/s aggregate at 4 slots
- **[GLM-5.3-Flash distributed](glm-5.3-flash-distributed/)** — hybrid KDA/MLA, small KV cache, 43.88 tok/s; needs the lightning-indexer patch
- **[DeepSeek-V4-Flash-Vision distributed](deepseek-v4-distributed/)** — 119 GiB across 2×128 GB, 33.67 tok/s, vision-capable

## Measured comparison

Same two machines, same harness, 4 concurrent slots, aggregate decode throughput:

- **Qwen3.8-Flash-Next Q4_K_XL + MTP:** 84.97 tok/s
- **Qwen3.8-Flash-Next Q4_K_XL:** 70.27 tok/s
- **GLM-5.3-Flash IQ3_XXS:** 43.88 tok/s
- **DeepSeek-V4-Flash-Vision Q3_K_XL:** 33.67 tok/s

Speculative decoding helped exactly one of the three — see [the analysis](qwen3.8-flash-next-mtp/#deciding-whether-speculation-helps).

## Shared references

- **[Metal patches](shared/metal-patches.md)** — the diffs, what they fix, what they are worth
- **[Distributed RPC setup](shared/rpc-setup.md)** — Thunderbolt bridge, RDMA, connection lifecycle, macOS launch-chain trap
- **[Measuring throughput honestly](shared/measuring.md)** — why `eval time` per slot is not a decode rate

## Hardware used

- **2× Mac Studio M3 Ultra, 128 GB unified memory each**, connected by Thunderbolt 5
- macOS 26.6, `iogpu.wired_limit_mb=114688` on both machines
- Models on external NVMe (Thunderbolt 5)

Your numbers will differ. The *relative* effects (patch vs. no patch, speculation vs. none) should hold on comparable hardware.

## Contributing

Corrections and additional measurements are welcome — especially from owners of different Apple Silicon configurations. Please include the exact command line and raw log excerpt for any number you report.

## License

Apache-2.0. See [LICENSE](LICENSE).
