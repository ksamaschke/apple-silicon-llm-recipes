# Apple Silicon LLM Recipes

> Step-by-step recipes for serving large GGUF models on Apple Silicon Macs — including **distributed inference across two machines** over Thunderbolt RPC.

Modeled on the excellent [NVIDIA DGX Spark Playbooks](https://github.com/NVIDIA/dgx-spark-playbooks) format: every recipe is a single README with prerequisites, numbered steps, measured expectations, and a troubleshooting table.

## About

These recipes cover what the llama.cpp docs leave out for Apple Silicon:

- Splitting a model that does **not** fit on one machine across two Macs via `ggml-rpc-server`
- Metal kernel patches that materially change MoE throughput
- Which settings are measured wins and which are measured losses
- Failure modes specific to macOS (Local Network Privacy, RPC connection lifecycle)

**Every performance number in these recipes was measured on real hardware.** Where something was not measured, it says so.

## Available Recipes

> GLM and Qwen recipes are being rewritten from measured configurations and will return shortly.

- **[DeepSeek-V4-Flash-Vision distributed across two Macs](deepseek-v4-distributed/)** — 119 GiB model on 2×128 GB, 33.7 tok/s aggregate at 4 concurrent slots

## Shared references

- **[Metal patches for MoE and MLA models](shared/metal-patches.md)** — the diffs, what they fix, and what they are worth
- **[Distributed RPC setup](shared/rpc-setup.md)** — Thunderbolt bridge, RDMA, connection lifecycle
- **[Measuring throughput honestly](shared/measuring.md)** — why `eval time` per slot is not a decode rate

## Hardware used

All measurements come from:

- **2× Mac Studio M3 Ultra, 128 GB unified memory each**, connected by Thunderbolt 5
- macOS 26.6, `iogpu.wired_limit_mb=114688` on both machines
- Models on external NVMe (Thunderbolt 5)

Your numbers will differ. The *relative* effects (patch vs. no patch, 2 slots vs. 4) should hold on comparable hardware.

## Contributing

Corrections and additional measurements are welcome — especially from M4 Max / M3 Ultra owners with different memory configurations. Please include the exact command line and raw log excerpt for any number you report.

## License

Apache-2.0. See [LICENSE](LICENSE).
