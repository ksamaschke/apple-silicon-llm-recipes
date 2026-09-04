# Metal patches for MoE and MLA models

Patches applied to llama.cpp's Metal backend for the models in these recipes. Each one states what it fixes and what it was worth.

## 1. MoE expert-batch threshold (`ne21 >= 8`)

**File:** `ggml/src/ggml-metal/ggml-metal-common.cpp`

```diff
 bool ggml_metal_op_mul_mat_id_use_mm(const struct ggml_tensor * op, bool has_simdgroup_mm) {
     const int64_t ne00 = op->src[0]->ne[0];
     const int64_t ne21 = op->src[2]->ne[1];

-    return has_simdgroup_mm && ne00 >= 64 && ne21 >= 32;
+    return has_simdgroup_mm && ne00 >= 64 && ne21 >= 8;
 }
```

**What it fixes.** `ne21` is how many tokens get routed to a given expert in one batch. With 4 concurrent slots — or with speculative verify batches — that number is typically 4–8, far below the upstream threshold of 32. Everything below the threshold falls back to the vector kernel, which is much slower for this shape.

**Measured effect** (DeepSeek-V4-Flash-Vision, 4 slots, 45k context, distributed over 2 Macs):

- Without: 12.6 tok/s aggregate, with one slot stuck at 3.14 tok/s
- With: 33.67 tok/s aggregate, all slots even

The pathological slow slot disappears entirely. At 1 slot the difference is small (18.47 → 18.98 tok/s) — this patch is about *concurrency*.

The CUDA backend made the analogous change in [ggml-org#27342](https://github.com/ggml-org/llama.cpp/pull/27342).

## 2. Lightning indexer: runtime head count

**Files:** `ggml-metal-device.m`, `ggml-metal-impl.h`, `ggml-metal-ops.cpp`, `kernels/fa.metal`

Upstream hardcodes the head count for the lightning indexer op via a compile-time constant. DeepSeek-V4 has 64 heads, GLM-5.3-Flash has 32 — one of them fails the support check and falls off the Metal path.

The patch passes the head count at runtime and relaxes the check to "whole number of head tiles":

```diff
         case GGML_OP_LIGHTNING_INDEXER:
             if (op->src[0]->ne[0] != OP_LIGHTNING_INDEXER_DK ||
-                op->src[0]->ne[1] != OP_LIGHTNING_INDEXER_NH) {
+                op->src[0]->ne[1] % OP_LIGHTNING_INDEXER_NHPTG != 0 ||
+                op->src[0]->ne[1] < OP_LIGHTNING_INDEXER_NHPTG) {
                 return false;
             }
```

Add `int32_t n_head;` to `ggml_metal_kargs_lightning_indexer` in `ggml-metal-impl.h`, populate it in `ggml_metal_op_lightning_indexer`, and change the kernel loop in `fa.metal`:

```diff
-        FOR_UNROLL (short i_head = 0; i_head < NH; i_head += NHPTG) {
+        for (short i_head = 0; i_head < (short) args.n_head; i_head += NHPTG) {
```

**What it's worth:** required for GLM-5.3-Flash to use the Metal lightning-indexer path at all. Not a tuning knob — without it the op is unsupported.

## 3. RDMA probe diagnostics

**File:** `ggml/src/ggml-rpc/transport-apple.cpp`

Upstream returns `nullptr` silently when the Apple RDMA probe fails, so a silent fallback to TCP looks identical to a working RDMA connection. The patch logs why:

```diff
 static bool rdma_library_present() {
     if (!rdma_library_present()) {
+        GGML_LOG_ERROR("RDMA(Apple/UC) probe: librdma not present\n");
         return nullptr;
     }
```

and, when no device matches the target GID:

```diff
-    if (!ctx) return nullptr;
+    if (!ctx) {
+        char want[64];
+        snprintf(want, sizeof(want), "%u.%u.%u.%u",
+                 target_gid[12], target_gid[13], target_gid[14], target_gid[15]);
+        GGML_LOG_ERROR("RDMA(Apple/UC) probe: no device with an ACTIVE port "
+                       "matching gid %s (%d devices scanned)\n", want, ndev);
+        return nullptr;
+    }
```

**What it's worth:** no throughput change. It turns "why is this slow" into a one-line log answer. Look for `RDMA(Apple/UC) activated` in the worker log to confirm you are not silently on TCP.

## Applying these

Patch **both** machines and rebuild from the same commit with identical flags. A server and worker built differently fail at runtime with `send failed bytes_sent=0`.

```shell
cmake -B build -DGGML_METAL=ON -DGGML_RPC=ON -DGGML_BLAS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
```

`-DGGML_BLAS=OFF` on the worker specifically — BLAS on the worker measurably slows RPC decode.
