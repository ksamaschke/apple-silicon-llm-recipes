# Distributed RPC setup on Apple Silicon

How to split one model across two Macs with llama.cpp's RPC backend, and the macOS-specific failure modes that make it look broken when it isn't.

## Why Thunderbolt, not Ethernet

Layer-split inference sends activations across the link on **every token**. The interconnect sits directly in the decode path. A Thunderbolt cable between two Macs gives a low-latency point-to-point link and, on Apple Silicon, access to RDMA — which llama.cpp uses when it can.

Connect the two machines with a Thunderbolt cable and give the bridge interface static addresses, e.g. `192.168.0.1` (server) and `192.168.0.2` (worker). Verify:

```shell
ping -c 3 192.168.0.2
```

Sub-millisecond round-trip is what you want (measured: 0.74 ms).

**Use the Thunderbolt address in `--rpc`.** Passing the LAN address of the same machine works — and quietly routes every token over Ethernet.

## Build both sides identically

```shell
cmake -B build -DGGML_METAL=ON -DGGML_RPC=ON -DGGML_BLAS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
```

Two rules:

- **Same commit, same flags, both machines.** Mismatch produces `send failed bytes_sent=0` at runtime.
- **`-DGGML_BLAS=OFF` on the worker.** BLAS on the worker measurably slows RPC decode.

## The worker accepts exactly one connection

`ggml-rpc-server` accepts **one connection per process lifetime**. This is the single most common cause of "Failed to connect".

Consequences:

- Do not test the port with `nc`, `telnet`, or `lsof` — that consumes the connection, and the real server then fails
- Restart the worker immediately before every server start
- If the server crashes, restart the worker too before retrying

```shell
pkill -f ggml-rpc-server; sleep 5
script -q /dev/null ~/llama.cpp-rpc/build/bin/ggml-rpc-server \
  --host 0.0.0.0 --port 50053 > ~/worker.log 2>&1 &
```

`script -q /dev/null` gives a pseudo-terminal so output is line-buffered. Without it the log stays empty and you cannot tell what the worker is doing.

Confirm in the log:

- `Accepts: 1` — worker ready
- `RDMA(Apple/UC) activated` — RDMA in use rather than plain TCP

## macOS Local Network Privacy: launch chain matters

macOS grants local-network permission based on the **process launch chain**. A deeply nested launch loses the grant, and llama.cpp reports it as:

```
invalid device: RPC0
Failed to connect to 192.168.0.2:50053
```

This looks like a network problem. It is not — the network is fine, the process simply is not allowed to use it.

**Broken:** `ssh → screen → script → screen → llama-server`

**Works:** one flat `screen` launched directly over SSH:

```shell
ssh user@server 'screen -dmS dsprod /bin/bash -c "~/llama.cpp-rpc/build/bin/llama-server ... "'
```

Also avoid `nohup` for `llama-server`. The fix here was the *launch method*, not the code.

## Splitting the layers

`--tensor-split` distributes layers across devices in `--device` order:

```shell
--device RPC0,MTL0 --n-gpu-layers 999 --tensor-split 72,90
```

72 layers on the remote worker, 90 on the local GPU. Tune to the memory available on each side and confirm by watching both machines' memory during load.

Set the usable GPU memory ceiling on **both** machines (does not survive reboot):

```shell
sudo sysctl iogpu.wired_limit_mb=114688
```

## Required with `--parallel > 1`

```
--cache-ram 0
```

Not optional when combining RPC with multiple slots — see [ggml-org#26128](https://github.com/ggml-org/llama.cpp/issues/26128).

## Readiness check

```shell
until curl -s http://127.0.0.1:8241/health | grep -q '"ok"'; do sleep 10; done
```

Grep the **body**. `curl` exits 0 on HTTP 503 while the model is still loading, so the exit code tells you nothing.

## Quick diagnosis

| Symptom | First thing to check |
|---|---|
| `invalid device: RPC0` | Launch chain — flatten it |
| `Failed to connect` | Worker's one connection already used; restart worker |
| `send failed bytes_sent=0` | Build mismatch between the two machines |
| Empty worker log | Missing `script -q /dev/null` |
| Slow decode, no obvious cause | `RDMA(Apple/UC) activated` in worker log? Using the Thunderbolt IP? |
