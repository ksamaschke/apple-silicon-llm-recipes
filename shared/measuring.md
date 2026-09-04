# Measuring throughput honestly

Most of the confusing numbers in distributed llama.cpp benchmarking come from measuring the wrong window. This page documents the traps encountered while producing the numbers in these recipes.

## `eval time` per slot is not a decode rate

llama.cpp reports `eval time` per slot as **wall-clock time including waiting**. With `--parallel 4`, a slot that spends half its time waiting for the batch scheduler still reports that time as its own. Summing per-slot rates from the server log overstates throughput; reading a single slot's rate understates it.

Measure from the client side, and be explicit about which window you mean.

## Three different "aggregate" numbers

For 4 sessions generating 1200 tokens each, all three of these are defensible and they differ:

**Sum of individual rates.** Each session's tokens divided by its *own* wall time, summed. This is an upper bound — it silently assumes perfect overlap.

**Window to last finisher.** All 4800 tokens divided by the time from first start to last finish. Conservative; penalizes stragglers.

**Fully-parallel window.** All tokens divided by the interval during which *all* slots were actually decoding. The most honest number, and the hardest to get right.

In an early run these gave 32.81, 28.1 and 6.60 tok/s for the same measurement. The 6.60 included a cold 4×45k prefill in the window — it was a throughput number for a whole cycle, not a decode rate, and reporting it as "aggregate decode" was misleading.

## Fix: barrier-synchronize the clients

Have every client prefill its context first, then wait at a barrier, then start generating together. When all sessions start and end within ~0.1 s of each other, the three numbers above converge and the ambiguity disappears.

Measured example (4 slots, 45k context):

```
SUM of individual rates :  33.67 tok/s
WINDOW to last finisher :  33.66 tok/s
FULLY PARALLEL          :  33.67 tok/s
```

When those three agree, the number means something.

## Separate prefill from decode

Prompt processing and token generation have completely different performance characteristics:

- **Prompt processing:** ~217–292 tok/s on the measured setup, roughly flat with prompt size
- **Decode:** ~8.4 tok/s per slot at 4 concurrent slots

Mixing them produces numbers that depend entirely on your prompt-to-generation ratio. Report them separately.

Also watch the cache: a request reporting `cached=21976 neu=2047` did almost no prefill. The `usage.prompt_tokens_details.cached_tokens` field in the OpenAI-compatible response tells you which case you are in.

## Check plausibility before believing a number

A rate that seems too good usually is. Cross-checks that caught real errors:

- **Does the model even fit?** 119 GiB weights + 43 GiB KV on a 112 GiB machine means the measurement cannot be single-node, whatever the log says.
- **Did the operation actually happen?** A slot restore that reports `n_restored: 0` and reads a 12 MB file for 24k tokens restored nothing — the subsequent 95 s prefill confirmed it.
- **Is the comparison fair?** Two runs with different context lengths, cache states, or concurrency are not comparable, however tempting the delta.
