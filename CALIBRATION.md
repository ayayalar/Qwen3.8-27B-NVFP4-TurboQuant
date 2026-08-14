# CALIBRATION LOG — everything tried before the recipe worked

Honest record of rejected paths (all run live on the target GPU, 2026-08-14),
with the observation that killed each one. This is the raw material behind the
"deliberately NOT used" table in README.

## Attempts that FAILED / regressed

### 1. Naive boot: fp8 KV at 256K
```
vllm serve ... --max-model-len 262144 --kv-cache-dtype fp8 [--gpu-memory-utilization 0.98]
```
→ `ValueError: To serve ... max seq len (262144), (8.18 GiB KV cache is needed,
larger than the available KV cache memory (6.01 GiB). Estimated max model length 203840`.
Model weights (22.13 GiB) + vision encoder cache + runtime overhead leave only
~6–7 GiB. Adding MTP to this config *shrank* available KV further (5.84 GiB).

### 2. `--kv-cache-dtype nvfp4` (4-bit, but no backend)
```
ValueError: No valid attention backend found for cuda ... kv_cache_dtype=nvfp4 ...
Reasons: {FLASH_ATTN: [kv_cache_dtype not supported], FLASHINFER: [...], TRITON_ATTN: [...],
FLEX_ATTENTION: [...], TURBOQUANT: [...]}
```
Every backend rejects NVFP4 KV on this board. Dead end regardless of everything
else.

### 3. MTP speculative decoding (the trap)
```
--speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'
```
Served fine at boot (it boots *cleanly* — that's the trap), but output was garbage:
- `2+2` → `4` **then** `I I I I I I I ...` (repetition collapse mid-stream)
- tool calls → `None` / empty
- needle recall → empty, reasoning loop `think think think ...`

Reproduced at 262144 with turboquant (both fp8 KV and 4-bit KV), with and without
the KV pin. Conclusion: **MTP + this model's KV path is the garbler, not the KV
quant itself.** Same weights + same KV dtype *without* MTP generate cleanly.

### 4. kv-cache-memory-bytes (pin) ALONE with MTP still garbled
Pinning KV (5 GiB) + MTP: boots, 270,510-token pool (still > 256K), but every
control output empty / no tool call / no recall. Confirms #3 is unconditional on pin.

### 5. `--num-gpu-blocks-override` gymnastics
Tried to force KV allocation (2800/2700/1370/1200 blocks). Runs into the same
physically-over-committed memory as fp8: at 2800 blocks vLLM tried to allocate
8.38 GiB on a card already holding 22.3 GiB of weights + torch.compile buffers →
`torch.OutOfMemoryError` during `initialize_kv_cache_tensors`. The *pin* is the
clean equivalent of override without the landmine, because it tells the allocator
exactly how much to reserve.

### 6. `--enforce-eager` (without pin) for long requests
Without a KV pin, auto-fit grabs 1.66× the needed pool (435,924 tokens / 7.11 GiB)
and long requests (120–170K) died: `CUDA out of memory. Tried to allocate 190 MiB`
— the card was 100% committed. (This is also the minefield-registry "KV sizing"
class of failure.) Removing enforce-eager alone doesn't fix that; the pin does.

## What the final config changed vs each failure

| Failure | Final config element that resolves it |
|---|---|
| fp8 KV doesn't fit | `turboquant_4bit_nc` halves KV bytes/token |
| MTP garbles 4-bit output | MTP removed entirely (`--speculative-config` absent) |
| NVFP4 KV unsupported | not used; 4-bit turboquant used instead |
| auto-fit overallocates → OOM | `--kv-cache-memory-bytes 5368709120` (5 GiB pinned) |
| OOM during override | pin replaces block-override |
| slow decode | CUDA graphs kept ON (no enforce-eager) — 43% faster |

## Benchmark harness bugs we hit (worked through, not config bugs)

- **First harness no-op'd** (orphaned loop body) → re-written; always smoke-test
  a harness before trusting it.
- **Scored the wrong field:** scoring reasoning text (which re-states the prompt)
  made needles look broken; scoring `content` only was correct.
- **Exact-match scoring false-fails** a model that paraphrases ("quantum computing
  articles" ≠ "quantum computing") — substring matching is the fair metric.
- **Tiny `max_tokens` truncates content** while the real answer is in `reasoning`.
- Note the `bench_framework` config guard: for "fp8"-tag runs it only tests
  lengths ≤ 131072 (because fp8 can't reach 262144).

## Why these exact numbers

- `5368709120` = 5 GiB. Empirically 306,325 KV tokens (1.17× of 262,144) with
  ~2–3 GiB of GPU headroom left for prefill activations. The estimator's own
  "maximum concurrency for 262,144 tokens per request: 1.17x" confirms this is
  sized correctly.
- `max-num-batched-tokens 512` keeps prefill chunks small → bounded activation
  peak for long-context requests (no re-alloc surprise).
- `max-num-seqs 4` provides mild concurrency without reducing KV below 1 request.
