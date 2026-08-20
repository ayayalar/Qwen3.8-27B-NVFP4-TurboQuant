# CALIBRATION LOG — everything tried before the recipe worked

Honest record of rejected paths (all run live on the target GPU, 2026-08-14),
with the observation that killed each one. This is the raw material behind the
"deliberately NOT used" table in README.

## Attempts that FAILED / regressed

### 1. Naive boot: fp8 KV at 256K
```
vllm serve ... --max-model-len 262144 --kv-cache-dtype fp8 [--gpu-memory-utilization 0.98]
```
→ `ValueError: To serve at least one request with the model's max seq len (262144),
(8.18 GiB KV cache is needed, which is larger than the available KV cache memory
(6.66 GiB). Based on the available memory, the estimated maximum model length is 213248`.

Model loading took 21.34 GiB; the vision encoder cache + runtime overhead leave
only ~5–7 GiB of KV headroom. The exact "estimated max length" varies with the
available memory at boot time — we captured 158,368 (this load, KV pinned to
5 GiB), 213,248 (no pin), and earlier runs reported 203,840 / 227,360. What is
stable is the direction: fp8 cannot serve 262,144 on this card.

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
the KV pin. **Root cause found 2026-08-15 (this edit):** the garble is a CUDA-graph
capture artifact, not the KV dtype. With `FULL_AND_PIECEWISE` cudagraph mode (the
0.27.1 default at opt level ≥ 2), the spec-verify path inside
`TurboQuantAttentionImpl.forward()` does a GPU→CPU sync via
`query_start_loc.tolist()` and produces attention over the wrong chunk — draft
tokens get rejected every time (`Accepted: 0`, `Per-position acceptance rate:
0.000`) and long-context output collapses. **Stepping CUDA graphs to PIECEWISE
(no FULL capture) makes MTP+turboquant fully correct** — math, tool calls, and
needle recall pass, with MTP accepting drafts:
- single-stream decode: **~97–103 tok/s** (official `speed_test.py` = 96.8;
  vs 54.8 non-spec, i.e. ~1.8×) at full 262,144. *Earlier "166.9" claims were a
  measurement artifact — wall-time divided by requested max_tokens (500) instead
  of actual completion tokens (~314). The harness-consistent number is ~100.*
- 4-way concurrent: **~306–311 tok/s aggregate** (official harness; vs 196.9 non-spec)
- needles 8K / 64K / 131K / 196K all PASS (marker in `content`); tools 12/12
  vs 10/12 non-spec; code-edit PASS

Two equivalent ways to get PIECEWISE on 0.27.1 (no source patch needed):
1. `--enforce-eager` (PIECEWISE→NONE, costs ~40% decode — works, ~99 tok/s), or
2. **dynamic** speculative decoding — add
   `"num_speculative_tokens_per_batch_size": [[1,4,3]]` to the spec config.
   vLLM's own `_maybe_override_dynamic_sd_cudagraph_mode` then automatically
   downgrades FULL→PIECEWISE with a warning ("for reliability"). This is the
   supported route and what `MTP=1` in `scripts/start.sh` uses.

Official `benchmark/bench_framework.py t4` results with `MTP=1` (full suite,
live 2026-08-15): tool calls **12/12** (vs 10/12 non-spec), needles 8K/64K/131K
PASS. **Extreme-window caveat (RESOLVED):** the 196K needle initially OOM'd the
engine in `TurboQuantAttentionImpl._continuation_prefill`
(`k_full[:cached_len] = k_cached_trim.to(qdtype)` → `torch.OutOfMemoryError`,
414 MiB needed vs 414 MiB free at 211K computed tokens). Fix, exactly as the
OOM message recommends: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
With that set by `MTP=1` in `scripts/start.sh`, verified live 2026-08-15:
**196K needle completes with the marker in `content` and the engine stays up**
(HTTP 200 after the prefill). `MTP=1` is now verified across the full 262K
window on this stack.

The open upstream PR #40914 ([K+1 spec-verify routing]) targets the same class
of bug but its dispatch predicate never fires on 0.27.1 (verified: 0/3508
calls eligible) — the PIECEWISE/dynamic-SD route is what actually fixes it on
this version. Conclusion: **MTP + turboquant KV is not the garbler; FULL
CUDA-graph capture is.** Same weights + same KV dtype *without* FULL graphs
generate cleanly.

Latency cost of the fix (measured 2026-08-15, streaming first-delta):
MTP=1 vs MTP=0 is ~2–9% higher TTFT (0.21 vs 0.19s at 1K; 19.09 vs 17.52s at
64K). The throughput win (~1.8× decode) buys more than that back for
generation-heavy workloads, but interactive users waiting on the first token
should weigh MTP=0. See README "Time-to-first-token".

### 4. kv-cache-memory-bytes (pin) ALONE with MTP still garbled
Pinning KV (5 GiB) + MTP: boots, 270,510-token pool (still > 256K), but every
control output empty / no tool call / no recall. At the time this looked like
"#3 is unconditional on pin" — with the root cause now known (FULL CUDA-graph
capture), this test actually reflects the same defect: it was run with
FULL_AND_PIECEWISE graphs. Under PIECEWISE the identical pin+MTP config is
fully correct (see #3).

### 5. `--num-gpu-blocks-override` gymnastics
Tried to force KV allocation (2800/2700/1370/1200 blocks). Runs into the same
physically-over-committed memory as fp8: at 2800 blocks vLLM tried to allocate
8.38 GiB on a card already holding ~21.3 GiB of loaded weights + torch.compile
buffers → `torch.OutOfMemoryError` during `initialize_kv_cache_tensors`. The *pin* is the
clean equivalent of override without the landmine, because it tells the allocator
exactly how much to reserve.

### 6. `--enforce-eager` (without pin) for long requests
Without a KV pin, auto-fit grabs 1.66× the needed pool (435,924 tokens / 7.11 GiB)
and long requests (120–170K) died: `CUDA out of memory. Tried to allocate 190 MiB`
— the card was 100% committed. (This is also the minefield-registry "KV sizing"
class of failure.) Removing enforce-eager alone doesn't fix that; the pin does.

Re-measured 2026-08-14 (this edit): with the pin in place and `--enforce-eager`,
single-stream decode drops from 54.8 → 39.5 tok/s and 4-way aggregate from
196.9 → 150.8 tok/s (~28% slower). CUDA graphs are worth keeping.

### 7. KV de-over-provisioning: making a genuine 262K request survive (verified 2026-08-17)

**The claim that never existed.** All prior "full 262K verified" wording meant
`--max-model-len 262144` (a config setting) with decode measured on ~40-token
prompts. No genuine 262K-length prefill was ever run: the longest actual prompt
tests were the 196K needles. This mattered because a real 262K prefill did not
work on the shipped config.

**Failure mode.** Single-stream requests with ~245K / ~258K prompt tokens
(prefill, thinking off, real usage tokens) crashed the engine repeatedly:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 472.00 MiB.
GPU 0 has a total capacity of 31.40 GiB of which 450.62 MiB is free.
... 153.73 MiB reserved but unallocated ...
```

This occurred **with** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in
force — the same mitigation that fixed the 196K case (§3) is insufficient at
245-258K. The engine died (`EngineDeadError`); 200K and 214K prefills
survived, 245K crashed. The allocation shortfall was only ~21 MiB: this is a
tight activation-headroom problem, not a KV-capacity one (the pool held
~306K tokens).

**Root cause.** `--kv-cache-memory-bytes 5800000000` reserves 5.4 GiB of KV but
a single request at the cap needs only ~5.02 GiB of KV (vLLM's own estimator:
a 5.25e9 pin caps max model length at 255,840). The 0.38 GiB of over-provision
buys 4-way concurrency headroom but starves prefill activations.

**Verified fix.** De-over-provision the pin: `KV_BYTES=5500000000` (5.12 GiB)
frees ~0.3 GiB for prefill activations while the KV pool (~268K tokens) still
covers one max-length sequence. Genuine single-stream requests then complete
end-to-end with the engine healthy afterwards:

- **262,122-token prefill** (99.99% of cap, calibrated via `/tokenize`,
  `usage.prompt_tokens` confirmed) — completes in ~122s, engine HTTP 200 after.
- Needle recall at **260,640** tokens — marker in `content`. ✅
- Code-edit at **260,665** tokens — emitted `return x * 5`. ✅
- Decode at the cap: ~11-12 tok/s (prose, thinking off); full decode-vs-context
  curve in README §"Decode vs context".

**Trade-off and default.** The 5.5e9 pin reduces the pool ~306K → ~268K tokens,
so four concurrent long-context sequences will run short on KV. The shipped
default (5.4 GiB) stays the default for 4-way workloads; `KV_BYTES=5500000000`
is the documented single-stream/262K profile (now an env hook in `start.sh`).
Do not publish "~150 tok/s at 262K": that number is short-context-only, and a
262K decode at full depth is ~12 tok/s for prose.

## What the final config changed vs each failure

| Failure | Final config element that resolves it |
|---|---|
| fp8 KV doesn't fit | `turboquant_4bit_nc` halves KV bytes/token |
| MTP garbles 4-bit output | run MTP under PIECEWISE graphs (dynamic spec-decode; `MTP=1`) |
| NVFP4 KV unsupported | not used; 4-bit turboquant used instead |
| auto-fit overallocates → OOM | `--kv-cache-memory-bytes 5368709120` (5 GiB pinned) |
| OOM during override | pin replaces block-override |
| slow decode | CUDA graphs kept ON (no enforce-eager) — ~28% faster (re-measured 2026-08-14; earlier log noted 43%) |

## Benchmark harness bugs we hit (worked through, not config bugs)

- **First harness no-op'd** (orphaned loop body) → re-written; always smoke-test
  a harness before trusting it.
- **Scored the wrong field:** scoring reasoning text (which re-states the prompt)
  made needles look broken; scoring `content` only was correct.
- **Exact-match scoring false-fails** a model that paraphrases ("quantum computing
  articles" ≠ "quantum computing") — substring matching is the fair metric.
- **Tiny `max_tokens` truncates content** while the real answer is in `reasoning`.
- **`enable_thinking:false` collapses the counting-prompt benchmark.** With
  thinking off, the model emits `1` (2 completion tokens) and stops — the
  harness's long-output counting prompt only sustains generation *because* of
  thinking, so "24 tok/s" there is 2 tokens ÷ request overhead, not a real
  decode slowdown. Any thinking-off throughput comparison must force a long
  output (e.g. `ignore_eos` + high `min_tokens` or a genuinely long prompt);
  raw per-token decode is not faster with thinking off.
- Note the `bench_framework` config guard: for "fp8"-tag runs it only tests
  lengths ≤ 131072 (because fp8 can't reach 262144).

## max-num-batched-tokens sweep under MTP (measured 2026-08-15, §8)

Live sweep on the reference 5090, MTP K3 + dynamic spec (PIECEWISE), same
pin/util/seqs, only `--max-num-batched-tokens` (per-arm `mnbt`) or K varied.
Official harness (`speed_test.py`), single-run per arm except the headline arm.

| arm | mnbt | K | single tok/s | 4-way agg tok/s | mean accept | tokens emitted |
|---|---|---:|---:|---:|---:|---:|
| default | 512 | 3 | **96.9** | **311.6** | 2.39 | 400 (thinking-only head) |
| 1024 | 1024 | 3 | **148.5** (3/3 runs, deterministic) | 203.6 | 3.77 | 340 (content visible) |
| 2048 | 2048 | 3 | 88.1 | 194.6 | 2.16 | 400 |
| 4096 | 4096 | 3 | 96.3 | 197.9 | 2.45 | 400 |
| K4 | 2048 | 4 | 145.0 | 145.6 | 3.02 | 106 (early stop) |
| K5 | 2048 | 5 | 138.7 | 181.2 | 3.79 | 251 |

Interpretation:
- **mnbt=1024 is the verified default**: +53% single-stream (148.5 vs 96.9),
  deterministic across runs, acceptance 3.77, AND +6% 4-way aggregate (330.5 vs
  310.4) on a clean boot. The earlier "−35% aggregate" (203.6) was a warm-boot /
  leftover-GPU-state artifact from the stacked sweep — a clean-boot A/B
  (2026-08-16) refutes it. `MNBT` env exposes both; default is 1024.
- Configs that speed up single-stream also *changed generation shape* (fewer
  tokens emitted, content visible in the counting bench, higher K → early
  stops). Only mnbt=1024/k3 produced the full 340-token stable profile; other
  fast arms (K4/K5) were shorter outputs, so treat their single-stream wins as
  optimistic vs. sustained decode.

## Why these exact numbers

- `5368709120` = 5 GiB. Empirically 306,325 KV tokens (1.17× of 262,144) with
  ~2–3 GiB of GPU headroom left for prefill activations. The estimator's own
  "maximum concurrency for 262,144 tokens per request: 1.17x" confirms this is
  sized correctly.
- `max-num-batched-tokens 1024` (default) is the verified optimum under MTP with
  K3: best single-stream (+53% vs 512) and no aggregate penalty on a clean
  boot (+6%). 2048/4096 regress both. Knob: `MNBT` (512 available).
- `max-num-seqs 4` provides mild concurrency without reducing KV below 1 request.


## 7. FlashInfer b12x NVFP4 GEMM opt-in vs stock Cutlass (A/B, 2026-08-19)

Motivation: HF Qwen3.8-27B discussion #132 claims `--linear-backend flashinfer_b12x`
(SM120 warp-level MMA FP4 GEMM via FlashInfer) yields +6% decode vs stock auto.
Same vLLM version (0.27.1), RTX 5090, so directly applicable in principle.

Required temporary work on this stock env (both one-line patches applied live
for the A/B, then reverted — pristine stock restored 2026-08-19; the file now
carries no trace). Re-apply both to retry:
1. **Registry**: `FlashInferB12xNvFp4LinearKernel` was imported + mapped but never
   added to `_POSSIBLE_NVFP4_KERNELS["CUDA"]`, so the opt-in flag hard-failed
   (`--linear-backend=flashinfer_b12x ... no 'flashinfer_b12x' kernel exists for
   NVFP4 layers`). Added it at the END of the list so auto-selection priority is
   unchanged; `--linear-backend flashinfer_b12x` now resolves.
2. **Map entry**: the b12x backend set contained only the B12x class, so the
   filter ALSO refused the model's W8A8-FP8 layers (unsloth NVFP4 blend carries
   CompressedTensorsW8A8Fp8 layers; the thread's weights were pure NVFP4).
   Added `CutlassFP8ScaledMMLinearKernel` as fallback in the `flashinfer_b12x`
   map entry — used only for FP8 layers, never for NVFP4 GEMM.
3. **Env**: `has_flashinfer()` is False at boot unless nvcc is on PATH (no
   prebuilt cubins in the 0.6.16.post3 wheel). Thread requires
   `CUDA_HOME=<cuda>` + `MAX_JOBS=2`; we used the box's own CUDA 13.3.
   (Environment only, no file change — survives.)

Measurement (fresh boot each arm, same script = repo speed harness, pre-bench
soak with Triton-JIT flush, 6x 400-token full-length single-stream decode,
all `finish=length`, content-empty/reasoning-heavy as this stack normally does):

| arm | tok/s per rep | mean | median | sd |
|-----|--------------|------|--------|----|
| base (Cutlass, stock) | 123.4 133.9 132.7 130.4 131.5 121.2 | 128.9 | 131.5 | 5.4 |
| b12x | 147.5 119.3 107.6 137.8 149.4 124.2 | 131.0 | 131.0 | 15.5 |

Verdict: NO improvement on this stack. Means/medians statistically tied
(+1.6%/-0.4% on mean, exactly tied on median) and b12x var is ~3x worse with a
slow tail (two 107–124 reps vs base's tight 121–134 band). Correctness retained
(2+2=4, Paris) — b12x is numerically sane here, just not faster. The +6% figure
came from a fp8-KV + static-MTP/full-graph stack; on 4-bit KV + dynamic
spec/PIECEWISE graphs + unsloth weights it does not reproduce.

Decision: stock launch unchanged. b12x is a manual opt-in only — to retry
you must re-apply the two one-line patches (registry + map fallback) above,
then pass `--linear-backend flashinfer_b12x` with nvcc on PATH.
