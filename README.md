# Qwen3.8-27B-NVFP4 serving recipe — full 256K context on a single RTX 5090

Validated serving configuration for running `unsloth/Qwen3.8-27B-NVFP4` at its full
**262,144 (256K) token** context on one NVIDIA RTX 5090 (32 GB VRAM), including
working tool-calling and long-context reliability.

> Status: **reproduced and verified** on 2026-08-14 against `vLLM 0.27.1`,
> `unsloth-nvfp4-env`, CUDA 13.3 / driver 610.43, RTX 5090.
> Every flag choice in the config below was made against a failed alternative.
> See [CALIBRATION.md](CALIBRATION.md) for the full rejection log — several
> "obvious" configs look correct and silently degrade the model.

## QuickStart

Requirements: a 32 GB (or larger) NVIDIA GPU, ~35 GB free disk (model weights
~22 GiB + vLLM/torch venv ~8 GiB), and `uv` or `python3` available. All
scripts are idempotent and `$HOME`-relative; see the
`Files` section for environment-variable overrides.

```bash
git clone https://github.com/ayayalar/Qwen3.8-27B-NVFP4-TurboQuant.git
cd Qwen3.8-27B-NVFP4-TurboQuant

# 1. One-time bootstrap: creates the vLLM venv and downloads the model (~22 GiB).
#    Safe to re-run; skips anything already in place.
./scripts/setup.sh

# 2. Start the server (background, pidfile + log in /tmp)
./scripts/start.sh

# 3. Smoke-test against the OpenAI-compatible endpoint
curl http://localhost:8000/v1/models

# 4. Stop the server (graceful drain, then SIGTERM/SIGKILL fallback)
./scripts/stop.sh
```

Most users never need to touch any flag — `start.sh` already carries the
validated configuration. Override paths/ports with environment variables
(`MODEL_DIR`, `VLLM_BIN`, `PORT`, …) only if your setup differs from the
defaults. The full manual launch command, with the reasoning for every flag,
is below.

## Constraints that shape this config

A 27B model at NVFP4 occupies **~21.3 GiB** of the 5090's 31.8 GiB VRAM after
load (measured `Model loading took 21.34 GiB`; checkpoint is 21.81 GiB on disk).
The native 256K window needs ~8.18 GiB of fp8 KV cache, more than the ~6–7 GiB
that remains after weights + runtime. You cannot simply "set a bigger context" —
naive attempts politely refuse at boot or OOM at the first long request.

The recipe that works combines three non-obvious choices:

1. **4-bit KV cache** (`turboquant_4bit_nc`) instead of fp8 — halves KV footprint
   so a full 256K window fits. (fp8 only reaches ≤ ~200K on this card.)
2. **KV cache pinned in bytes** (`--kv-cache-memory-bytes 5GiB`) instead of the
   auto-fitter. Auto-fit over-reserved a 1.66× oversized pool (435K tokens),
   leaving ~0 MB for prefill activations → long requests died with
   `CUDA out of memory`. Pinning to 5 GiB yields 306K tokens of KV (1.17× a max
   request) plus activation headroom.
3. **MTP speculative decoding OFF.** This is the counterintuitive one — see below.

## The recipe

```bash
vllm serve /path/to/unsloth/Qwen3.8-27B-NVFP4 \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name unsloth/Qwen3.8-27B-NVFP4 \
  --max-model-len 262144 \
  --kv-cache-dtype turboquant_4bit_nc \
  --kv-cache-memory-bytes 5368709120 \    # 5 GiB, pinned (see calibration)
  --max-num-seqs 4 \
  --max-num-batched-tokens 512 \
  --gpu-memory-utilization 0.98 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3
```

**Deliberately NOT used** (each item is a failed or harmful alternative):

| Flag | Why not |
|---|---|
| MTP spec-decode (`--speculative-config {mtp, num_speculative_tokens:3}`) | **Works on 0.27.1 only with the caveats documented in CALIBRATION §3.** Naïve use (static K3 + default FULL CUDA graphs) garbles output — we saw 125.8 tok/s with empty `content` and an eventual `CUDA error: illegal memory access`. That was the FULL-cudagraph spec-verify bug, not MTP itself. Use it as `MTP=1` in `scripts/start.sh` (dynamic spec → PIECEWISE graphs + expandable-segments allocator): correct across the full 262K window, ~1.8× decode, tools 12/12. Otherwise the plain config stays at ~55 tok/s. |
| `--kv-cache-dtype nvfp4` | No vLLM attention backend supports NVFP4 KV on this GPU — fails at boot with "No valid attention backend found" listing every backend rejecting it. |
| `--kv-cache-dtype turboquant_4bit_nc` + MTP | Only garbles under **FULL** CUDA graphs; correct under PIECEWISE (see row above / CALIBRATION §3). |
| `--enforce-eager` | Not needed *once KV is pinned* — costs CUDA graphs + torch.compile (re-measured 2026-08-14: ≈28% slower decode). Only keep it if you are OOM-ing with auto-fit. |
| `--kv-cache-dtype fp8` at 262144 | Fails boot: needs 8.18 GiB at 262144; available KV memory varies by boot state (measured 5.0–6.66 GiB → vLLM's "estimated max length" landed at 158,368 / 213,248 / 203,840 / 227,360 on different runs). fp8 is only usable at ≤ ~200K on this card. |
| `--gpu-memory-utilization 0.985+` | Exceeds free GPU memory at boot ("Free memory less than desired utilization"). `0.98` is the practical ceiling on a 32 GB card. |

## Benchmark results (validated 2026-08-14, this configuration)

### Long-context recall — needle-in-haystack

Deterministic probe: a unique `NEEDLE-XXXX <status>` marker sentence embedded at
~60% depth in unique, non-repeating filler, scored substring-match against the
`content` field. Reported per prompt-token count.

| Prompt tokens | Result |
|---:|---|
| 8K | ✅ PASS |
| 64K | ✅ PASS |
| 131K | ✅ PASS |
| 196K | ✅ PASS |

### Agentic code-edit

Prompt: locate `def double_value(x): return x * 2` inside a large document and
rewrite it to multiply by 4. Scored on the emitted function line.

| Prompt tokens | Result | Model output |
|---:|---|---|
| 64K | ✅ PASS | `def double_value(x): return x * 4` |
| 131K | ✅ PASS | `def double_value(x): return x * 4` |

### Tool calling

12 structured-call tasks × 2 reps (web_search, send_email, book_flight, add,
get_weather, search_files), scored on tool name + argument values.

| Metric | Result |
|---|---:|
| Passing calls | 10 / 12 |
| Verified args (correct tool + values) | 10 / 12 |

The 2 misses (`book_flight`) produced no tool call at all — identical behavior on
fp8/KV-pinned configs, i.e. a model quirk, not this recipe.

### Decode throughput

| Workload | Tokens/s |
|---|---:|
| Single request | ~55 |
| 4 concurrent requests (aggregate) | ~197 |
| Per-request under 4-way load | ~49 |

`--enforce-eager` downgrades these to ~40 / ~151 (re-measured, see CALIBRATION) — keep CUDA graphs on.

### Benchmark provenance

- Scripts: `benchmark/bench_framework.py` (recall + code-edit + tools),
  `benchmark/control_test.py`, `benchmark/speed_test.py` — stdlib-only, hit
  `POST /v1/chat/completions` on the served endpoint, no external deps.
- All numbers measured on the reference hardware/software in
  [Environment used for validation](#environment-used-for-validation) against
  the *final* recipe flags (KV pinned, no MTP, CUDA graphs on).
- Scoring caveats (why substring matching on `content` only, and not exact
  matching on `reasoning`): see [CALIBRATION.md](CALIBRATION.md).

## Known quirks (read before wiring clients)

- **Answers can land in `reasoning`, not `content`** on long/complex prompts:
  `content` may be empty while the answer sits in the `reasoning` field. Parse
  both. (`--reasoning-parser qwen3` extracts thinking into a separate field.)
- 4-bit KV is slightly lossier than fp8 at the extreme edge of the window. All
  structured tests here pass, but for fidelity-critical long-context recall you
  can trade context down (~200K, the fp8 ceiling on this card) for fp8.
- **TurboQuant kernel path** requires FlashAttention v2 in this vLLM build. The
  launcher pins `--attention-config.flash_attn_version=2` explicitly so a future
  vLLM upgrade can't silently pick FA3 and change decode behavior.
- **No CPU offloading** — the whole point of this recipe is GPU-only. We never
  touch the 61 GB system RAM for serving, so don't add `--cpu-offload-gb`.

## Environment used for validation

- Hardware: PowerSpec PC — RTX 5090 32 GB (GB202, Blackwell), 32 cores, 61 GB RAM
- Software: Ubuntu 24.x, driver 610.43.02, CUDA UMD 13.3, Python 3.13 venv
  `unsloth-nvfp4-env` | vLLM 0.27.1 | flashinfer | nvidia-cutlass-dsl (per
  unsloth's install instructions: `uv pip install "vllm>=0.25.0"
  "flashinfer-python>=0.6.13" "nvidia-cutlass-dsl>=4.5.2" --torch-backend=auto`)
- Model: `unsloth/Qwen3.8-27B-NVFP4` downloaded via `hf download` to
  `$HOME/models/unsloth/Qwen3.8-27B-NVFP4`

## Files

```
scripts/setup.sh           — one-time bootstrap: create vLLM env + download model (~22 GiB), idempotent
scripts/start.sh           — start the server in the background (pidfile + log); MTP=1 opt-in spec-decode
scripts/stop.sh            — graceful stop (SIGTERM, then SIGKILL after timeout)
benchmark/bench_framework.py — tool-call + needle + code-edit benchmark (stdlib-only)
benchmark/control_test.py  — minimal A/B control (stdlib-only)
benchmark/speed_test.py    — decode speed measurement (single + 4-way concurrent)
```

Lifecycle:
```bash
# one-time bootstrap: creates the vLLM venv and downloads the weights if missing
./scripts/setup.sh                                   # idempotent — safe to re-run

# defaults resolve under $HOME (portable across machines); override with env vars
MODEL_DIR=/absolute/path/... ./scripts/start.sh      # background, pidfile written
./scripts/stop.sh                                     # drains, then SIGTERM/SIGKILL

# MTP speculative decoding is ON by default (~1.8x single-stream decode speed,
# verified correct at full 262K on 0.27.1). Disable with:
MTP=0 ./scripts/start.sh
```

`scripts/start.sh` runs with speculative decoding (dynamic `num_speculative_tokens_per_batch_size` → vLLM steps cudagraph from `FULL_AND_PIECEWISE` to `PIECEWISE`; the FULL capture path corrupts turboquant-KV output under speculation on 0.27.1 — see CALIBRATION.md §3) and sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (required for the full-262K window under MTP). `MTP=0` restores the original non-spec configuration verbatim: 5 GiB KV pin, no speculation, no allocator override.

All benchmark scripts are `urllib`-only (no deps) and target an OpenAI-compatible
`/v1` endpoint. Scoring is substring-based and scores the **content** field only —
see CALIBRATION.md for why naive exact-match scoring of reasoning-plus-content
produces false negatives.

To reproduce the published numbers, run the full suite against a live server
(started by `scripts/start.sh`):

```bash
python3 benchmark/speed_test.py        # decode throughput (single + 4-way)
python3 benchmark/control_test.py      # minimal smoke/control
python3 benchmark/bench_framework.py t4   # tools + needles + code-edit (README suite)
```

The framework's `TAG` argument selects the length set — use a tag containing
`t4` for the full suite (matches README), or `fp8`/anything else for the
reduced ≤131K set. Endpoint/port/output are env-overridable (`BENCH_URL`,
`BENCH_MODEL`, `BENCH_OUT`). Numbers in this README were re-verified
2026-08-14 against the reference box (tool 10/12, all needles PASS, code-edit PASS).

## License

MIT. The recipe, scripts, and measurements are free to reuse; the model weights
are not included (Apache-2.0, from unsloth/Qwen).
