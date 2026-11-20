# Experimental NOVA Cube attention

NOVA Cube uses calibrated compressed K/V history with exact sink/recent windows.
Select it with `nova_kv.backend="cube"` under `--additional-config` on Model
Runner V2. It does not require the DirectDecode shared library. Calibration
artifacts are still required; see [NOVA-KV](nova_kv.md).

## Prototype scope

- Calibrated dense `qwen3_5_text` full attention, D256, up to six query groups/KV head.
- BF16 weights, Model Runner V2, synchronous scheduling, PP1, batch at most four.
- Eager or `FULL_DECODE_ONLY` graph mode, context at most 96K.
- No prefix cache, speculative decoding, KV transfer or context parallelism.
- A5 is the measured target. Model/task accuracy is still pending.

Compressed allocator pages persist across steps. Prefill temporarily reconstructs
one request's current layer K/V for FIA. Decode uses lookup QK and on-chip INT2
V unpacking for Cube PV, with a separate exact window and FP32 LSE reduction.
There is no persistent full-history BF16 mirror. The measured batch-4 tuning is
fixed in this prototype, without the benchmark-only environment overrides.

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve /path/to/model \
  --tensor-parallel-size 2 --dtype bfloat16 --kv-cache-dtype bfloat16 \
  --max-model-len 91024 --max-num-seqs 4 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 --no-enable-prefix-caching \
  --no-async-scheduling --enable-chunked-prefill \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,3,4]}' \
  --additional-config '{"nova_kv":{"backend":"cube","artifact_path":"/path/to/artifacts","sink_tokens":64,"recent_tokens":256}}'
```

## Capacity evidence

On Qwen3.8-27B, TP2, the same two A5 cards, 90% GPU memory utilization,
91,024 maximum tokens/request, batch 4 and an 8,192-token chunk budget:

```text
BF16: NPU KV cache size: 1,790,138 tokens, Maximum concurrency for 91,024 tokens per request: 19.67x
Cube: NPU KV cache size: 11,219,905 tokens, Maximum concurrency for 91,024 tokens per request: 123.26x
```

This is **6.27x allocator KV token capacity**, or 9,429,767 more tokens
(526.76%). Concurrency is an allocator estimate, not validated serving
concurrency. Only batch 1 and 4 were exercised; active batch above four is
rejected by the prototype. Engine capacity is not summed across TP ranks.

D256 logical records occupy 134 bytes/token/KV head versus 1,024 bytes for
BF16 K+V, a theoretical 7.64x record ratio. Whole-engine capacity also reflects
padding, hybrid linear-attention state, exact windows, codebooks and workspaces.
Transient prefill BF16 staging uses about 176 MiB/rank at 90K with two local KV heads.

## Performance evidence

One warmed unprofiled HTTP wave per arm, same A5 pair and prompts, no preemption
or sampled competing process; each request generates 1,024 tokens:

| TP2 workload | BF16 Decode tok/s | Cube Decode tok/s | BF16 TPOT ms | Cube TPOT ms |
| --- | ---: | ---: | ---: | ---: |
| 90K input / batch 4 | 125.40 | 112.34 | 48.77 | 54.11 |
| 30K input / batch 1 | 50.89 | 45.59 | 19.63 | 21.91 |

The 10% maximum degradation target is not achieved. Pure B4 step time is
35.61 ms versus BF16 31.90 ms. B4 TPOT also includes requests decoding
alongside other requests' prefill. These results are from the measured source
snapshot, not a claim of compatibility with current upstream main.

Separate 30K/B1 Decode profiling, 12 steps, rank 0, totals across 16 layers:

| Kernel | BF16 ms/step | Cube ms/step |
| --- | ---: | ---: |
| `FusedInferAttentionScore` | 3.5557 | - |
| `_score_persistent` | - | 0.9234 |
| `_split_weights` | - | 0.1578 |
| `_pv_prepared` | - | 0.5576 |
| `_merge_compressed` | - | 0.3806 |
| `_exact_and_cold` | - | 1.9826 |

Kernel totals are not end-to-end critical-path measurements. Encoding and
Query/LUT preparation are additional. Batch 1 still uses the older exact-window
merge; batch 4 has fused encoding/projection, one-pass reduction and ordered
side-stream execution. Full task accuracy for this combination has not been
evaluated; texts are not uniformly identical across runs. This branch remains
experimental pending API review, quality, device and upstream coverage.

## Branch validation and baseline

The independent branch is based on pinned NOVA commit `ba4d720b7` to preserve
the framework ABI used in the experiment. It is not rebased onto current
upstream main. Before a PR is opened, the allocator, graph and runner changes
must be ported and checked against the intended upstream base.

Relocated branch checks include 36 configuration/backend tests and an A5
comparison against the measured cold-Decode implementation at 30K/B1,
90K/B4 and empty/sink/window boundaries, including captured graph replay.
All three kernel comparisons produced zero maximum absolute output error.
An A5 TP2 smoke run also completed 30K/B1 and 90K/B4 chunked prefill,
graph capture and 16 generated tokens per request through the packaged backend.
These checks do not replace a model/task accuracy evaluation or repeat of the
full performance benchmark after porting.

Ruff, codespell, typos, Markdown lint, import/package checks, logger checks,
long-function documentation and symbolic-meta checks passed. The complete
`format.sh ci` run did not pass on Windows: clang-format crashed, the shell
hooks lacked a compatible Bash/ShellCheck setup, and actionlint installation
did not complete. No C++, shell or workflow files are changed in this commit.
The Gitleaks hook remains to be rerun in a compatible environment.

The module holds one cached side stream per device in each worker process.
Synchronous scheduling serializes calls; the current stream waits for all
side-stream writes before the backend returns. Query/LUT and split buffers
are graph-owned and sized from fixed batch/head/context geometry. Concurrent
or asynchronous model execution is outside this prototype's supported scope.
