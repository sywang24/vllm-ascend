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
There is no persistent full-history BF16 mirror. The measured batch-1/4 tuning is
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
| 90K input / batch 4 | 125.67 | 115.05 | 48.69 | 53.51 |
| 30K input / batch 1 | 51.10 | 53.61 | 19.55 | 18.63 |

These post-optimization measurements meet the 10% Decode degradation target
in both workloads. At 90K/B4, throughput is 8.45% lower, pure step time is
34.77 ms versus BF16 31.83 ms (+9.23%), and TPOT is 9.90% higher. At 30K/B1,
throughput is 4.90% higher and TPOT is 4.71% lower. B4 TPOT includes requests
decoding alongside other requests' prefill. B4 TTFT remains 10.52% higher
(32.96 s versus 29.82 s), outside a 10% TTFT target. Each result is one warmed
wave, not a statistical guarantee across runs. Results use the isolated
packaged source, not current upstream main.

Separate 30K/B1 Decode profiling, 12 steps, rank 0, totals across 16 layers:

| Kernel | BF16 ms/step | Cube ms/step |
| --- | ---: | ---: |
| `FusedInferAttentionScore` | 3.5576 | - |
| `_score_persistent` | - | 1.0367 |
| `_split_weights` | - | 0.1822 |
| `_pv_prepared` | - | 0.5585 |
| `merge_one_pass` | - | 0.3825 |
| `_exact_tiles` | - | 0.4465 |
| `_merge_exact_tiles` | - | 0.0579 |

Kernel totals are not end-to-end critical-path measurements. Encoding and
Query/LUT preparation are additional. The former 30K/B1 `_exact_and_cold`
cost 1.9826 ms/step; tiled exact attention plus merging now total 0.5044 ms
(-74.56%). Both batches use fused encoding/projection, one-pass cold reduction,
and exact tiles on the side stream. Batch 4 schedules exact tiles before
encoding; long-context B4 uses 4,096-token PV splits instead of 1,024 tokens.
Full task accuracy for this combination has not been
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

The subsequent Decode optimization passed 64 numerical cases covering both
batches, both ranks and all 16 full-attention layers, including 64 dynamic
captured graph replays and invalid slots. Compressed packets matched exactly;
maximum absolute attention-output error against the serial reference was
6.8e-6. Empty and partially filled exact windows were checked separately.
The final HTTP and profiling workloads were rerun after these changes;
raw evidence remains under `/home/w00807672/nova-kv-cube-rfc-20261120` on A5.

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
