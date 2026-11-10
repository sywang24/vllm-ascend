# NOVA-KV (Experimental)

NOVA-KV stores Qwen3.5 dense-text full-attention history using group-VQ keys
and packed INT2 values. At head dimension 256, each KV head occupies 134 bytes
per token: 64 key-code bytes, 64 value-code bytes and three BF16 parameters.
Sink tokens and a recent-token ring retain exact K/V. Linear-attention state
uses the existing Mamba allocator.

## Supported Configuration

This branch supports Model Runner V1 and V2 on Ascend 910B with triton-ascend and a
matching CANN toolkit. Query head dimension must be 256, with one to six query
heads per KV head. Tensor parallel calibration artifacts must match the model.
The feature is disabled unless `nova_kv` is present in additional configuration.

Disable prefix caching, speculative decoding, KV transfer, context parallelism,
FlashComm sequence parallelism and NZ KV layout. Cross-layer KV sharing, sliding
attention, ALiBi, logits soft caps and unequal K/V dimensions are unsupported.
Compressed inference requires synchronous scheduling, pipeline parallel size 1,
and the built-in NOVA scheduler. Uniform single-token decode can use
`FULL_DECODE_ONLY` graph mode; prefill and mixed batches run eagerly. Calibration
capture still requires `--enforce-eager`.

Exact-window slots follow request IDs and remain reserved while requests are
unscheduled. Finished and preempted requests release their slots. Before each
scheduling step, workers check their retained lengths. If any worker has lost a
window, the scheduler preempts that request and recomputes its history from
position zero. This check adds a host-side worker RPC per scheduling step.

## Build the Decode Kernel

The AscendC launcher is an opt-in standalone build; it does not register an
ACLNN operator or change the default wheel build.

```bash
cmake -S csrc/attention/nova_kv -B build/nova_kv \
  -DSOC_VERSION=Ascend910B3 \
  -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/ascend-toolkit/latest
cmake --build build/nova_kv -j
```

Set `direct_library_root` to the absolute build directory. Its `lib` directory
must contain `libnova_attention_shared.so` and `nova_kernel_abi.json`. The
launcher validates ABI version, FP16 LUT, 128-token tiles, head dimension and
query-group limits before loading the library. Old experimental launcher
options such as `decode_backend`, `kernel_library_root`, `direct_half_lut` and
`direct_tile_tokens` are not accepted by this integration.

## Capture and Calibrate

Capture representative prompts with the original BF16 attention backend:

```bash
VLLM_USE_V2_MODEL_RUNNER=0 vllm serve /models/Qwen3.5 \
  --no-enable-prefix-caching --enforce-eager \
  --additional-config '{"nova_kv":{"capture_path":"/data/nova-capture"}}'
```

Capture writes one reservoir and query-moment file per layer and TP rank.
Use a new capture directory for each model, TP layout and calibration dataset.
Keep at least 256 samples; all rank files must come from the same capture run.
Calibration can run on CPU:

```bash
python examples/nova_kv/calibrate.py --capture /data/nova-capture \
  --output /data/nova-artifacts --device cpu --iterations 25
```

## Serve With Compressed History

Select V2 explicitly:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve /models/Qwen3.5 \
  --no-enable-prefix-caching --no-async-scheduling --kv-cache-dtype auto \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --additional-config '{"nova_kv":{"artifact_path":"/data/nova-artifacts","direct_library_root":"/workspace/vllm-ascend/build/nova_kv"}}'
```

The V1 implementation and entry point remain available: use the same command
with `VLLM_USE_V2_MODEL_RUNNER=0`. Both runners require synchronous scheduling.
V2's asynchronous scheduling is not supported by NOVA's window-recovery protocol.
Use `--enforce-eager` instead of the compilation configuration for eager execution.

V2 retains NOVA window slots by request ID independently of its permanent request
state rows. Each step gathers slots in the actual input-batch order into a stable
device buffer. Graph padding and capture requests use `-1` and cannot write real
windows or cache pages. Full-attention layers use packed byte views; linear
attention keeps the existing hybrid model-state lifecycle and allocator.

| Option | Default | Constraint |
| --- | --- | --- |
| `artifact_path` | unset | Required for compression; excludes `capture_path` |
| `capture_path` | unset | Required for capture; excludes `artifact_path` |
| `direct_library_root` | unset | Required for compression |
| `sink_tokens` | 192 | Nonnegative |
| `recent_tokens` | 256 | Positive; ring allocates one extra current-token slot |
| `codec_batch_size` | 256 | Positive; bounds VQ distance workspace |
| `sample_tokens` | 8192 | Positive; capture reservoir size |
| `direct_split_tokens` | 1024 | 128, 256, 512 or 1024 |
| `prefill_query_tile` | 2048 | 1 through 2048 |
| `prefill_kv_tile` | 2048 | Positive |

## Memory Accounting

For `L` full-attention layers, local KV heads `H`, head dimension `D=256`,
active history tokens `T`, capacity `B`, sink size `S`, recent size `R` and
query groups `G`, the packed payload is `L * H * T * 134` bytes. The BF16
full-attention payload is `L * H * T * 1024` bytes, a payload ratio of 7.64x.

Additional memory includes allocator page padding, artifacts and their FP32
copies, current K/V, transformed queries, LUTs, FIA tiles and
runtime reservations. The exact windows allocate
`L * (B + 1) * (S + R + 1) * H * D * 4` bytes in BF16.
Decode split arrays alone allocate
`L * B * H * G * ceil(max_model_len / direct_split_tokens) * (D + 16) * 4`
bytes, plus outputs and offset tables reported by `DirectDecode.workspace_bytes()`.
Fixed state is allocated during memory profiling. Measure process peak memory
for the actual batch/context configuration; 7.64x is not the total-device-memory
compression ratio, especially for hybrid models with unchanged linear state.

## Validation

### V2 Operator Interfaces

Model Runner V2 selects `NovaV2AttentionBackend`. Its implementation calls the
five interfaces in `vllm_ascend/ops/nova_kv.py`: `nova_quant_cache`,
`nova_quant_query_lut`, `nova_quant_flash_attention_metadata`,
`nova_quant_flash_attention` and `nova_exact_window_update`.
These are Python composition interfaces over the reference codec, Triton kernels,
standalone AscendC DirectDecode kernel and FIA, rather than registered CANN ops.
V1 retains its existing attention implementation.

The metadata interface uses host tensor geometry to choose the context bound and
report split-workspace bytes. Decode lengths and request slots remain device
tensors, including graph padding. Query LUT generation is separate from
DirectDecode, which consumes the supplied LUT without generating it again.
Prefill reads historical windows before updating them, retaining bounded FIA
reconstruction and causal attention over the current chunk.

Rebuild the standalone kernel for allocations that need 64-bit page offsets.
The generated ABI manifest advertises `page_offset_bits: 64`; V2 rejects an
oversized cache when an older library lacks this capability. The launcher ABI
and artifact encoding are unchanged. This refactor alone makes no throughput
or quality claim.

### Test Commands

CPU tests cover packing, codec geometry, tensor-parallel artifacts, exact-window
ownership and recovery. NPU tests build the standalone kernel and cover
numerical Decode results, paged writes, padding and graph replay:

```bash
pytest -q tests/ut/quantization/methods/kv_cache/test_nova_kv.py tests/ut/test_nova_config.py
pytest -q tests/ut/attention/test_nova_request_state.py
pytest -q tests/ut/worker/v2/test_nova_kv.py tests/ut/worker/test_model_runner_v2_mamba.py
pytest -q tests/ut/ops/test_nova_kv.py tests/ut/patch/worker/test_patch_nova_kv.py
pytest -q tests/e2e/pull_request/one_card/test_nova_kv.py
```

This port changes the original experimental integration to the current vLLM
interfaces. It still requires full-model numerical validation against BF16,
chunked-prefill/decode tests, and model-level quality/throughput
measurements before production use. Earlier experimental measurements do not
validate this revision.
