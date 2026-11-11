# Single-Stream Ascend C Attention

This optional backend specializes the existing D-Quant prefix attention layout:
Ascend910B3, 12 BF16 query heads, two KV heads, head size 256, pages of 128 tokens,
64-token mean windows, and K/V records of 176/114 bytes. The rANS payloads remain
155/109 bytes. Unsupported layouts and installations without the native library
continue to use Triton.

The decoder copies one 64-token tile into UB, expands payload bytes within UB,
uses native vector gathers for the packed rANS table, and folds scale and offset
into QK/PV. Only attention partials and log-sum-exp values are written to HBM.
QK uses scalar-vector multiply-add. PV reduces the six grouped query heads in one
vector instruction and batches 32 channels before vector packing and writeback.
The PV loop reuses the query region after QK and LSE computation, preserving the
LSE entries. No per-channel scalar result copy or scalar/vector handoff is needed. Low-byte
state and symbol extraction uses integer bit masks. These changes preserve the
record format and exact decoded symbols, final rANS state and byte cursor.
The implementation reserves 129,536 bytes of UB per vector core. It does not
allocate a dense KV mirror or a persistent symbol cache.

Each vector core scans the device-side compressed lengths and assigns live
64-token tiles by their cumulative index modulo the number of cores. The task
count per core differs by at most one, including uneven and empty requests.
This avoids the fixed 128-partition stride that leaves cores idle for short
prefixes and creates abrupt workload imbalance as requests grow. Output tile
indices, rANS streams and the launch ABI remain unchanged. Graph replay reads
the current lengths without a CPU transfer or an intermediate HBM task list.

After sourcing the CANN environment, build from the repository root:

```sh
cmake -S csrc/dquant -B build_dquant_native \
  -DASCEND_CANN_PACKAGE_PATH="$ASCEND_HOME_PATH" -DCMAKE_BUILD_TYPE=Release
cmake --build build_dquant_native -j4
cmake --install build_dquant_native --prefix "$PWD"
```

The installed library is `vllm_ascend/libdquant_native.so`. Each cache owns its
backend handle. The supported decode path selects it automatically when that
library is present. Removing the optional library before launching a new service
restores Triton selection.

The native NPU regression requires this library to be installed:

```sh
pytest -q tests/ut/ops/test_dquant_native.py tests/ut/ops/test_dquant_full_decode.py \
  tests/ut/ops/test_dquant_channel_partition.py
```

The tests cover batch sizes 1/4/32, permuted pages, exact inactive-buffer guards, random, zero and
independently scaled query heads, changing query
values and compressed lengths after capture, all-empty replay, and stateful
request reordering, cache flushes, request replacement and stale-prefix rejection.
Model accuracy must be evaluated separately; kernel tolerances do not establish
GPQA equivalence.

An optional probe checks exact symbols, final integer states and byte cursors
against a CPU decoder for K/V, including partial tiles and graph replay after
changing records. Build it with `-DDQUANT_BUILD_TEST_KERNELS=ON` in
`build_dquant_native`, then run `pytest -q tests/ut/ops/test_dquant_native_state.py`.
The probe is disabled by default and is not loaded by the attention backend.
