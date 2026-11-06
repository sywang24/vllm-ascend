# NOVA-KV Integration Review

Base: `c15cdac3ca7ebcc61b2f6e1cfec6ecc9b219828a` in the user's vLLM Ascend
repository. Source: the local `staging_free_20261030` codec, bounded prefill,
exact-window operators and the KV-reuse half-LUT AscendC decode kernel.

## Findings Addressed

| Priority | Finding | Resolution |
| --- | --- | --- |
| P1 | The prototype changed an initialized attention object's class and targeted an older runner path. | Pass the backend through the existing constructor; integrate packed views in `worker/v1/model_runner.py`. |
| P1 | Decode could consume an exact window whose request history was evicted. | Reserve slots by request lifetime, check retained lengths on workers before scheduling, and preempt/recompute only requests with missing history. |
| P1 | Backend cache customization discarded physical page padding. | Preserve padding and byte-storage offsets; accept the allocator's signed or unsigned byte backing. |
| P1 | Persistent FP32 codebooks and codec/LUT temporaries could appear after memory profiling. | Initialize persistent codec copies eagerly and exercise codec/LUT allocation in the profiling path. |
| P1 | History assertions failed under NPU graph replay with error 507011. | Remove device ownership assertions, mask padding, retain persistent request-slot inputs and fix split launch geometry for uniform decode graphs. |
| P2 | TP artifact slicing accepted incompatible head geometry. | Validate partition/replication geometry, rank, tensor shape, version and finite data. |
| P2 | Empty encoding and reconstruction used ambiguous reshapes or empty concatenation. | Return typed empty packets and use explicit dimensions. |
| P2 | Launch configuration silently accepted typos and depended on unrelated external libraries. | Use the strict additional-config schema and ship one standalone half-LUT kernel plus ABI manifest. |
| P2 | Padded zero-query requests could read beyond token slot metadata. | Mask token-slot loads using query intervals and zero padded output. |
| P2 | CMake's empty default build type left the CANN merge command incomplete. | Select Release when no build type was provided; rebuild succeeded. |

## Structure

- Backend and metadata: `vllm_ascend/attention/nova_kv.py`.
- Codec and artifacts: `vllm_ascend/quantization/methods/kv_cache/nova_kv/`.
- Triton operators and launcher: `vllm_ascend/ops/triton/nova_*.py`.
- AscendC source: `csrc/attention/nova_kv/` with an opt-in standalone build.
- Minimal upstream patch: `vllm_ascend/patch/worker/patch_nova_kv.py`.
- Recovery scheduler: `vllm_ascend/core/nova_scheduler.py`, selected through the existing engine-core platform patch.
- Calibration: `examples/nova_kv/calibrate.py`.
- User configuration and memory formulas: the NOVA-KV feature guide.

The runner reshape change is needed because its current generic hook assumes
dense K/V planes. The upstream patch lifecycle and removal plan are recorded in
`vllm_ascend/patch/__init__.py`.

## Validation and Remaining Risk

On 2026-11-06, 46 CPU tests passed using an external isolated harness that mocked
vLLM/device imports while executing the branch's codec, config, patch and ABI
tests. This does not establish compatibility with a complete installed vLLM
runtime. The standard pytest invocation stopped during collection because that
runtime was absent on the validation host.

Ruff checks/formatting, Markdown lint and spelling checks passed for the changed
files. Python package-init and boolean-context checks also passed. The required
`bash format.sh ci` was attempted in a Linux validation copy: after normalizing
Windows line endings and selecting the installed Node runtime for that copy,
Actionlint setup failed downloading Go dependencies from `proxy.golang.org`.
The repository's pre-commit configuration was not changed.

The standalone kernel compiled on Ascend 910B3. Five NPU cases passed: native
Decode against reconstructed attention for one and six query groups, padded
page writes, request reordering/empty-query padding, and rejection of lost
history in an isolated process. The operator harness
used real torch-npu and triton-ascend, replacing the legacy container's broken
`vllm.triton_utils` import adapter. It did not load the port's full serving stack.

The graph/recovery follow-up on 2026-11-06 passed 52 isolated CPU codec,
configuration, patch and ABI tests; six request-state/recovery tests using the
installed vLLM scheduler; and 13 engine-core patch tests with unrelated platform
initializers isolated. Recovery tests verify that upstream preemption frees
cache state, preserves prompt/output tokens, resets progress and queues the
request for recomputation from zero.

The follow-up's available lint hooks passed, including Ruff, Markdown lint,
spelling, secret scanning and Python source checks. The unmodified full format
script could not complete: Node bootstrap failed, Actionlint's Go dependencies
could not be downloaded, and ShellCheck was absent. An external validation
configuration selected the installed Node runtime and skipped the two
unavailable workflow/shell tools; this change contains no workflow or shell
edits.

Five follow-up NPU cases passed, covering both one and six query heads per KV
head. The full NOVA attention forward passed graph replay against eager execution
and independently reconstructed attention at lengths 33, 171 and 389 after
capture with dummy length 1. The cases cover request-slot changes, padded rows
and multiple decode splits. CPU/device arithmetic and the standalone kernel are
real; runner imports are isolated because the container's older vLLM stack
lacks current repository interfaces. This is an attention-level graph test,
not a complete serving test.

Compressed inference supports `NONE` and `FULL_DECODE_ONLY`, synchronous
scheduling and pipeline parallel size 1. Calibration capture remains eager.
The retained-length check introduces one host worker RPC per active scheduling
step. Model Runner V2 is rejected. Complete serving, chunked-prefill numerics,
preemption workloads, quality, throughput and memory measurements still require
validation before production use.

The release fragment uses PR number zero until an upstream PR is assigned.
