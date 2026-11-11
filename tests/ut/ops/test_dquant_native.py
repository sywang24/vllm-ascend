# SPDX-License-Identifier: Apache-2.0
"""Native decoder: page permutation, inactive guards and dynamic graph replay."""

import inspect
from pathlib import Path

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.attention.dquant_attention import DQuantAttentionCache
from vllm_ascend.core.kv_cache_interface import DQuantAttentionSpec
from vllm_ascend.ops.triton.dquant_fused import paged_dquant_batch_attention_npu


@pytest.mark.skipif(not torch.npu.is_available(), reason="Ascend NPU unavailable")
@pytest.mark.parametrize("query_kind", ["random", "zero", "scaled_heads"])
@pytest.mark.parametrize("batch", [1, 4, 32])
def test_native_pages_and_dynamic_graph(monkeypatch, query_kind, batch):
    """Compare native attention against Triton across page changes and graph replay."""
    torch.npu.set_device(0)
    if torch.npu.get_device_name() != "Ascend910B3":
        pytest.skip("Native specialization requires Ascend910B3")
    library = Path(inspect.getfile(DQuantAttentionCache)).resolve().parents[1] / "libdquant_native.so"
    if not library.is_file():
        pytest.skip("Optional native library has not been built")
    monkeypatch.setenv("VLLM_ASCEND_DQUANT_FUSED_ATTENTION", "1")
    torch.manual_seed(11919)
    spec = DQuantAttentionSpec(
        block_size=128,
        num_kv_heads=2,
        head_size=256,
        dtype=torch.bfloat16,
        dquant_recent_window=128,
        dquant_key_window=64,
        dquant_key_levels=8,
        dquant_value_levels=6,
        dquant_key_container_bytes=155,
        dquant_value_container_bytes=109,
        dquant_alpha=1.4,
    )
    pages, maximum = 8, 1024
    storage = [torch.zeros((batch * pages, 128, width), dtype=torch.uint8, device="npu") for width in (176, 114)]
    cache = DQuantAttentionCache(storage, spec, max_num_seqs=batch)
    assert cache.native_attention is not None, "Build and install csrc/dquant before this test"
    print("Native source:", inspect.getfile(DQuantAttentionCache), flush=True)
    print("Native library:", cache.native_attention.library._name, flush=True)
    tables = torch.stack([torch.randperm(pages, device="npu").to(torch.int32) + row * pages for row in range(batch)])
    for row in range(batch):
        positions = torch.arange(maximum, device="npu")
        slots = tables[row, positions // 128].long() * 128 + positions % 128
        keys = torch.randn((maximum, 2, 256), dtype=torch.bfloat16, device="npu")
        values = torch.randn_like(keys)
        values[:, 0] += 2
        values[:, 1] -= 3
        cache.write_sequence(str(row), slots, keys, values, maximum)
    query = torch.randn((batch, 12, 256), dtype=torch.bfloat16, device="npu")

    def refresh_query():
        query.normal_()
        if query_kind == "zero":
            query.zero_()
        elif query_kind == "scaled_heads":
            query.mul_(torch.linspace(-3, 3, 12, dtype=query.dtype, device=query.device)[None, :, None])

    refresh_query()

    def repeat_lengths(pattern):
        return [pattern[row % len(pattern)] for row in range(batch)]

    compressed = torch.tensor(repeat_lengths([64, 128, 192, 0]), dtype=torch.int32, device="npu")
    outputs = tuple(
        torch.full(shape, 173.0, dtype=torch.float32, device="npu")
        for shape in ((batch, maximum // 64, 12, 256), (batch, maximum // 64, 12))
    )

    def call(native, partials=None):
        return paged_dquant_batch_attention_npu(
            query,
            tables,
            compressed,
            256**-0.5,
            cache.key.storage,
            cache.key.scales,
            cache.key.offsets,
            cache.fused_key_lut,
            cache.value.storage,
            cache.value.scales,
            cache.value.offsets,
            cache.fused_value_lut,
            block_size=128,
            head_size=256,
            num_kv_heads=2,
            key_container_bytes=155,
            value_container_bytes=109,
            key_window=64,
            mean_offset=160,
            stripe_bytes=16,
            chunk_tokens=64,
            tile_tokens=64,
            compressed_max=maximum,
            partials=partials,
            native_backend=cache.native_attention if native else None,
        )

    def check():
        reference = call(False)
        for row, length in enumerate(compressed.cpu().tolist()):
            active = length // 64
            if active:
                for actual, expected in zip(outputs, reference[:2]):
                    torch.testing.assert_close(actual[row, :active], expected[row, :active], atol=3e-4, rtol=2e-3)
            for output in outputs:
                assert bool((output[row, active:].cpu() == 173.0).all())

    call(True, outputs)
    torch.npu.synchronize()
    check()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="global"):
        call(True, outputs)
    for pattern in ([192, 0, 64, 128], [0], [576], [0, 64, 576, 896], [64, 128, 192, 0]):
        lengths = repeat_lengths(pattern)
        compressed.copy_(torch.tensor(lengths, dtype=torch.int32, device="npu"))
        refresh_query()
        for output in outputs:
            output.fill_(173.0)
        torch.npu.synchronize()
        graph.replay()
        torch.npu.synchronize()
        check()
    assert not cache.native_attention.supports(query.to(torch.float16), tables, 64, 64)
    assert not cache.native_attention.supports(query[:, :6], tables, 64, 64)
    assert not cache.native_attention.supports(query, tables.long(), 64, 64)
    assert not cache.native_attention.supports(query, tables, 16, 16)
    compressed = compressed.long()
    actual = call(True)
    expected = call(False)
    for row, length in enumerate(compressed.cpu().tolist()):
        active = length // 64
        if active:
            for result, reference in zip(actual[:2], expected[:2]):
                torch.testing.assert_close(result[row, :active], reference[row, :active])
