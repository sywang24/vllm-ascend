# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm_ascend.attention.dquant_attention import DQuantAttentionCache, RecentKVState
from vllm_ascend.core.kv_cache_interface import DQuantAttentionSpec
from vllm_ascend.ops.triton.dquant import hadamard_rotate_npu
from vllm_ascend.ops.triton.dquant_finalize import finalize_dquant_attention_npu
from vllm_ascend.ops.triton.dquant_fused import merge_dquant_attention_partials_npu, raw_dquant_attention_npu

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="Ascend NPU is unavailable")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("heads", [4, 6, 12])
@pytest.mark.parametrize("chunks,tail,has_current", [(1, 0, False), (13, 191, True), (509, 31, True)])
def test_fused_finalize_matches_separate_kernels(dtype, heads, chunks, tail, has_current):
    torch.manual_seed(1031)
    width, kv_heads = 256, 2
    parts = torch.randn((chunks, heads, width), device="npu")
    lses = torch.randn((chunks, heads), device="npu") * 5
    query = torch.randn((1, heads, width), dtype=dtype, device="npu")
    key = torch.randn((tail, kv_heads, width), dtype=dtype, device="npu")
    value = torch.randn_like(key)
    current = (
        tuple(torch.randn((1, kv_heads, width), dtype=dtype, device="npu") for _ in range(2)) if has_current else None
    )
    scale = width**-0.5
    expected, prefix_lse = merge_dquant_attention_partials_npu(parts, lses)
    expected = hadamard_rotate_npu(expected)
    if tail or current is not None:
        raw_key = key if current is None else torch.cat((key, current[0]))
        raw_value = value if current is None else torch.cat((value, current[1]))
        raw, raw_lse = raw_dquant_attention_npu(query, raw_key, raw_value, scale)
        expected, _ = merge_dquant_attention_partials_npu(
            torch.stack((expected, raw)), torch.stack((prefix_lse, raw_lse))
        )
    output = torch.empty(query.shape, dtype=torch.float32, device="npu")
    actual = finalize_dquant_attention_npu(parts, lses, query, key, value, scale, current, output=output)
    assert actual is output
    torch.testing.assert_close(actual[0], expected, atol=2e-5, rtol=2e-5)
    typed_output = torch.empty_like(query)
    actual = finalize_dquant_attention_npu(parts, lses, query, key, value, scale, current, output=typed_output)
    assert actual is typed_output
    # The changed reduction order can straddle an FP16/BF16 rounding midpoint.
    torch.testing.assert_close(actual.float(), expected[None], atol=2e-5, rtol=torch.finfo(dtype).eps)


@pytest.mark.parametrize("prefix_lse", [-1000.0, 1000.0])
def test_fused_finalize_keeps_extreme_lse_finite(prefix_lse):
    query = torch.zeros((1, 4, 256), dtype=torch.bfloat16, device="npu")
    parts = torch.zeros((13, 4, 256), device="npu")
    lses = torch.full((13, 4), prefix_lse, device="npu")
    empty = torch.empty((0, 2, 256), dtype=query.dtype, device="npu")
    current = (
        torch.zeros((1, 2, 256), dtype=query.dtype, device="npu"),
        torch.ones((1, 2, 256), dtype=query.dtype, device="npu"),
    )
    actual = finalize_dquant_attention_npu(parts, lses, query, empty, empty, 256**-0.5, current)
    expected = torch.ones_like(query) if prefix_lse < 0 else torch.zeros_like(query)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.fixture(scope="module", params=[torch.float16, torch.bfloat16])
def short_prefix_cache(request):
    if torch.npu.get_device_name() != "Ascend910B3":
        pytest.skip("Short-prefix dispatch is calibrated for Ascend910B3")
    with pytest.MonkeyPatch.context() as environment:
        environment.setenv("VLLM_ASCEND_DQUANT_FUSED_ATTENTION", "1")
        environment.setenv("VLLM_ASCEND_DQUANT_SYMBOL_CACHE", "0")
        environment.setenv("VLLM_ASCEND_DQUANT_FUSED_TILE_TOKENS", "64")
        environment.setenv("VLLM_ASCEND_DQUANT_FUSED_CHUNK_TOKENS", "64")
        torch.manual_seed(1031)
        spec = DQuantAttentionSpec(
            block_size=128,
            num_kv_heads=2,
            head_size=256,
            dtype=request.param,
            dquant_recent_window=128,
            dquant_key_window=64,
            dquant_key_levels=8,
            dquant_value_levels=6,
            dquant_alpha=1.4,
            dquant_key_container_bytes=155,
            dquant_value_container_bytes=109,
        )
        total = 1344 + 128
        blocks = (total + 127) // 128
        storage = [
            torch.zeros((blocks, 128, w), dtype=torch.uint8, device="npu")
            for w in (spec.key_record_bytes, spec.value_record_bytes)
        ]
        cache = DQuantAttentionCache(storage, spec)
        table = torch.arange(blocks, dtype=torch.int32, device="npu").flip(0).contiguous()
        positions = torch.arange(total, device="npu")
        slots = table[positions // 128].long() * 128 + positions % 128
        key = torch.randn((total, 2, 256), dtype=request.param, device="npu")
        value = torch.randn_like(key)
        cache.write_sequence("a", slots, key, value, total)
        assert cache.recent["a"].compressed == 1344
        yield cache, table, key, value, slots


@pytest.mark.parametrize("compressed,tile", [(64, 16), (640, 16), (704, 32), (1280, 32), (1344, 64)])
def test_short_prefix_dispatch_matches_fixed_tile(short_prefix_cache, compressed, tile, monkeypatch):
    cache, table, key, value, slots = short_prefix_cache
    dtype = key.dtype
    total = compressed + 128
    # Mean windows are independent, so the longer cache also covers each prefix.
    state = RecentKVState(
        key[compressed:total],
        value[compressed:total],
        slots[compressed:total],
        length=128,
        total=total,
        compressed=compressed,
    )
    cache.recent["a"] = state
    query = torch.randn((1, 12, 256), dtype=dtype, device="npu")
    current = tuple(torch.randn((1, 2, 256), dtype=dtype, device="npu") for _ in range(2))
    original = cache._rans_attention_partials
    parts, lses = original(query, table, compressed, 256**-0.5, 64, 64)
    expected = finalize_dquant_attention_npu(
        parts,
        lses,
        query,
        state.key[: state.length],
        state.value[: state.length],
        256**-0.5,
        current,
        output=torch.empty(query.shape, dtype=torch.float32, device="npu"),
    )
    seen = []

    def track(*args):
        seen.append(args[-2:])
        return original(*args)

    monkeypatch.setattr(cache, "_rans_attention_partials", track)
    actual = cache._fused_decode_attention(
        query, table, total, 256**-0.5, "a", current, output=torch.empty(query.shape, dtype=torch.float32, device="npu")
    )
    assert seen == [(tile, tile)]
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
