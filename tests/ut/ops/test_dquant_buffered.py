"""Check buffered rANS reads against encoded symbols and the scalar kernel."""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.attention.dquant_attention import DQuantAttentionCache
from vllm_ascend.core.kv_cache_interface import DQuantAttentionSpec
from vllm_ascend.ops.triton.dquant import paged_rans_dequantize_npu
from vllm_ascend.quantization.methods.kv_cache.dquant_codec import (
    _rans_encode,
    frequencies_from_probabilities,
)
from vllm_ascend.quantization.methods.kv_cache.dquant_npu_cache import DQuantNPUCache


@pytest.mark.parametrize("width", [37, 128, 1024])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_buffered_paged_read(width, dtype):
    table = frequencies_from_probabilities([0.5, 0.25, 0.125, 0.125])
    patterns = [[(i + slot) % 4 for i in range(width)] for slot in range(8)]
    encoded = [_rans_encode(pattern, table) for pattern in patterns]
    container = max(len(payload) for payload in encoded) + 4
    metadata_offset = (container + 1) & ~1
    storage = torch.zeros((8, metadata_offset + 36), dtype=torch.uint8, device="npu")
    cache = DQuantNPUCache(8, width, container, 4, device="npu", storage=storage)
    # Use an explicit probability model independent of the cache's Gaussian defaults.
    expanded = [[], [], []]
    for symbol, (frequency, cumulative) in enumerate(zip(table.frequencies, table.cumulative)):
        for values, item in zip(expanded, (symbol, frequency, cumulative)):
            values.extend([item] * frequency)
    luts = tuple(torch.tensor(values, dtype=torch.int32, device="npu") for values in expanded)
    cache.scales.fill_(0.125)
    cache.offsets.fill_(-0.375)
    for slot, payload in enumerate(encoded):
        storage[slot, : len(payload)] = torch.tensor(list(payload), dtype=torch.uint8, device="npu")
    block_table = torch.tensor([3, 1, 2, 0], dtype=torch.int32, device="npu")
    args = (storage, block_table, 5, 2, container, width, cache.scales, cache.offsets, luts, dtype)
    expected_slots = [7, 2, 3, 4, 5]
    expected = (
        torch.tensor([patterns[i] for i in expected_slots], dtype=torch.float32, device="npu") * 0.125 - 0.375
    ).to(dtype)
    scalar = paged_rans_dequantize_npu(*args, start_token=1, output_chunk=1)
    torch.testing.assert_close(scalar, expected, rtol=0, atol=0)
    for chunk in (16, 32, 64):
        output = paged_rans_dequantize_npu(*args, start_token=1, output_chunk=chunk)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
    empty = paged_rans_dequantize_npu(
        storage, block_table, 0, 2, container, width, cache.scales, cache.offsets, luts, dtype
    )
    assert empty.shape == (0, width)
    with pytest.raises(ValueError, match="output_chunk"):
        paged_rans_dequantize_npu(*args, output_chunk=3)


def test_buffered_cache_read_and_range():
    spec = DQuantAttentionSpec(
        block_size=8, num_kv_heads=4, head_size=256, dtype=torch.bfloat16, dquant_recent_window=4, dquant_key_window=4
    )
    storage = [
        torch.zeros((3, 8, width), dtype=torch.uint8, device="npu")
        for width in (spec.key_record_bytes, spec.value_record_bytes)
    ]
    cache = DQuantAttentionCache(storage, spec)
    assert cache.output_chunk == 1
    table = torch.tensor([2, 0, 1], dtype=torch.int32, device="npu")
    slots = (table.long()[:, None] * 8 + torch.arange(8, device="npu")).flatten()[:20]
    torch.manual_seed(98)
    key = torch.randn((20, 4, 256), dtype=torch.bfloat16, device="npu")
    cache.write_sequence("a", slots, key, torch.randn_like(key), 20)
    expected = cache.read(table, 20, key.dtype, "a")
    cache.set_output_chunk(32)
    for actual, reference in zip(cache.read(table, 20, key.dtype, "a"), expected):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    for actual, reference in zip(cache.read_compressed_range(table, 4, 12, key.dtype), expected):
        torch.testing.assert_close(actual, reference[4:12], rtol=0, atol=0)
    with pytest.raises(ValueError, match="output chunk"):
        cache.set_output_chunk(3)
