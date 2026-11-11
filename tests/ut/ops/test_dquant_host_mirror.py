from unittest.mock import patch

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.attention.dquant_attention import DQuantAttentionCache
from vllm_ascend.core.kv_cache_interface import DQuantAttentionSpec


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("host_limit", [0, 24, 128])
def test_host_overflow_is_exact_incremental_bounded_and_released(dtype, host_limit):
    torch.npu.set_device(0)
    spec = DQuantAttentionSpec(
        block_size=64, num_kv_heads=2, head_size=128, dtype=dtype, dquant_key_window=4, dquant_recent_window=8
    )
    storage = [
        torch.zeros(4, 64, width, device="npu", dtype=torch.uint8)
        for width in (spec.key_record_bytes, spec.value_record_bytes)
    ]
    cache = DQuantAttentionCache(storage, spec, output_chunk=64)
    cache.mirror_cap = 16
    cache.host_mirror_cap = host_limit
    table = torch.tensor([3, 0, 2, 1], device="npu", dtype=torch.int32)
    keys = torch.randn(64, 2, 128, device="npu", dtype=dtype)
    values = torch.randn_like(keys)
    tokens = torch.arange(64, device="npu")
    slots = table[tokens // 64].long() * 64 + tokens % 64
    cache.write_sequence("a", slots[:28], keys[:28], values[:28], 28)
    cache.read(table, 28, dtype, "a")
    decode = cache.read_compressed_range
    with patch.object(cache, "read_compressed_range", wraps=decode) as read:
        for end in (28, 29, 33, 37, 44):
            if cache.recent["a"].total < end:
                begin = cache.recent["a"].total
                cache.write_sequence("a", slots[begin:end], keys[begin:end], values[begin:end], end)
            actual = cache.read(table, end, dtype, "a")
            compressed = cache.recent["a"].compressed
            reference = decode(table, 0, compressed, dtype)
            tail = cache.recent["a"]
            for result, prefix, recent in zip(actual, reference, (tail.key, tail.value)):
                torch.testing.assert_close(result, torch.cat((prefix, recent[: tail.length])), atol=0, rtol=0)
        before_calls = read.call_count
        cache.read(table, 44, dtype, "a")
        if host_limit >= 44:
            assert read.call_count == before_calls
    if host_limit > cache.mirror_cap:
        assert cache.host_mirror_storage_bytes == (host_limit - 16) * 2 * 128 * 2 * 2
        assert cache.host_mirror["a"][2].is_pinned()
    else:
        assert cache.host_mirror_storage_bytes == 0
    assert cache.mirror["a"][0] <= 16
    cache.release(["a"])
    assert cache.host_mirror_storage_bytes == 0
    assert not cache.mirror_tail_end and not cache.host_mirror


def test_pinned_overflow_transfers_are_ordered_without_host_synchronization():
    torch.npu.set_device(0)
    spec = DQuantAttentionSpec(block_size=64, num_kv_heads=2, head_size=128, dtype=torch.bfloat16)
    storage = [
        torch.zeros(2, 64, width, device="npu", dtype=torch.uint8)
        for width in (spec.key_record_bytes, spec.value_record_bytes)
    ]
    cache = DQuantAttentionCache(storage, spec)
    cache.host_mirror_cap = 128
    table = torch.tensor([1, 0], device="npu", dtype=torch.int32)
    for repeat in range(20):
        keys = torch.randn(64, 2, 128, device="npu", dtype=spec.dtype)
        values = torch.randn_like(keys)
        cache.host_mirror.clear()

        def read(table, start, end, dtype, keys=keys, values=values):
            return keys[start:end], values[start:end]

        with patch.object(cache, "read_compressed_range", side_effect=read):
            actual = cache._read_overflow(table, 16, 64, spec.dtype, "a")
            torch.testing.assert_close(actual[0], keys[16:], atol=0, rtol=0)
            torch.testing.assert_close(actual[1], values[16:], atol=0, rtol=0)
