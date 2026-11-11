from unittest.mock import patch

import pytest
import torch

from vllm_ascend.attention.dquant_attention import DQuantAttentionCache
from vllm_ascend.core.kv_cache_interface import DQuantAttentionSpec


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("cap", [0, 16, 1024])
def test_decode_reuse_matches_original_across_flush_growth_and_cap(dtype, cap):
    torch.npu.set_device(0)
    torch.manual_seed(20261027)
    spec = DQuantAttentionSpec(
        block_size=64,
        num_kv_heads=2,
        head_size=256,
        dtype=dtype,
        dquant_recent_window=8,
        dquant_key_window=4,
        dquant_key_levels=128,
        dquant_value_levels=128,
        dquant_key_container_bytes=481,
        dquant_value_container_bytes=481,
        dquant_alpha=1.0,
    )

    def make():
        storage = [
            torch.zeros((12, 64, width), device="npu", dtype=torch.uint8)
            for width in (spec.key_record_bytes, spec.value_record_bytes)
        ]
        cache = DQuantAttentionCache(storage, spec, output_chunk=64)
        cache.mirror_cap = cap
        return cache

    actual, original = make(), make()
    original._single_decode_kv = lambda *args: None
    table = torch.tensor([[3, 1, 0, 2, 5, 4, 7, 6, 9, 8, 11, 10]], device="npu", dtype=torch.int32)
    length = 510 if cap == 1024 else 14
    key = torch.randn((length + 30, 2, 256), dtype=dtype, device="npu")
    value = torch.randn_like(key)
    positions = torch.arange(key.shape[0], device="npu")
    slots = table[0, positions // 64].long() * 64 + positions % 64
    mask = torch.triu(torch.ones((2048, 2048), device="npu", dtype=torch.bool), diagonal=1)
    for cache in (actual, original):
        cache.write_sequence("request", slots[:length], key[:length], value[:length], length)
    reused = 0
    for index in range(length, length + 30):
        query = torch.randn((1, 4, 256), dtype=dtype, device="npu")
        current = (key[index : index + 1], value[index : index + 1])
        if cap == 1024:
            with patch.object(torch, "cat", side_effect=AssertionError("decode assembled the full KV")):
                dense = actual._single_decode_kv(table[0], index, dtype, "request", current)
            assert dense is not None
            reused += 1
            old = original.read(table[0], index, dtype, "request")
            for result, prefix, token in zip(dense, old, current):
                torch.testing.assert_close(result, torch.cat((prefix, token)), atol=0, rtol=0)
        results = []
        for cache in (actual, original):
            results.append(
                cache.attention(
                    query,
                    table,
                    [1],
                    [index + 1],
                    256**-0.5,
                    torch.empty_like(query),
                    mask,
                    request_ids=["request"],
                    updates=(slots[index : index + 1], *current),
                )
            )
        torch.testing.assert_close(results[0], results[1], atol=0, rtol=0)
        assert actual.recent["request"].compressed == original.recent["request"].compressed
    if cap == 1024:
        assert reused == 30
        assert actual.mirror["request"][1] == 1024
    actual.release(["request"])
    assert not actual.recent and not actual.mirror
    assert not actual.mirror_tail_end
    actual.write_sequence("request", slots[:2], key[:2] + 1, value[:2], 2)
    recovered, _ = actual.read(table[0], 2, dtype, "request")
    torch.testing.assert_close(recovered, key[:2] + 1, atol=0, rtol=0)


def test_decode_reuse_skips_unchanged_tail_and_recovers_after_gap():
    torch.npu.set_device(0)
    spec = DQuantAttentionSpec(
        block_size=64, num_kv_heads=2, head_size=128, dtype=torch.bfloat16, dquant_recent_window=8, dquant_key_window=4
    )
    storage = [
        torch.zeros(4, 64, width, device="npu", dtype=torch.uint8)
        for width in (spec.key_record_bytes, spec.value_record_bytes)
    ]
    cache = DQuantAttentionCache(storage, spec, output_chunk=64)
    cache.mirror_cap = 128
    table = torch.tensor([3, 0, 2, 1], device="npu", dtype=torch.int32)
    key = torch.randn(40, 2, 128, device="npu", dtype=spec.dtype)
    value = torch.randn_like(key)
    tokens = torch.arange(40, device="npu")
    slots = table[tokens // 64].long() * 64 + tokens % 64
    cache.write_sequence("a", slots[:16], key[:16], value[:16], 16)
    cache._single_decode_kv(table, 16, spec.dtype, "a", (key[16:17], value[16:17]))
    cache.write_sequence("a", slots[16:17], key[16:17], value[16:17], 17)
    copy_sizes = []
    original_copy = torch.Tensor.copy_

    def copy(destination, source, *args, **kwargs):
        copy_sizes.append(source.numel())
        return original_copy(destination, source, *args, **kwargs)

    with patch.object(torch.Tensor, "copy_", copy):
        cache._single_decode_kv(table, 17, spec.dtype, "a", (key[17:18], value[17:18]))
    assert copy_sizes == [256, 256]
    cache.write_sequence("a", slots[17:27], key[17:27], value[17:27], 27)
    reference = cache.read(table, 27, spec.dtype, "a")
    actual = cache._single_decode_kv(table, 27, spec.dtype, "a", (key[27:28], value[27:28]))
    for result, previous, current in zip(actual, reference, (key[27:28], value[27:28])):
        torch.testing.assert_close(result, torch.cat((previous, current)), atol=0, rtol=0)
    cache.release(["a"])
    assert not cache.mirror_tail_end
