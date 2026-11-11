from unittest.mock import patch

import pytest
import torch
import torch_npu

from vllm_ascend.attention.dquant_attention import DQuantAttentionCache
from vllm_ascend.core.kv_cache_interface import DQuantAttentionSpec


@pytest.mark.parametrize("cap", [0, 8, 32])
def test_decode_batch_matches_independent_sequences(cap):
    torch.npu.set_device(0)
    torch.manual_seed(20261026)
    spec = DQuantAttentionSpec(
        block_size=4,
        num_kv_heads=2,
        head_size=256,
        dtype=torch.bfloat16,
        dquant_recent_window=4,
        dquant_key_window=4,
        dquant_key_levels=32,
        dquant_value_levels=32,
        dquant_key_container_bytes=353,
        dquant_value_container_bytes=353,
        dquant_alpha=1.0,
    )
    storage = [
        torch.zeros((24, 4, width), device="npu", dtype=torch.uint8)
        for width in (spec.key_record_bytes, spec.value_record_bytes)
    ]
    cache = DQuantAttentionCache(storage, spec, output_chunk=64)
    cache.mirror_cap = cap
    tables = torch.tensor(
        [[2, 0, 5, 1, 12, 13, 14, 15], [7, 3, 6, 4, 16, 17, 18, 19], [10, 8, 9, 11, 20, 21, 22, 23]],
        device="npu",
        dtype=torch.int32,
    )
    lengths = [8, 12, 0]
    for index, length in enumerate(lengths):
        positions = torch.arange(length, device="npu")
        slots = tables[index, positions // 4].long() * 4 + positions % 4
        key = torch.randn((length, 2, 256), device="npu", dtype=torch.bfloat16)
        cache.write_sequence(str(index), slots, key, torch.randn_like(key), length)
    for step in range(8):
        order = [0, 1, 2] if step % 2 == 0 else [2, 0, 1]
        query = torch.randn((3, 4, 256), device="npu", dtype=torch.bfloat16)
        key = torch.randn((3, 2, 256), device="npu", dtype=torch.bfloat16)
        value = torch.randn_like(key)
        slots = torch.stack([tables[i, lengths[i] // 4] * 4 + lengths[i] % 4 for i in order])
        expected = []
        for row, index in enumerate(order):
            k, v = key[row : row + 1], value[row : row + 1]
            if lengths[index]:
                old_k, old_v = cache.read(tables[index], lengths[index], query.dtype, str(index))
                k, v = torch.cat((old_k, k)), torch.cat((old_v, v))
            result, _ = torch_npu.npu_fused_infer_attention_score(
                query=query[row : row + 1],
                key=k,
                value=v,
                input_layout="TND",
                num_heads=4,
                num_key_value_heads=2,
                scale=256**-0.5,
                actual_seq_lengths=[1],
                actual_seq_lengths_kv=[k.shape[0]],
                sparse_mode=0,
            )
            expected.append(result.view(1, 4, 256))
        native = torch_npu.npu_fused_infer_attention_score
        with patch.object(torch_npu, "npu_fused_infer_attention_score", wraps=native) as fused:
            actual = cache.attention(
                query,
                tables[order],
                [1, 2, 3],
                [lengths[i] + 1 for i in order],
                256**-0.5,
                torch.empty_like(query),
                None,
                request_ids=[str(i) for i in order],
                updates=(slots, key, value),
            )
            assert fused.call_count == 1
        torch.testing.assert_close(actual, torch.cat(expected), atol=0, rtol=0)
        lengths = [length + 1 for length in lengths]
        for index, length in enumerate(lengths):
            assert cache.recent[str(index)].total == length
    cache.release(["0", "1", "2"])
    assert not cache.recent and not cache.mirror


def test_mirror_growth_has_bounded_slack():
    cache = object.__new__(DQuantAttentionCache)
    cache.spec = DQuantAttentionSpec(
        block_size=128,
        num_kv_heads=2,
        head_size=256,
        dtype=torch.bfloat16,
        dquant_key_window=64,
        dquant_recent_window=512,
    )
    cache.mirror_cap = 32768
    for target in [64, 1024, 1536, 7680, 8192, 32256, 32768]:
        rows = cache._mirror_rows(target)
        assert target <= rows <= cache.mirror_cap
        assert rows - target < 512
