from unittest.mock import patch

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.attention.dquant_attention import DQuantAttentionCache
from vllm_ascend.core.kv_cache_interface import DQuantAttentionSpec
from vllm_ascend.ops.triton.dquant import dquant_quantize_npu
from vllm_ascend.quantization.methods.kv_cache.dquant_npu_cache import DQuantNPUCache


def make_cache():
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
        torch.zeros((8, 4, width), device="npu", dtype=torch.uint8)
        for width in (spec.key_record_bytes, spec.value_record_bytes)
    ]
    return DQuantAttentionCache(storage, spec, output_chunk=64), storage


@pytest.mark.parametrize("cap", [0, 8, 10])
def test_mirror_cap_preserves_prefix_and_matches_uncached_reader(cap):
    cache, storage = make_cache()
    cache.mirror_cap = cap
    uncached = DQuantAttentionCache(storage, cache.spec, output_chunk=64)
    uncached.mirror_cap = 0
    table = torch.tensor([2, 0, 5, 1, 7, 3, 6, 4], device="npu", dtype=torch.int32)
    pos = torch.arange(32, device="npu")
    slots = table[pos // 4].long() * 4 + pos % 4
    torch.manual_seed(20261026)
    key = torch.randn((32, 2, 256), device="npu", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    previous = 0
    for length in (8, 12, 16, 17, 24, 32):
        cache.write_sequence("a", slots[previous:length], key[previous:length], value[previous:length], length)
        uncached.recent = cache.recent
        actual = cache.read(table, length, key.dtype, "a")
        expected = uncached.read(table, length, key.dtype, "a")
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        if cap:
            filled, rows, _, _ = cache.mirror["a"]
            assert filled <= rows <= cap
            if length >= 16:
                assert filled == cap // 4 * 4
        else:
            assert not cache.mirror
        previous = length
    cache.release(["a"])
    assert not cache.mirror


def test_fixed_table_write_avoids_rate_reduction():
    cache = DQuantNPUCache(4, 512, 353, 32, alpha=1.0)
    values = torch.randn((4, 512), device="npu", dtype=torch.bfloat16)
    with patch.object(torch.Tensor, "min", side_effect=AssertionError("unexpected table reduction")):
        cache.write(torch.arange(4, device="npu"), values)


def test_overflow_is_rejected_before_record_commit():
    cache = DQuantNPUCache(4, 512, 353, 32, alpha=1.0)
    before = cache.storage.clone()
    values = torch.randn((4, 512), device="npu", dtype=torch.bfloat16)
    fake_payload = torch.zeros((4, 353), device="npu", dtype=torch.uint8)
    fake_status = torch.tensor([0, 1, 0, 0], device="npu", dtype=torch.int32)
    with (
        patch(
            "vllm_ascend.quantization.methods.kv_cache.dquant_npu_cache.rans_encode_npu",
            return_value=(fake_payload, fake_status),
        ),
        pytest.raises(RuntimeError, match="refusing to commit"),
    ):
        cache.write(torch.arange(4, device="npu"), values)
    assert torch.equal(cache.storage, before)
    assert not cache.initialized.any().item()


def test_state_mismatch_cannot_silently_replace_attention():
    cache, _ = make_cache()
    q = torch.randn((1, 4, 256), device="npu", dtype=torch.bfloat16)
    kv = torch.randn((1, 2, 256), device="npu", dtype=torch.bfloat16)
    output = torch.empty_like(q)
    with pytest.raises(ValueError, match="refusing to substitute"):
        cache.attention(
            q,
            torch.tensor([[0]], device="npu"),
            [1],
            [2],
            256**-0.5,
            output,
            None,
            request_ids=["missing"],
            updates=(torch.tensor([1], device="npu"), kv, kv),
        )


@pytest.mark.parametrize("budget", [2050.0, 2792.0])
def test_fast_and_constrained_paths_respect_rate_budget(budget):
    cache = DQuantNPUCache(4, 512, 353, 32, alpha=1.0)
    values = torch.randn((4, 512), device="npu", dtype=torch.bfloat16)
    symbols, scales, offsets = dquant_quantize_npu(
        values, cache.rates, budget, minimum_symbol_rate=cache.minimum_symbol_rate
    )
    assert torch.all(cache.rates[symbols.long()].sum(1) <= budget).item()
    reconstructed = offsets[:, None].float() + scales[:, None].float() * symbols.float()
    assert torch.isfinite(reconstructed).all().item()
