# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace

import pytest
import torch

from vllm_ascend.attention.dquant_attention import DQuantAttentionCache
from vllm_ascend.core.kv_cache_interface import DQuantAttentionSpec
from vllm_ascend.ops.triton.dquant import rans_decode_npu


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tokens", [256, 4096])
def test_paged_symbol_attention_matches_rans(dtype, tokens, monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_DQUANT_FUSED_ATTENTION", "1")
    monkeypatch.setenv("VLLM_ASCEND_DQUANT_MIRROR_TOKENS", "0")
    monkeypatch.setenv("VLLM_ASCEND_DQUANT_HOST_MIRROR_TOKENS", "0")
    spec = DQuantAttentionSpec(
        block_size=128,
        num_kv_heads=2,
        head_size=256,
        dtype=dtype,
        dquant_recent_window=128,
        dquant_key_window=64,
        dquant_key_container_bytes=155,
        dquant_value_container_bytes=109,
        dquant_symbol_cache=True,
    )
    pages = tokens // 128 + 1
    storage = [
        torch.zeros((pages, 128, width), dtype=torch.uint8, device="npu")
        for width in (spec.key_record_bytes, spec.value_record_bytes)
    ]
    cache = DQuantAttentionCache(storage, spec)
    table = torch.arange(pages, device="npu", dtype=torch.int32).flip(0).contiguous()
    positions = torch.arange(tokens, device="npu")
    slots = table[positions // 128].long() * 128 + positions % 128
    torch.manual_seed(59)
    key = torch.randn((tokens, 2, 256), device="npu", dtype=dtype)
    value = torch.randn_like(key)
    prefix = tokens - 2
    cache.write_sequence("a", slots[:prefix], key[:prefix], value[:prefix], prefix)
    cache.set_decode_chunk_size(0)
    q = torch.randn((1, 4, 256), device="npu", dtype=dtype)
    current = key[prefix : prefix + 1], value[prefix : prefix + 1]
    dense_key, dense_value = cache.read(table, prefix, dtype, "a")
    from vllm_ascend.ops.triton.dquant_fused import raw_dquant_attention_npu

    expected, _ = raw_dquant_attention_npu(
        q, torch.cat((dense_key, current[0])), torch.cat((dense_value, current[1])), 256**-0.5
    )
    actual = cache._fused_decode_attention(q, table, prefix, 256**-0.5, "a", current)
    torch.testing.assert_close(actual.float().reshape_as(expected), expected, atol=0.025, rtol=0.025)
    # The cache stores exactly the rANS symbols, including slot reuse.
    state = cache.recent["a"]
    used = slots[: state.compressed]
    for part in (cache.key, cache.value):
        decoded = rans_decode_npu(part.payloads[used].contiguous(), spec.stream_length, *part.decode_luts)
        packed = part.symbols[used]
        unpacked = torch.stack((packed & 15, packed >> 4), dim=-1).reshape_as(decoded)
        torch.testing.assert_close(unpacked.to(decoded.dtype), decoded, rtol=0, atol=0)
    old = cache.key.symbols[used].clone()
    cache.write_sequence("a", slots[:prefix], -key[:prefix], value[:prefix], prefix)
    assert not torch.equal(old, cache.key.symbols[used])
    # All page bytes, including symbol sidecars and mean stripes, move together.
    src, dst = int(table[0].item()), int(table[-1].item())
    for part in (cache.key, cache.value):
        expected_page = part.storage[src * 128 : (src + 1) * 128].clone()
        part.copy_blocks(torch.tensor([[src, dst]], device="npu"), 128)
        torch.testing.assert_close(part.storage[dst * 128 : (dst + 1) * 128], expected_page)
    assert spec.real_page_size_bytes == 128 * (spec.key_record_bytes + spec.value_record_bytes)
    assert spec.key_record_bytes == 432 and spec.value_record_bytes == 370
    assert not cache.mirror and not cache.host_mirror


def test_symbol_page_budget_and_invalid_layout():
    base = DQuantAttentionSpec(
        block_size=128,
        num_kv_heads=2,
        head_size=256,
        dtype=torch.bfloat16,
        dquant_key_window=64,
        dquant_key_container_bytes=155,
        dquant_value_container_bytes=109,
    )
    symbols = replace(base, dquant_symbol_cache=True)
    assert symbols.real_page_size_bytes - base.real_page_size_bytes == 128 * 512
    with pytest.raises(ValueError, match="Symbol cache"):
        replace(symbols, dquant_key_levels=32)
    with pytest.raises(ValueError, match="Symbol cache"):
        replace(symbols, dquant_key_window=8)
