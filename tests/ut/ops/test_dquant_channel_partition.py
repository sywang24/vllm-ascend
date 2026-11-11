# SPDX-License-Identifier: Apache-2.0
"""Check channel ownership across padded GQA groups and inactive batch tiles."""

import pytest
import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.attention.dquant_attention import DQuantAttentionCache
from vllm_ascend.core.kv_cache_interface import DQuantAttentionSpec
from vllm_ascend.ops.triton.dquant import hadamard_rotate_npu
from vllm_ascend.ops.triton.dquant_batch_persistent import _paged_dquant_batch_persistent_window_attention_kernel
from vllm_ascend.ops.triton.dquant_batch_window import _paged_dquant_batch_window_attention_kernel
from vllm_ascend.ops.triton.dquant_fused import paged_dquant_attention_npu

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="Ascend NPU is unavailable")


@pytest.mark.parametrize(
    "dtype,heads,chunk,width,partitions",
    [
        (torch.bfloat16, 12, 64, 256, 0),
        (torch.float16, 12, 64, 256, 0),
        (torch.bfloat16, 6, 32, 256, 0),
        (torch.float16, 24, 16, 256, 0),
        (torch.bfloat16, 4, 16, 16, 0),
        (torch.bfloat16, 12, 64, 256, 1),
        (torch.bfloat16, 12, 64, 256, 128),
    ],
)
def test_channel_groups_match_single_request_and_preserve_inactive_tiles(
    dtype, heads, chunk, width, partitions, monkeypatch
):
    monkeypatch.setenv("VLLM_ASCEND_DQUANT_FUSED_ATTENTION", "1")
    torch.manual_seed(1103 + heads + width)
    spec = DQuantAttentionSpec(
        block_size=128,
        num_kv_heads=2,
        head_size=width,
        dtype=dtype,
        dquant_recent_window=128,
        dquant_key_window=64,
    )
    batch = 3
    pages = 4
    storage = [
        torch.zeros((batch * pages, 128, size), device="npu", dtype=torch.uint8)
        for size in (spec.key_record_bytes, spec.value_record_bytes)
    ]
    cache = DQuantAttentionCache(storage, spec)
    tables = torch.stack([torch.randperm(pages, device="npu").to(torch.int32) + i * pages for i in range(batch)])
    for i, length in enumerate((192, 256)):
        positions = torch.arange(length, device="npu")
        slots = tables[i, positions // 128].long() * 128 + positions % 128
        key = torch.randn((length, 2, width), dtype=dtype, device="npu")
        values = torch.randn_like(key)
        values[:, 0] += 2
        values[:, 1] -= 3
        cache.write_sequence(str(i), slots, key, values, length)
    lengths = [cache.recent[str(i)].compressed for i in range(2)] + [0]
    maximum = max(lengths)
    chunks = triton.cdiv(maximum, chunk)
    compressed = torch.tensor(lengths, dtype=torch.int32, device="npu")
    query = torch.randn((batch, heads, width), dtype=dtype, device="npu")
    output = torch.full((batch, chunks, heads, width), 173.0, dtype=torch.float32, device="npu")
    lse = torch.full((batch, chunks, heads), 173.0, dtype=torch.float32, device="npu")
    kernel = (
        _paged_dquant_batch_persistent_window_attention_kernel
        if partitions
        else _paged_dquant_batch_window_attention_kernel
    )
    launch_options = {"PARTITIONS": partitions, "has_auto_blockify_blacklist_op": True} if partitions else {}
    kernel[(batch * (partitions or chunks),)](
        hadamard_rotate_npu(query.float()),
        query,
        cache.key.storage,
        tables,
        cache.fused_key_lut,
        cache.key.scales,
        cache.key.offsets,
        cache.value.storage,
        tables,
        cache.fused_value_lut,
        cache.value.scales,
        cache.value.offsets,
        output,
        lse,
        compressed,
        chunks,
        tables.stride(0),
        width**-0.5,
        NUM_Q_HEADS=heads,
        NUM_KV_HEADS=2,
        HEAD_SIZE=width,
        BLOCK_SIZE=128,
        BLOCK_H=triton.next_power_of_2(heads),
        BLOCK_T=chunk,
        CHUNK_TOKENS=chunk,
        KEY_ROW_BYTES=cache.key.row_stride_bytes,
        KEY_METADATA_STRIDE=cache.key.scales.stride(0),
        KEY_CONTAINER_BYTES=cache.key.container_bytes,
        VALUE_ROW_BYTES=cache.value.row_stride_bytes,
        VALUE_METADATA_STRIDE=cache.value.scales.stride(0),
        VALUE_CONTAINER_BYTES=cache.value.container_bytes,
        TABLE_SIZE=256,
        SCALE_BITS=8,
        LOWER_BOUND=1 << 23,
        KEY_WINDOW=64,
        MEAN_OFFSET=cache.key.record_bytes,
        STRIPE_BYTES=spec.mean_stripe_bytes,
        IO_DTYPE=tl.bfloat16 if dtype == torch.bfloat16 else tl.float16,
        num_warps=4,
        **launch_options,
    )
    for i, length in enumerate(lengths):
        active = triton.cdiv(length, chunk)
        if active:
            expected, expected_lse = paged_dquant_attention_npu(
                query[i : i + 1],
                tables[i],
                length,
                width**-0.5,
                cache.key.storage,
                cache.key.scales,
                cache.key.offsets,
                cache.fused_key_lut,
                cache.value.storage,
                cache.value.scales,
                cache.value.offsets,
                cache.fused_value_lut,
                block_size=128,
                head_size=width,
                num_kv_heads=2,
                key_container_bytes=cache.key.container_bytes,
                value_container_bytes=cache.value.container_bytes,
                key_window=64,
                mean_offset=cache.key.record_bytes,
                stripe_bytes=spec.mean_stripe_bytes,
                chunk_tokens=chunk,
                tile_tokens=chunk,
            )
            torch.testing.assert_close(output[i, :active], expected, atol=3e-4, rtol=2e-3)
            torch.testing.assert_close(lse[i, :active], expected_lse, atol=3e-4, rtol=2e-3)
            assert torch.isfinite(output[i, :active]).all()
        assert torch.all(output[i, active:] == 173)
        assert torch.all(lse[i, active:] == 173)
