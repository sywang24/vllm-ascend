# SPDX-License-Identifier: Apache-2.0
"""V2 NOVA operations composed from the existing codec and NPU kernels."""

from dataclasses import dataclass

import torch
import torch_npu

from vllm_ascend.ops.triton.nova_bounded_fia_prefill import bounded_fia_prefill
from vllm_ascend.ops.triton.nova_bounded_prefill import exact_window_reader, planar_cold_reader
from vllm_ascend.ops.triton.nova_compressed_state import write_planar_packets


@dataclass(frozen=True)
class NovaAttentionPlan:
    max_tokens: int
    split_tokens: int
    workspace_bytes: int


def nova_quant_cache(key, value, codec, views, slots, *, chunk_size):
    """Encode and scatter into allocator-owned planar cache pages."""
    encoded_key, encoded_value, scales = codec.encode(key, value, chunk_size=chunk_size)
    write_planar_packets(
        views,
        slots.long().contiguous(),
        encoded_key.contiguous(),
        encoded_value.contiguous(),
        scales.contiguous(),
    )


def nova_quant_query_lut(query, codec, direct_decode):
    """Create the FP32 query statistics and FP16 lookup table for decode."""
    query = query.float()
    transformed = torch.einsum("bhgd,hde->bhge", query, codec.key_inverse.transpose(-1, -2)).contiguous()
    mean = torch.einsum("bhgd,hd->bhg", query, codec.key_mean).contiguous()
    lut = direct_decode.prepare_lut(transformed, codec._books_float)
    return transformed, mean, lut


def nova_quant_flash_attention_metadata(direct_decode, block_table, views, max_model_len):
    """Build a host-side launch plan from fixed geometry, never device values."""
    key, _, scales = views
    max_tokens = min(max_model_len, block_table.shape[1] * key.shape[-1])
    if max(key.stride(0), scales.stride(0), block_table.stride(0)) >= 2**32:
        raise ValueError("NOVA DirectDecode ABI v1 requires uint32 page and table strides")
    # Old libraries retain the ABI but multiply page offsets in uint32.
    if direct_decode.page_offset_bits < 64 and (
        key.shape[0] * key.stride(0) > 2**32 or scales.shape[0] * scales.stride(0) > 2**32
    ):
        raise ValueError("NOVA cache requires a kernel rebuilt with 64-bit page offsets")
    return NovaAttentionPlan(max_tokens, direct_decode.split_tokens, direct_decode.workspace_bytes())


def nova_quant_flash_attention(
    query,
    codec,
    views,
    block_table,
    window,
    scale,
    *,
    transformed=None,
    mean=None,
    lut=None,
    request_slots=None,
    lengths=None,
    direct_decode=None,
    plan=None,
    key=None,
    value=None,
    old_length=None,
    request_slot=None,
    causal_mask=None,
    query_tile=None,
    kv_tile=None,
):
    """Attend to compressed history and exact tokens without a full KV mirror.

    Decode receives a LUT and a fixed launch plan. Prefill receives original
    chunk K/V and host metadata; it retains the bounded FIA implementation.
    Neither path updates the exact window while reading it.
    """
    if old_length is not None:
        if key is None or value is None or request_slot is None:
            raise ValueError("NOVA prefill requires current K/V and window ownership")
        count, heads, groups, dim = query.shape
        if old_length == 0:
            result, _ = torch_npu.npu_fused_infer_attention_score(
                query.permute(1, 2, 0, 3).reshape(1, heads * groups, count, dim).contiguous(),
                key.transpose(0, 1).unsqueeze(0).contiguous(),
                value.transpose(0, 1).unsqueeze(0).contiguous(),
                atten_mask=causal_mask if count > 1 else None,
                input_layout="BNSD",
                actual_seq_lengths=[count],
                actual_seq_lengths_kv=[count],
                num_heads=heads * groups,
                num_key_value_heads=heads,
                scale=scale,
                sparse_mode=3 if count > 1 else 0,
                pre_tokens=2147483647,
                next_tokens=0,
                inner_precise=1,
            )
            return result.reshape(heads, groups, count, dim).permute(2, 0, 1, 3), None
        cold = planar_cold_reader(*views, block_table, codec)
        exact = exact_window_reader(window.key, window.value, request_slot, window.sink, window.ring, old_length)
        return bounded_fia_prefill(
            query,
            key,
            value,
            old_length,
            cold,
            exact,
            window.sink,
            window.ring - 1,
            query_tile=query_tile,
            kv_tile=kv_tile,
            scale=scale,
        )
    if any(x is None for x in (transformed, mean, lut, request_slots, lengths, direct_decode, plan)):
        raise ValueError("NOVA decode requires a LUT, device metadata and a launch plan")
    starts = lengths.clamp(0, window.sink)
    ends = torch.maximum(starts, lengths - window.ring).contiguous()
    cold, cold_lse = direct_decode(
        transformed,
        mean,
        *views,
        codec.codebooks,
        block_table,
        starts,
        ends,
        plan.max_tokens,
        scale,
        lut=lut,
    )
    cold = torch.einsum("bhgd,hde->bhge", cold, codec.value_transform.transpose(-1, -2))
    result, lse, _ = window.attend_and_merge(query, request_slots, lengths, scale, cold, cold_lse)
    return result, lse


def nova_exact_window_update(
    window, key, value, *, request_slots=None, positions=None, request_slot=None, old_length=None
):
    """Write decode rows or retain the non-overlapping rows of a prefill chunk."""
    if request_slot is not None:
        if old_length is None or request_slots is not None or positions is not None:
            raise ValueError("invalid NOVA prefill window metadata")
        window.seed_chunk(key, value, request_slot, old_length)
    else:
        if request_slots is None or positions is None or old_length is not None:
            raise ValueError("invalid NOVA decode window metadata")
        window.write_rows(key, value, request_slots, positions)
