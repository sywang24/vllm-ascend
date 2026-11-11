# SPDX-License-Identifier: Apache-2.0
"""Fused D-Quant decode attention over fixed-size rANS KV records.

The kernel decodes each token stream once, keeps the decoded levels on chip,
folds the per-token affine transform into QK/PV, and emits only attention
outputs. Keys and values are never materialized as dense KV tensors in HBM.
Hadamard inverses are folded into the query and the merged output.
"""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.dquant import hadamard_rotate_npu
from vllm_ascend.ops.triton.dquant_batch_persistent import _paged_dquant_batch_persistent_window_attention_kernel
from vllm_ascend.ops.triton.dquant_batch_window import _paged_dquant_batch_window_attention_kernel
from vllm_ascend.ops.triton.dquant_fused_window import _paged_dquant_window_attention_kernel


def make_packed_decode_lut(frequencies, device):
    """Pack symbol (8), frequency (9), and cumulative (8) into one word."""
    if not frequencies or len(frequencies) > 256 or sum(frequencies) != 256:
        raise ValueError("Fused rANS decoding requires a 256-entry probability table")
    entries, cumulative = [], 0
    for symbol, frequency in enumerate(frequencies):
        if frequency <= 0:
            raise ValueError("Fused rANS frequencies must be positive")
        entry = symbol | (frequency << 8) | (cumulative << 17)
        entries.extend([entry] * frequency)
        cumulative += frequency
    return torch.tensor(entries, dtype=torch.int32, device=device)


@triton.jit(do_not_specialize=["TOKENS"])
def _raw_dquant_attention_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    output_ptr,
    lse_ptr,
    SCALE,
    TOKENS,
    HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    head = tl.program_id(0)
    kv_head = head // (HEADS // KV_HEADS)
    dims = tl.arange(0, WIDTH)
    token_offsets = tl.arange(0, BLOCK_T)
    query = tl.load(query_ptr + head * WIDTH + dims).to(tl.float32)
    maximum = tl.full((), float("-inf"), tl.float32)
    denominator = tl.full((), 0.0, tl.float32)
    numerator = tl.zeros((WIDTH,), tl.float32)
    for start in range(tl.cdiv(TOKENS, BLOCK_T)):
        tokens = start * BLOCK_T + token_offsets
        offsets = (tokens[:, None] * KV_HEADS + kv_head) * WIDTH + dims[None, :]
        keys = tl.load(key_ptr + offsets, mask=tokens[:, None] < TOKENS, other=0.0).to(tl.float32)
        scores = tl.sum(keys * query[None, :], axis=1) * SCALE
        scores = tl.where(tokens < TOKENS, scores, float("-inf"))
        next_maximum = tl.maximum(maximum, tl.max(scores, axis=0))
        alpha = tl.exp(maximum - next_maximum)
        probabilities = tl.exp(scores - next_maximum)
        values = tl.load(value_ptr + offsets, mask=tokens[:, None] < TOKENS, other=0.0).to(tl.float32)
        numerator = numerator * alpha + tl.sum(probabilities[:, None] * values, axis=0)
        denominator = denominator * alpha + tl.sum(probabilities, axis=0)
        maximum = next_maximum
    tl.store(output_ptr + head * WIDTH + dims, numerator / denominator)
    tl.store(lse_ptr + head, maximum + tl.log(denominator))


def raw_dquant_attention_npu(query, key, value, scale):
    """Compute the bounded uncompressed tail without a large native workspace."""
    if query.ndim != 3 or query.shape[0] != 1 or key.ndim != 3 or key.shape != value.shape:
        raise ValueError("Raw D-Quant attention requires one query and matching token/head/width K/V")
    tokens, kv_heads, width = key.shape
    heads = query.shape[1]
    if not tokens or not kv_heads or heads % kv_heads or query.shape[2] != width:
        raise ValueError("Raw D-Quant attention head layout is invalid")
    if width <= 0 or width & (width - 1):
        raise ValueError("Raw D-Quant attention requires a power-of-two head width")
    if any(t.device != query.device or not t.is_contiguous() for t in (query, key, value)):
        raise ValueError("Raw D-Quant attention tensors must be contiguous on one device")
    output = torch.empty((heads, width), dtype=torch.float32, device=query.device)
    lse = torch.empty((heads,), dtype=torch.float32, device=query.device)
    _raw_dquant_attention_kernel[(heads,)](
        query,
        key,
        value,
        output,
        lse,
        scale,
        TOKENS=tokens,
        HEADS=heads,
        KV_HEADS=kv_heads,
        WIDTH=width,
        BLOCK_T=32,
    )
    return output, lse


@triton.jit(do_not_specialize=["CHUNKS"])
def _merge_attention_partials_kernel(
    parts_ptr,
    lse_ptr,
    output_ptr,
    output_lse_ptr,
    CHUNKS,
    HEADS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    head = tl.program_id(0)
    channel_block = tl.program_id(1)
    channels = channel_block * BLOCK_D + tl.arange(0, BLOCK_D)
    chunk_offsets = tl.arange(0, BLOCK_C)
    maximum = tl.full((), float("-inf"), tl.float32)
    denominator = tl.full((), 0.0, tl.float32)
    numerator = tl.zeros((BLOCK_D,), tl.float32)
    for start in range(tl.cdiv(CHUNKS, BLOCK_C)):
        chunks = start * BLOCK_C + chunk_offsets
        lse = tl.load(lse_ptr + chunks * HEADS + head, mask=chunks < CHUNKS, other=float("-inf"))
        next_maximum = tl.maximum(maximum, tl.max(lse, axis=0))
        alpha = tl.exp(maximum - next_maximum)
        weights = tl.exp(lse - next_maximum)
        outputs = tl.load(
            parts_ptr + (chunks[:, None] * HEADS + head) * WIDTH + channels[None, :],
            mask=(chunks[:, None] < CHUNKS) & (channels[None, :] < WIDTH),
            other=0.0,
        )
        numerator = numerator * alpha + tl.sum(weights[:, None] * outputs, axis=0)
        denominator = denominator * alpha + tl.sum(weights, axis=0)
        maximum = next_maximum
    tl.store(output_ptr + head * WIDTH + channels, numerator / denominator, mask=channels < WIDTH)
    if channel_block == 0:
        tl.store(output_lse_ptr + head, maximum + tl.log(denominator))


def merge_dquant_attention_partials_npu(parts, lses):
    """Merge split-KV outputs on device with bounded on-chip scratch."""
    if parts.ndim != 3 or lses.shape != parts.shape[:2] or not parts.shape[0]:
        raise ValueError("Attention partials require nonempty [chunks, heads, width] outputs and matching LSE")
    if parts.device != lses.device or not parts.is_contiguous() or not lses.is_contiguous():
        raise ValueError("Attention partials and LSE must be contiguous on one device")
    if parts.shape[0] == 1:
        return parts[0].float(), lses[0].float()
    chunks, heads, width = parts.shape
    output = torch.empty((heads, width), dtype=torch.float32, device=parts.device)
    output_lse = torch.empty((heads,), dtype=torch.float32, device=parts.device)
    _merge_attention_partials_kernel[(heads, triton.cdiv(width, 64))](
        parts,
        lses,
        output,
        output_lse,
        CHUNKS=chunks,
        HEADS=heads,
        WIDTH=width,
        BLOCK_C=64,
        BLOCK_D=64,
    )
    return output, output_lse


@triton.jit
def _rans_step(
    state,
    cursor,
    payload,
    packed_lut,
    CONTAINER_BYTES: tl.constexpr,
    TABLE_SIZE: tl.constexpr,
    SCALE_BITS: tl.constexpr,
    LOWER_BOUND: tl.constexpr,
):
    slot = (state & (TABLE_SIZE - 1)).to(tl.int32)
    entry = tl.load(packed_lut + slot)
    symbol = entry & 255
    frequency = (entry >> 8) & 511
    cumulative = (entry >> 17) & 255
    state = frequency * (state >> SCALE_BITS) + slot - cumulative
    # With total frequency 256, one byte always restores the lower bound.
    needs_byte = (state < LOWER_BOUND) & (cursor < CONTAINER_BYTES)
    byte = tl.load(payload + cursor, mask=needs_byte, other=0).to(tl.int32)
    state = tl.where(needs_byte, (state << 8) | byte, state)
    cursor += needs_byte.to(tl.int32)
    return state, cursor, symbol


@triton.jit
def _paged_dquant_fused_attention_kernel(
    query_ptr,
    original_query_ptr,
    key_storage_ptr,
    key_table_ptr,
    key_lut_ptr,
    key_scales_ptr,
    key_offsets_ptr,
    value_storage_ptr,
    value_table_ptr,
    value_lut_ptr,
    value_scales_ptr,
    value_offsets_ptr,
    output_ptr,
    lse_ptr,
    COMPRESSED,
    SCALE,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_T: tl.constexpr,
    CHUNK_TOKENS: tl.constexpr,
    KEY_ROW_BYTES: tl.constexpr,
    KEY_METADATA_STRIDE: tl.constexpr,
    KEY_CONTAINER_BYTES: tl.constexpr,
    VALUE_ROW_BYTES: tl.constexpr,
    VALUE_METADATA_STRIDE: tl.constexpr,
    VALUE_CONTAINER_BYTES: tl.constexpr,
    TABLE_SIZE: tl.constexpr,
    SCALE_BITS: tl.constexpr,
    LOWER_BOUND: tl.constexpr,
    KEY_WINDOW: tl.constexpr,
    MEAN_OFFSET: tl.constexpr,
    STRIPE_BYTES: tl.constexpr,
    IO_DTYPE: tl.constexpr,
):
    chunk = tl.program_id(0)
    chunk_start = chunk * CHUNK_TOKENS
    chunk_end = tl.minimum(chunk_start + CHUNK_TOKENS, COMPRESSED)
    heads_per_kv: tl.constexpr = NUM_Q_HEADS // NUM_KV_HEADS
    head_ids = tl.arange(0, BLOCK_H)
    dims = tl.arange(0, HEAD_SIZE)
    token_offsets = tl.arange(0, BLOCK_T)
    m_i = tl.full((BLOCK_H,), float("-inf"), dtype=tl.float32)
    lse_i = tl.full((BLOCK_H,), float("-inf"), dtype=tl.float32)
    value_acc = tl.zeros((BLOCK_H, HEAD_SIZE), dtype=tl.float32)
    rotated_query = tl.load(
        query_ptr + head_ids[:, None] * HEAD_SIZE + dims[None, :],
        mask=head_ids[:, None] < NUM_Q_HEADS,
        other=0.0,
    )
    query_sum = tl.sum(rotated_query, axis=1)
    for tile in range(tl.cdiv(chunk_end - chunk_start, BLOCK_T)):
        tokens = chunk_start + tile * BLOCK_T + token_offsets
        valid = tokens < chunk_end
        safe_tokens = tl.where(valid, tokens, 0)
        key_block = tl.load(key_table_ptr + safe_tokens // BLOCK_SIZE, mask=valid, other=0).to(tl.int32)
        key_slot = key_block * BLOCK_SIZE + safe_tokens % BLOCK_SIZE
        key_payload = key_storage_ptr + key_slot * KEY_ROW_BYTES
        key_state = (
            tl.load(key_payload, mask=valid, other=0).to(tl.int32)
            | (tl.load(key_payload + 1, mask=valid, other=0).to(tl.int32) << 8)
            | (tl.load(key_payload + 2, mask=valid, other=0).to(tl.int32) << 16)
            | (tl.load(key_payload + 3, mask=valid, other=0).to(tl.int32) << 24)
        )
        key_cursor = tl.full((BLOCK_T,), 4, dtype=tl.int32)
        key_scale = tl.load(key_scales_ptr + key_slot * KEY_METADATA_STRIDE, mask=valid, other=0).to(tl.float32)
        key_offset = tl.load(key_offsets_ptr + key_slot * KEY_METADATA_STRIDE, mask=valid, other=0).to(tl.float32)
        qk = tl.zeros((BLOCK_H, BLOCK_T), dtype=tl.float32)
        qk_mean = tl.zeros((BLOCK_H, BLOCK_T), dtype=tl.float32)
        for dimension in range(HEAD_SIZE * NUM_KV_HEADS):
            key_state, key_cursor, symbol = _rans_step(
                key_state,
                key_cursor,
                key_payload,
                key_lut_ptr,
                KEY_CONTAINER_BYTES,
                TABLE_SIZE,
                SCALE_BITS,
                LOWER_BOUND,
            )
            query_mask = (head_ids < NUM_Q_HEADS) & (dimension // HEAD_SIZE == head_ids // heads_per_kv)
            query_head = tl.load(
                query_ptr + head_ids * HEAD_SIZE + dimension % HEAD_SIZE,
                mask=query_mask,
                other=0.0,
            ).to(tl.float32)
            qk += query_head[:, None] * symbol.to(tl.float32)[None, :]
            if KEY_WINDOW:
                mean_byte = dimension * 2
                mean_record = (safe_tokens // KEY_WINDOW) * KEY_WINDOW + mean_byte // STRIPE_BYTES
                mean_block = tl.load(key_table_ptr + mean_record // BLOCK_SIZE, mask=valid, other=0).to(tl.int32)
                mean_slot = mean_block * BLOCK_SIZE + mean_record % BLOCK_SIZE
                mean_address = mean_slot * KEY_ROW_BYTES + MEAN_OFFSET + mean_byte % STRIPE_BYTES
                low = tl.load(key_storage_ptr + mean_address, mask=valid, other=0).to(tl.uint32)
                high = tl.load(key_storage_ptr + mean_address + 1, mask=valid, other=0).to(tl.uint32)
                bits = (low | (high << 8)).to(tl.uint16)
                if IO_DTYPE == tl.bfloat16:  # noqa: SIM300
                    mean = bits.to(tl.bfloat16, bitcast=True).to(tl.float32)
                else:
                    mean = bits.to(tl.float16, bitcast=True).to(tl.float32)
                original_query = tl.load(
                    original_query_ptr + head_ids * HEAD_SIZE + dimension % HEAD_SIZE,
                    mask=query_mask,
                    other=0.0,
                ).to(tl.float32)
                qk_mean += original_query[:, None] * mean[None, :]
        qk = qk * key_scale[None, :] + query_sum[:, None] * key_offset[None, :] + qk_mean
        qk = tl.where(valid[None, :], qk * (SCALE * 1.4426950408889634), float("-inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp2(m_i - m_ij)
        probabilities = tl.exp2(qk - m_ij[:, None])
        probabilities = tl.where((valid[None, :]) & (head_ids[:, None] < NUM_Q_HEADS), probabilities, 0.0)
        l_ij = tl.sum(probabilities, axis=1)
        lse_i = m_ij + tl.log2(tl.exp2(lse_i - m_ij) + l_ij)
        m_i = m_ij
        value_acc *= alpha[:, None]
        value_block = tl.load(value_table_ptr + safe_tokens // BLOCK_SIZE, mask=valid, other=0).to(tl.int32)
        value_slot = value_block * BLOCK_SIZE + safe_tokens % BLOCK_SIZE
        value_payload = value_storage_ptr + value_slot * VALUE_ROW_BYTES
        value_state = (
            tl.load(value_payload, mask=valid, other=0).to(tl.int32)
            | (tl.load(value_payload + 1, mask=valid, other=0).to(tl.int32) << 8)
            | (tl.load(value_payload + 2, mask=valid, other=0).to(tl.int32) << 16)
            | (tl.load(value_payload + 3, mask=valid, other=0).to(tl.int32) << 24)
        )
        value_cursor = tl.full((BLOCK_T,), 4, dtype=tl.int32)
        value_scale = tl.load(value_scales_ptr + value_slot * VALUE_METADATA_STRIDE, mask=valid, other=0).to(tl.float32)
        value_offset = tl.load(value_offsets_ptr + value_slot * VALUE_METADATA_STRIDE, mask=valid, other=0).to(
            tl.float32
        )
        scaled = probabilities * value_scale[None, :]
        offset_contribution = tl.sum(probabilities * value_offset[None, :], axis=1)
        # Vector contraction avoids the compiler's per-program HBM scratch
        # for Cube/Vector exchange. Each stream is decoded once for all heads.
        for dimension in range(HEAD_SIZE * NUM_KV_HEADS):
            value_state, value_cursor, symbol = _rans_step(
                value_state,
                value_cursor,
                value_payload,
                value_lut_ptr,
                VALUE_CONTAINER_BYTES,
                TABLE_SIZE,
                SCALE_BITS,
                LOWER_BOUND,
            )
            contribution = tl.sum(scaled * symbol.to(tl.float32)[None, :], axis=1)
            contribution += offset_contribution
            output_mask = (head_ids[:, None] // heads_per_kv == dimension // HEAD_SIZE) & (
                dims[None, :] == dimension % HEAD_SIZE
            )
            value_acc += tl.where(output_mask, contribution[:, None], 0.0)

    denominator = tl.exp2(m_i - lse_i)
    value_acc *= denominator[:, None]
    output_offset = chunk * NUM_Q_HEADS * HEAD_SIZE
    tl.store(
        output_ptr + output_offset + head_ids[:, None] * HEAD_SIZE + dims[None, :],
        value_acc,
        mask=head_ids[:, None] < NUM_Q_HEADS,
    )
    tl.store(
        lse_ptr + chunk * NUM_Q_HEADS + head_ids,
        lse_i * 0.6931471805599453,
        mask=head_ids < NUM_Q_HEADS,
    )


@triton.jit(do_not_specialize=["MAX_CHUNKS", "BLOCK_TABLE_STRIDE"])
def _paged_dquant_batch_attention_kernel(
    query_ptr,
    original_query_ptr,
    key_storage_ptr,
    key_table_ptr,
    key_lut_ptr,
    key_scales_ptr,
    key_offsets_ptr,
    value_storage_ptr,
    value_table_ptr,
    value_lut_ptr,
    value_scales_ptr,
    value_offsets_ptr,
    output_ptr,
    lse_ptr,
    compressed_ptr,
    MAX_CHUNKS,
    BLOCK_TABLE_STRIDE,
    SCALE,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_T: tl.constexpr,
    CHUNK_TOKENS: tl.constexpr,
    KEY_ROW_BYTES: tl.constexpr,
    KEY_METADATA_STRIDE: tl.constexpr,
    KEY_CONTAINER_BYTES: tl.constexpr,
    VALUE_ROW_BYTES: tl.constexpr,
    VALUE_METADATA_STRIDE: tl.constexpr,
    VALUE_CONTAINER_BYTES: tl.constexpr,
    TABLE_SIZE: tl.constexpr,
    SCALE_BITS: tl.constexpr,
    LOWER_BOUND: tl.constexpr,
    KEY_WINDOW: tl.constexpr,
    MEAN_OFFSET: tl.constexpr,
    STRIPE_BYTES: tl.constexpr,
    IO_DTYPE: tl.constexpr,
):
    """Decode one chunk for every request in a padded batch launch."""
    program = tl.program_id(0)
    batch = program // MAX_CHUNKS
    chunk = program - batch * MAX_CHUNKS
    compressed = tl.load(compressed_ptr + batch).to(tl.int32)
    active = chunk * CHUNK_TOKENS < compressed
    chunk_start = chunk * CHUNK_TOKENS
    chunk_end = tl.minimum(chunk_start + CHUNK_TOKENS, compressed)
    # Inactive programs execute one harmless dummy tile and never store output.
    work_end = tl.where(active, chunk_end, chunk_start + 1)
    heads_per_kv: tl.constexpr = NUM_Q_HEADS // NUM_KV_HEADS
    head_ids = tl.arange(0, BLOCK_H)
    head_mask = head_ids < NUM_Q_HEADS
    dims = tl.arange(0, HEAD_SIZE)
    token_offsets = tl.arange(0, BLOCK_T)
    m_i = tl.full((BLOCK_H,), float("-inf"), dtype=tl.float32)
    lse_i = tl.full((BLOCK_H,), float("-inf"), dtype=tl.float32)
    value_acc = tl.zeros((BLOCK_H, HEAD_SIZE), dtype=tl.float32)
    query_ptr += batch * NUM_Q_HEADS * HEAD_SIZE
    original_query_ptr += batch * NUM_Q_HEADS * HEAD_SIZE
    key_table_ptr += batch * BLOCK_TABLE_STRIDE
    value_table_ptr += batch * BLOCK_TABLE_STRIDE
    output_ptr += batch * MAX_CHUNKS * NUM_Q_HEADS * HEAD_SIZE
    lse_ptr += batch * MAX_CHUNKS * NUM_Q_HEADS
    rotated_query = tl.load(
        query_ptr + head_ids[:, None] * HEAD_SIZE + dims[None, :],
        mask=head_mask[:, None],
        other=0.0,
    )
    query_sum = tl.sum(rotated_query, axis=1)
    for tile in range(tl.cdiv(work_end - chunk_start, BLOCK_T)):
        tokens = chunk_start + tile * BLOCK_T + token_offsets
        valid = tokens < work_end
        safe_tokens = tl.where(valid, tokens, 0)
        key_block = tl.load(key_table_ptr + safe_tokens // BLOCK_SIZE, mask=valid, other=0).to(tl.int32)
        key_slot = key_block * BLOCK_SIZE + safe_tokens % BLOCK_SIZE
        key_payload = key_storage_ptr + key_slot * KEY_ROW_BYTES
        key_state = (
            tl.load(key_payload, mask=valid, other=0).to(tl.int32)
            | (tl.load(key_payload + 1, mask=valid, other=0).to(tl.int32) << 8)
            | (tl.load(key_payload + 2, mask=valid, other=0).to(tl.int32) << 16)
            | (tl.load(key_payload + 3, mask=valid, other=0).to(tl.int32) << 24)
        )
        key_cursor = tl.full((BLOCK_T,), 4, dtype=tl.int32)
        key_scale = tl.load(key_scales_ptr + key_slot * KEY_METADATA_STRIDE, mask=valid, other=0).to(tl.float32)
        key_offset = tl.load(key_offsets_ptr + key_slot * KEY_METADATA_STRIDE, mask=valid, other=0).to(tl.float32)
        qk = tl.zeros((BLOCK_H, BLOCK_T), dtype=tl.float32)
        qk_mean = tl.zeros((BLOCK_H, BLOCK_T), dtype=tl.float32)
        for dimension in range(HEAD_SIZE * NUM_KV_HEADS):
            key_state, key_cursor, symbol = _rans_step(
                key_state,
                key_cursor,
                key_payload,
                key_lut_ptr,
                KEY_CONTAINER_BYTES,
                TABLE_SIZE,
                SCALE_BITS,
                LOWER_BOUND,
            )
            query_mask = head_mask & (dimension // HEAD_SIZE == head_ids // heads_per_kv)
            query_head = tl.load(
                query_ptr + head_ids * HEAD_SIZE + dimension % HEAD_SIZE,
                mask=query_mask,
                other=0.0,
            ).to(tl.float32)
            qk += query_head[:, None] * symbol.to(tl.float32)[None, :]
            if KEY_WINDOW:
                mean_byte = dimension * 2
                mean_record = (safe_tokens // KEY_WINDOW) * KEY_WINDOW + mean_byte // STRIPE_BYTES
                mean_block = tl.load(key_table_ptr + mean_record // BLOCK_SIZE, mask=valid, other=0).to(tl.int32)
                mean_slot = mean_block * BLOCK_SIZE + mean_record % BLOCK_SIZE
                mean_address = mean_slot * KEY_ROW_BYTES + MEAN_OFFSET + mean_byte % STRIPE_BYTES
                low = tl.load(key_storage_ptr + mean_address, mask=valid, other=0).to(tl.uint32)
                high = tl.load(key_storage_ptr + mean_address + 1, mask=valid, other=0).to(tl.uint32)
                bits = (low | (high << 8)).to(tl.uint16)
                if IO_DTYPE == tl.bfloat16:  # noqa: SIM300
                    mean = bits.to(tl.bfloat16, bitcast=True).to(tl.float32)
                else:
                    mean = bits.to(tl.float16, bitcast=True).to(tl.float32)
                original_query = tl.load(
                    original_query_ptr + head_ids * HEAD_SIZE + dimension % HEAD_SIZE,
                    mask=query_mask,
                    other=0.0,
                ).to(tl.float32)
                qk_mean += original_query[:, None] * mean[None, :]
        qk = qk * key_scale[None, :] + query_sum[:, None] * key_offset[None, :] + qk_mean
        qk = tl.where(valid[None, :], qk * (SCALE * 1.4426950408889634), float("-inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp2(m_i - m_ij)
        probabilities = tl.exp2(qk - m_ij[:, None])
        probabilities = tl.where(valid[None, :] & head_mask[:, None], probabilities, 0.0)
        l_ij = tl.sum(probabilities, axis=1)
        lse_i = m_ij + tl.log2(tl.exp2(lse_i - m_ij) + l_ij)
        m_i = m_ij
        value_acc *= alpha[:, None]
        value_block = tl.load(value_table_ptr + safe_tokens // BLOCK_SIZE, mask=valid, other=0).to(tl.int32)
        value_slot = value_block * BLOCK_SIZE + safe_tokens % BLOCK_SIZE
        value_payload = value_storage_ptr + value_slot * VALUE_ROW_BYTES
        value_state = (
            tl.load(value_payload, mask=valid, other=0).to(tl.int32)
            | (tl.load(value_payload + 1, mask=valid, other=0).to(tl.int32) << 8)
            | (tl.load(value_payload + 2, mask=valid, other=0).to(tl.int32) << 16)
            | (tl.load(value_payload + 3, mask=valid, other=0).to(tl.int32) << 24)
        )
        value_cursor = tl.full((BLOCK_T,), 4, dtype=tl.int32)
        value_scale = tl.load(value_scales_ptr + value_slot * VALUE_METADATA_STRIDE, mask=valid, other=0).to(tl.float32)
        value_offset = tl.load(value_offsets_ptr + value_slot * VALUE_METADATA_STRIDE, mask=valid, other=0).to(
            tl.float32
        )
        scaled = probabilities * value_scale[None, :]
        offset_contribution = tl.sum(probabilities * value_offset[None, :], axis=1)
        for dimension in range(HEAD_SIZE * NUM_KV_HEADS):
            value_state, value_cursor, symbol = _rans_step(
                value_state,
                value_cursor,
                value_payload,
                value_lut_ptr,
                VALUE_CONTAINER_BYTES,
                TABLE_SIZE,
                SCALE_BITS,
                LOWER_BOUND,
            )
            contribution = tl.sum(scaled * symbol.to(tl.float32)[None, :], axis=1)
            contribution += offset_contribution
            output_mask = (head_ids[:, None] // heads_per_kv == dimension // HEAD_SIZE) & (
                dims[None, :] == dimension % HEAD_SIZE
            )
            value_acc += tl.where(output_mask, contribution[:, None], 0.0)
    denominator = tl.exp2(m_i - lse_i)
    value_acc *= denominator[:, None]
    output_offset = chunk * NUM_Q_HEADS * HEAD_SIZE
    tl.store(
        output_ptr + output_offset + head_ids[:, None] * HEAD_SIZE + dims[None, :],
        value_acc,
        mask=active & head_mask[:, None],
    )
    tl.store(
        lse_ptr + chunk * NUM_Q_HEADS + head_ids,
        lse_i * 0.6931471805599453,
        mask=active & head_mask,
    )


def paged_dquant_batch_attention_npu(
    query: torch.Tensor,
    block_tables: torch.Tensor,
    compressed: torch.Tensor,
    scale: float,
    key_storage: torch.Tensor,
    key_scales: torch.Tensor,
    key_offsets: torch.Tensor,
    key_decode_lut: torch.Tensor,
    value_storage: torch.Tensor,
    value_scales: torch.Tensor,
    value_offsets: torch.Tensor,
    value_decode_lut: torch.Tensor,
    *,
    block_size: int,
    head_size: int,
    num_kv_heads: int,
    key_container_bytes: int,
    value_container_bytes: int,
    key_window: int,
    mean_offset: int,
    stripe_bytes: int,
    chunk_tokens: int = 16,
    tile_tokens: int = 16,
    compressed_max: int | None = None,
    partials: tuple[torch.Tensor, torch.Tensor] | None = None,
    native_backend=None,
):
    """Decode compressed prefixes for a batch with one kernel launch."""
    if query.ndim != 3 or query.shape[0] < 1:
        raise ValueError("Batched fused D-Quant attention expects at least one query token")
    batch = query.shape[0]
    if num_kv_heads != 2:
        raise ValueError("Batched fused D-Quant attention currently requires two local KV heads")
    if compressed.shape != (batch,) or compressed.dtype not in (torch.int32, torch.int64):
        raise ValueError("Batched compressed lengths must be one integer per request")
    maximum = int(compressed_max if compressed_max is not None else compressed.max().item())
    if maximum <= 0 or maximum % (key_window or 1):
        raise ValueError("Batched compressed lengths must be positive and window aligned")
    if tile_tokens < 16 or chunk_tokens <= 0 or chunk_tokens % tile_tokens:
        raise ValueError("Fused chunk must be a multiple of at least 16 token tiles")
    if tile_tokens & (tile_tokens - 1):
        raise ValueError("Fused token tile size must be a power of two")
    query_heads, width = query.shape[1], query.shape[2]
    if width != head_size or query_heads % num_kv_heads:
        raise ValueError("Batched query layout does not match the fused D-Quant KV heads")
    if query.dtype not in (torch.float16, torch.bfloat16) or not query.is_contiguous():
        raise ValueError("Batched fused attention requires contiguous FP16 or BF16 queries")
    if block_tables.ndim != 2 or block_tables.shape[0] < batch or not block_tables.is_contiguous():
        raise ValueError("Batched block tables must be contiguous with one row per request")
    if block_tables.shape[1] * block_size < maximum:
        raise ValueError("Batched block tables do not cover the compressed prefixes")
    tensors = (
        query,
        compressed,
        block_tables,
        key_storage,
        key_scales,
        key_offsets,
        key_decode_lut,
        value_storage,
        value_scales,
        value_offsets,
        value_decode_lut,
    )
    if any(tensor.device != query.device for tensor in tensors):
        raise ValueError("Batched fused D-Quant tensors must share one device")
    for lut in (key_decode_lut, value_decode_lut):
        if lut.ndim != 1 or lut.numel() != 256 or lut.dtype != torch.int32 or not lut.is_contiguous():
            raise ValueError("Fused rANS tables must be contiguous int32 tensors of length 256")
    if block_size <= 0 or block_tables.dtype not in (torch.int32, torch.int64):
        raise ValueError("Batched block tables require positive pages and integer indices")
    if head_size <= 0 or head_size & (head_size - 1):
        raise ValueError("Batched fused attention requires a power-of-two head size")
    for storage, scales, offsets, container_bytes in (
        (key_storage, key_scales, key_offsets, key_container_bytes),
        (value_storage, value_scales, value_offsets, value_container_bytes),
    ):
        if storage.ndim != 2 or storage.dtype != torch.uint8 or not storage.is_contiguous():
            raise ValueError("Fused records must be contiguous two-dimensional byte storage")
        if container_bytes < 4 or container_bytes > storage.shape[1]:
            raise ValueError("Fused rANS payload does not fit the record")
        if any(metadata.ndim != 1 or metadata.numel() != storage.shape[0] for metadata in (scales, offsets)):
            raise ValueError("Fused metadata must contain one scale and offset per record")
        if scales.stride() != offsets.stride():
            raise ValueError("Fused scale and offset strides must match")
        if storage.numel() >= 2**31:
            raise ValueError("Fused byte storage requires signed 32-bit address offsets")
    if key_window and (
        block_size % key_window
        or stripe_bytes % 2
        or mean_offset < key_container_bytes
        or mean_offset + stripe_bytes > key_storage.shape[1]
        or stripe_bytes * key_window < head_size * num_kv_heads * 2
    ):
        raise ValueError("Fused key mean stripes do not cover an aligned window")
    max_chunks = triton.cdiv(maximum, chunk_tokens)
    if partials is None:
        output = torch.empty((batch, max_chunks, query_heads, head_size), dtype=torch.float32, device=query.device)
        lse = torch.empty((batch, max_chunks, query_heads), dtype=torch.float32, device=query.device)
    else:
        output, lse = partials
        if (
            output.shape != (batch, max_chunks, query_heads, head_size)
            or lse.shape != output.shape[:3]
            or any(t.dtype != torch.float32 or t.device != query.device or not t.is_contiguous() for t in partials)
        ):
            raise ValueError("D-Quant partial workspace does not match the decode layout")
    rotated_query = hadamard_rotate_npu(query.float())
    if (
        native_backend is not None
        and compressed.dtype == torch.int32
        and compressed.is_contiguous()
        and native_backend.supports(query, block_tables, chunk_tokens, tile_tokens)
    ):
        native_backend(
            rotated_query,
            query,
            key_storage,
            value_storage,
            block_tables,
            key_decode_lut,
            value_decode_lut,
            output,
            lse,
            compressed,
            scale,
        )
        return output, lse, chunk_tokens
    single_window = (
        key_window and chunk_tokens == tile_tokens and chunk_tokens <= key_window and key_window % chunk_tokens == 0
    )
    kernel = _paged_dquant_batch_window_attention_kernel if single_window else _paged_dquant_batch_attention_kernel
    programs = batch * max_chunks
    launch_options = {}
    if single_window and max_chunks > 128 and query_heads == 12 and head_size == 256 and query.dtype == torch.bfloat16:
        # A bounded grid skips padded graph chunks using device lengths.
        partitions = 128
        kernel = _paged_dquant_batch_persistent_window_attention_kernel
        programs = batch * partitions
        launch_options = {"PARTITIONS": partitions, "has_auto_blockify_blacklist_op": True}
    kernel[(programs,)](
        rotated_query,
        query,
        key_storage,
        block_tables,
        key_decode_lut,
        key_scales,
        key_offsets,
        value_storage,
        block_tables,
        value_decode_lut,
        value_scales,
        value_offsets,
        output,
        lse,
        compressed,
        max_chunks,
        block_tables.stride(0),
        scale,
        NUM_Q_HEADS=query_heads,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_SIZE=head_size,
        BLOCK_SIZE=block_size,
        BLOCK_H=triton.next_power_of_2(query_heads),
        BLOCK_T=tile_tokens,
        CHUNK_TOKENS=chunk_tokens,
        KEY_ROW_BYTES=key_storage.shape[1],
        KEY_METADATA_STRIDE=key_scales.stride(0),
        KEY_CONTAINER_BYTES=key_container_bytes,
        VALUE_ROW_BYTES=value_storage.shape[1],
        VALUE_METADATA_STRIDE=value_scales.stride(0),
        VALUE_CONTAINER_BYTES=value_container_bytes,
        TABLE_SIZE=256,
        SCALE_BITS=8,
        LOWER_BOUND=1 << 23,
        KEY_WINDOW=key_window,
        MEAN_OFFSET=mean_offset,
        STRIPE_BYTES=stripe_bytes,
        IO_DTYPE=tl.bfloat16 if query.dtype == torch.bfloat16 else tl.float16,
        num_warps=4,
        **launch_options,
    )
    return output, lse, chunk_tokens


def paged_dquant_attention_npu(
    query: torch.Tensor,
    block_table: torch.Tensor,
    compressed: int,
    scale: float,
    key_storage: torch.Tensor,
    key_scales: torch.Tensor,
    key_offsets: torch.Tensor,
    key_decode_lut: torch.Tensor,
    value_storage: torch.Tensor,
    value_scales: torch.Tensor,
    value_offsets: torch.Tensor,
    value_decode_lut: torch.Tensor,
    *,
    block_size: int,
    head_size: int,
    num_kv_heads: int,
    key_container_bytes: int,
    value_container_bytes: int,
    key_window: int,
    mean_offset: int,
    stripe_bytes: int,
    chunk_tokens: int = 16,
    tile_tokens: int = 16,
):
    """Return normalized chunked attention and natural-log LSE.

    Output is shaped ``[chunks, query_heads, head_size]`` in the rotated basis.
    The caller merges chunks and applies the inverse Hadamard transform once.
    """
    if query.ndim != 3 or query.shape[0] != 1:
        raise ValueError("Fused D-Quant decode attention expects one query token")
    if num_kv_heads != 2:
        raise ValueError("Fused D-Quant decode attention currently requires two local KV heads")
    if compressed <= 0 or compressed % (key_window or 1):
        raise ValueError("Compressed length must be positive and window aligned")
    if tile_tokens < 16 or chunk_tokens <= 0 or chunk_tokens % tile_tokens:
        raise ValueError("Fused chunk must be a multiple of at least 16 token tiles")
    if tile_tokens & (tile_tokens - 1):
        raise ValueError("Fused token tile size must be a power of two")
    if key_window and stripe_bytes <= 0:
        raise ValueError("Windowed keys require a positive mean stripe")
    query_heads, width = query.shape[1], query.shape[2]
    if width != head_size or query_heads % num_kv_heads:
        raise ValueError("Query layout does not match the fused D-Quant KV heads")
    tensors = (
        query,
        block_table,
        key_storage,
        key_scales,
        key_offsets,
        key_decode_lut,
        value_storage,
        value_scales,
        value_offsets,
        value_decode_lut,
    )
    if any(tensor.device != query.device for tensor in tensors):
        raise ValueError("Fused D-Quant tensors must share one device")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Fused D-Quant attention requires FP16 or BF16 queries")
    for lut in (key_decode_lut, value_decode_lut):
        if lut.ndim != 1 or lut.numel() != 256 or lut.dtype != torch.int32 or not lut.is_contiguous():
            raise ValueError("Fused rANS tables must be contiguous int32 tensors of length 256")
    if not query.is_contiguous() or head_size <= 0 or head_size & (head_size - 1):
        raise ValueError("Fused queries must be contiguous with a power-of-two head size")
    if block_size <= 0 or block_table.numel() < triton.cdiv(compressed, block_size):
        raise ValueError("Block table must cover the compressed prefix")
    if block_table.ndim != 1 or block_table.dtype not in (torch.int32, torch.int64) or not block_table.is_contiguous():
        raise ValueError("Block table must be a contiguous one-dimensional integer tensor")
    for storage, scales, offsets, container_bytes in (
        (key_storage, key_scales, key_offsets, key_container_bytes),
        (value_storage, value_scales, value_offsets, value_container_bytes),
    ):
        if storage.ndim != 2 or storage.dtype != torch.uint8 or not storage.is_contiguous():
            raise ValueError("Fused records must be contiguous two-dimensional byte storage")
        if container_bytes < 4 or container_bytes > storage.shape[1]:
            raise ValueError("Fused rANS payload does not fit the record")
        if any(metadata.ndim != 1 or metadata.numel() != storage.shape[0] for metadata in (scales, offsets)):
            raise ValueError("Fused metadata must contain one scale and offset per record")
        if scales.stride() != offsets.stride():
            raise ValueError("Fused scale and offset strides must match")
        if storage.numel() >= 2**31:
            raise ValueError("Fused byte storage requires signed 32-bit address offsets")
    if key_window and (
        block_size % key_window
        or stripe_bytes % 2
        or mean_offset < key_container_bytes
        or mean_offset + stripe_bytes > key_storage.shape[1]
        or stripe_bytes * key_window < head_size * num_kv_heads * 2
    ):
        raise ValueError("Fused key mean stripes do not cover an aligned window")
    table_size = 256
    chunks = triton.cdiv(compressed, chunk_tokens)
    output = torch.empty((chunks, query_heads, head_size), dtype=torch.float32, device=query.device)
    lse = torch.empty((chunks, query_heads), dtype=torch.float32, device=query.device)
    rotated_query = hadamard_rotate_npu(query.float())
    # A single tile contained within a key window shares one mean vector.
    # Larger or cross-window chunks retain the general online-softmax kernel.
    single_window = (
        key_window and chunk_tokens == tile_tokens and chunk_tokens <= key_window and key_window % chunk_tokens == 0
    )
    _kernel = _paged_dquant_window_attention_kernel if single_window else _paged_dquant_fused_attention_kernel
    _kernel[(chunks,)](
        rotated_query,
        query,
        key_storage,
        block_table,
        key_decode_lut,
        key_scales,
        key_offsets,
        value_storage,
        block_table,
        value_decode_lut,
        value_scales,
        value_offsets,
        output,
        lse,
        compressed,
        scale,
        NUM_Q_HEADS=query_heads,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_SIZE=head_size,
        BLOCK_SIZE=block_size,
        BLOCK_H=triton.next_power_of_2(query_heads),
        BLOCK_T=tile_tokens,
        CHUNK_TOKENS=chunk_tokens,
        KEY_ROW_BYTES=key_storage.shape[1],
        KEY_METADATA_STRIDE=key_scales.stride(0),
        KEY_CONTAINER_BYTES=key_container_bytes,
        VALUE_ROW_BYTES=value_storage.shape[1],
        VALUE_METADATA_STRIDE=value_scales.stride(0),
        VALUE_CONTAINER_BYTES=value_container_bytes,
        TABLE_SIZE=table_size,
        SCALE_BITS=table_size.bit_length() - 1,
        LOWER_BOUND=1 << 23,
        KEY_WINDOW=key_window,
        MEAN_OFFSET=mean_offset,
        STRIPE_BYTES=stripe_bytes,
        IO_DTYPE=tl.bfloat16 if query.dtype == torch.bfloat16 else tl.float16,
        num_warps=4,
    )
    return output, lse
