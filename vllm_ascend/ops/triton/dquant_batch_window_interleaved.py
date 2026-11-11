# SPDX-License-Identifier: Apache-2.0
"""Batched one-window rANS attention specialization."""

from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.dquant_fused_window import _rans_step


@triton.jit(do_not_specialize=["MAX_CHUNKS", "BLOCK_TABLE_STRIDE"])
def _paged_dquant_batch_window_attention_kernel(
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
    value_symbol_ptr,
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
    """Decode one key window for each request in a padded batch."""
    program = tl.program_id(0)
    batch = program // MAX_CHUNKS
    chunk = program - batch * MAX_CHUNKS
    compressed = tl.load(compressed_ptr + batch).to(tl.int32)
    active = chunk * CHUNK_TOKENS < compressed
    if active == 0:
        return
    chunk_start = chunk * CHUNK_TOKENS
    head_ids = tl.arange(0, BLOCK_H)
    dims = tl.arange(0, HEAD_SIZE)
    head_mask = head_ids < NUM_Q_HEADS
    tokens = chunk_start + tl.arange(0, BLOCK_T)
    valid = tokens < compressed
    safe = tl.where(valid, tokens, 0)
    LOG2E: tl.constexpr = 1.4426950408889634
    window = chunk_start // KEY_WINDOW if KEY_WINDOW else 0
    heads_per_kv: tl.constexpr = NUM_Q_HEADS // NUM_KV_HEADS

    query_ptr += batch * NUM_Q_HEADS * HEAD_SIZE
    original_query_ptr += batch * NUM_Q_HEADS * HEAD_SIZE
    key_table_ptr += batch * BLOCK_TABLE_STRIDE
    value_table_ptr += batch * BLOCK_TABLE_STRIDE
    value_symbol_ptr += program * BLOCK_T * HEAD_SIZE * NUM_KV_HEADS
    rotated_query = tl.load(
        query_ptr + head_ids[:, None] * HEAD_SIZE + dims[None, :], mask=head_mask[:, None], other=0.0
    )
    original_query = tl.load(
        original_query_ptr + head_ids[:, None] * HEAD_SIZE + dims[None, :], mask=head_mask[:, None], other=0.0
    )
    query_sum = tl.sum(rotated_query, axis=1)

    key_block = tl.load(key_table_ptr + safe // BLOCK_SIZE, mask=valid, other=0).to(tl.int32)
    key_slot = key_block * BLOCK_SIZE + safe % BLOCK_SIZE
    key_payload = key_storage_ptr + key_slot * KEY_ROW_BYTES
    key_state = (
        tl.load(key_payload, mask=valid, other=0).to(tl.int32)
        | (tl.load(key_payload + 1, mask=valid, other=0).to(tl.int32) << 8)
        | (tl.load(key_payload + 2, mask=valid, other=0).to(tl.int32) << 16)
        | (tl.load(key_payload + 3, mask=valid, other=0).to(tl.int32) << 24)
    )
    key_cursor = tl.full((BLOCK_T,), 4, dtype=tl.int32)
    key_lut = tl.load(key_lut_ptr + tl.arange(0, TABLE_SIZE)).to(tl.float32, bitcast=True)
    key_scale = tl.load(key_scales_ptr + key_slot * KEY_METADATA_STRIDE, mask=valid, other=0).to(tl.float32)
    key_offset = tl.load(key_offsets_ptr + key_slot * KEY_METADATA_STRIDE, mask=valid, other=0).to(tl.float32)

    value_block = tl.load(value_table_ptr + safe // BLOCK_SIZE, mask=valid, other=0).to(tl.int32)
    value_slot = value_block * BLOCK_SIZE + safe % BLOCK_SIZE
    value_payload = value_storage_ptr + value_slot * VALUE_ROW_BYTES
    value_state = (
        tl.load(value_payload, mask=valid, other=0).to(tl.int32)
        | (tl.load(value_payload + 1, mask=valid, other=0).to(tl.int32) << 8)
        | (tl.load(value_payload + 2, mask=valid, other=0).to(tl.int32) << 16)
        | (tl.load(value_payload + 3, mask=valid, other=0).to(tl.int32) << 24)
    )
    value_cursor = tl.full((BLOCK_T,), 4, dtype=tl.int32)
    value_lut = tl.load(value_lut_ptr + tl.arange(0, TABLE_SIZE)).to(tl.float32, bitcast=True)
    value_scale = tl.load(value_scales_ptr + value_slot * VALUE_METADATA_STRIDE, mask=valid, other=0).to(tl.float32)
    value_offset = tl.load(value_offsets_ptr + value_slot * VALUE_METADATA_STRIDE, mask=valid, other=0).to(tl.float32)

    qk = tl.zeros((BLOCK_H, BLOCK_T), dtype=tl.float32)
    qk_mean = tl.zeros((BLOCK_H,), dtype=tl.float32)
    if KEY_WINDOW:
        mean_byte = ((head_ids[:, None] // heads_per_kv) * HEAD_SIZE + dims[None, :]) * 2
        mean_record = window * KEY_WINDOW + mean_byte // STRIPE_BYTES
        mean_block = tl.load(key_table_ptr + mean_record // BLOCK_SIZE, mask=head_mask[:, None], other=0).to(tl.int32)
        mean_slot = mean_block * BLOCK_SIZE + mean_record % BLOCK_SIZE
        mean_address = mean_slot * KEY_ROW_BYTES + MEAN_OFFSET + mean_byte % STRIPE_BYTES
        low = tl.load(key_storage_ptr + mean_address, mask=head_mask[:, None], other=0).to(tl.uint32)
        high = tl.load(key_storage_ptr + mean_address + 1, mask=head_mask[:, None], other=0).to(tl.uint32)
        bits = (low | (high << 8)).to(tl.uint16)
        if IO_DTYPE == tl.bfloat16:  # noqa: SIM300
            mean = bits.to(tl.bfloat16, bitcast=True).to(tl.float32)
        else:
            mean = bits.to(tl.float16, bitcast=True).to(tl.float32)
        qk_mean = tl.sum(original_query.to(tl.float32) * mean, axis=1)
    for dimension in range(HEAD_SIZE * NUM_KV_HEADS):
        key_state, key_cursor, symbol = _rans_step(
            key_state, key_cursor, key_payload, key_lut, KEY_CONTAINER_BYTES, TABLE_SIZE, SCALE_BITS, LOWER_BOUND
        )
        value_state, value_cursor, value_symbol = _rans_step(
            value_state,
            value_cursor,
            value_payload,
            value_lut,
            VALUE_CONTAINER_BYTES,
            TABLE_SIZE,
            SCALE_BITS,
            LOWER_BOUND,
        )
        tl.store(value_symbol_ptr + dimension * BLOCK_T + tl.arange(0, BLOCK_T), value_symbol, mask=valid)
        kv = dimension // HEAD_SIZE
        q_mask = head_mask & (head_ids // heads_per_kv == kv)
        query_element = tl.load(query_ptr + head_ids * HEAD_SIZE + dimension % HEAD_SIZE, mask=q_mask, other=0.0).to(
            tl.float32
        )
        qk += query_element[:, None] * symbol.to(tl.float32)[None, :]

    qk = qk * key_scale[None, :] + query_sum[:, None] * key_offset[None, :] + qk_mean[:, None]
    qk = tl.where(valid[None, :] & head_mask[:, None], qk * (SCALE * LOG2E), float("-inf"))
    maximum = tl.max(qk, axis=1)
    probabilities = tl.exp2(qk - maximum[:, None])
    denominator = tl.sum(probabilities, axis=1)
    lse = maximum + tl.log2(denominator)
    inverse = 1.0 / denominator

    weighted = probabilities * value_scale[None, :]
    offset_part = tl.sum(probabilities * value_offset[None, :], axis=1)
    value_acc = tl.zeros((BLOCK_H, HEAD_SIZE), dtype=tl.float32)
    output_ptr += batch * MAX_CHUNKS * NUM_Q_HEADS * HEAD_SIZE
    lse_ptr += batch * MAX_CHUNKS * NUM_Q_HEADS
    for dimension in range(HEAD_SIZE * NUM_KV_HEADS):
        symbol = tl.load(value_symbol_ptr + dimension * BLOCK_T + tl.arange(0, BLOCK_T), mask=valid, other=0)
        contribution = tl.sum(weighted * symbol.to(tl.float32)[None, :], axis=1)
        contribution += offset_part
        output_mask = (head_ids[:, None] // heads_per_kv == dimension // HEAD_SIZE) & (
            dims[None, :] == dimension % HEAD_SIZE
        )
        value_acc += tl.where(output_mask, contribution[:, None], 0.0)
    tl.store(
        output_ptr + (chunk * NUM_Q_HEADS + head_ids[:, None]) * HEAD_SIZE + dims[None, :],
        value_acc * inverse[:, None],
        mask=head_mask[:, None],
    )
    tl.store(lse_ptr + chunk * NUM_Q_HEADS + head_ids, lse * 0.6931471805599453, mask=head_mask)
