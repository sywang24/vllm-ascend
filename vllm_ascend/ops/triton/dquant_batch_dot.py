# SPDX-License-Identifier: Apache-2.0
"""Cross-request vectorized one-window rANS attention specialization."""

from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.dquant_fused_window import _rans_step


@triton.jit(do_not_specialize=["MAX_CHUNKS", "BLOCK_TABLE_STRIDE"])
def _paged_dquant_batch_vector_window_attention_kernel(
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
    BATCH: tl.constexpr,
    BLOCK_D: tl.constexpr,
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
    """Decode one key window for all requests in the same vector lanes."""
    chunk = tl.program_id(0)
    chunk_start = chunk * CHUNK_TOKENS
    batch_ids = tl.arange(0, BATCH)
    head_ids = tl.arange(0, BLOCK_H)
    dims = tl.arange(0, HEAD_SIZE)
    token_offsets = tl.arange(0, BLOCK_T)
    tokens = chunk_start + token_offsets
    compressed = tl.load(compressed_ptr + batch_ids).to(tl.int32)
    active = chunk_start < compressed
    valid = (tokens[None, :] < compressed[:, None]) & active[:, None]
    safe = tl.where(valid, tokens[None, :], 0)
    valid_flat = tl.reshape(valid, (BATCH * BLOCK_T,))
    safe_flat = tl.reshape(safe, (BATCH * BLOCK_T,))
    head_mask = head_ids < NUM_Q_HEADS
    heads_per_kv: tl.constexpr = NUM_Q_HEADS // NUM_KV_HEADS
    LOG2E: tl.constexpr = 1.4426950408889634
    window = chunk_start // KEY_WINDOW if KEY_WINDOW else 0

    query = tl.load(
        query_ptr
        + batch_ids[:, None, None] * NUM_Q_HEADS * HEAD_SIZE
        + head_ids[None, :, None] * HEAD_SIZE
        + dims[None, None, :],
        mask=head_mask[None, :, None],
        other=0.0,
    )
    original_query = tl.load(
        original_query_ptr
        + batch_ids[:, None, None] * NUM_Q_HEADS * HEAD_SIZE
        + head_ids[None, :, None] * HEAD_SIZE
        + dims[None, None, :],
        mask=head_mask[None, :, None],
        other=0.0,
    )
    query_sum = tl.sum(query, axis=2)

    block_offsets = batch_ids[:, None] * BLOCK_TABLE_STRIDE + safe // BLOCK_SIZE
    block_offsets_flat = tl.reshape(block_offsets, (BATCH * BLOCK_T,))
    key_block = tl.load(key_table_ptr + block_offsets_flat, mask=valid_flat, other=0).to(tl.int32)
    key_slot_flat = key_block * BLOCK_SIZE + safe_flat % BLOCK_SIZE
    key_slot = tl.reshape(key_slot_flat, (BATCH, BLOCK_T))
    key_payload_flat = key_storage_ptr + key_slot_flat * KEY_ROW_BYTES
    key_state = (
        tl.load(key_payload_flat, mask=valid_flat, other=0).to(tl.int32)
        | (tl.load(key_payload_flat + 1, mask=valid_flat, other=0).to(tl.int32) << 8)
        | (tl.load(key_payload_flat + 2, mask=valid_flat, other=0).to(tl.int32) << 16)
        | (tl.load(key_payload_flat + 3, mask=valid_flat, other=0).to(tl.int32) << 24)
    )
    key_cursor = tl.full((BATCH * BLOCK_T,), 4, dtype=tl.int32)
    key_lut = tl.load(key_lut_ptr + tl.arange(0, TABLE_SIZE)).to(tl.float32, bitcast=True)
    key_scale = tl.load(
        key_scales_ptr + key_slot * KEY_METADATA_STRIDE,
        mask=valid,
        other=0,
    ).to(tl.float32)
    key_offset = tl.load(
        key_offsets_ptr + key_slot * KEY_METADATA_STRIDE,
        mask=valid,
        other=0,
    ).to(tl.float32)

    qk = tl.zeros((BATCH, BLOCK_H, BLOCK_T), dtype=tl.float32)
    qk_mean = tl.zeros((BATCH, BLOCK_H), dtype=tl.float32)
    if KEY_WINDOW:
        mean_byte = ((head_ids[:, None] // heads_per_kv) * HEAD_SIZE + dims[None, :]) * 2
        mean_record = window * KEY_WINDOW + mean_byte // STRIPE_BYTES
        mean_block = tl.load(
            key_table_ptr + batch_ids[:, None, None] * BLOCK_TABLE_STRIDE + mean_record[None, :, :] // BLOCK_SIZE,
            mask=head_mask[None, :, None],
            other=0,
        ).to(tl.int32)
        mean_slot = mean_block * BLOCK_SIZE + mean_record[None, :, :] % BLOCK_SIZE
        mean_address = mean_slot * KEY_ROW_BYTES + MEAN_OFFSET + mean_byte[None, :, :] % STRIPE_BYTES
        low = tl.load(
            key_storage_ptr + mean_address,
            mask=head_mask[None, :, None],
            other=0,
        ).to(tl.uint32)
        high = tl.load(
            key_storage_ptr + mean_address + 1,
            mask=head_mask[None, :, None],
            other=0,
        ).to(tl.uint32)
        bits = (low | (high << 8)).to(tl.uint16)
        if IO_DTYPE == tl.bfloat16:  # noqa: SIM300
            mean = bits.to(tl.bfloat16, bitcast=True).to(tl.float32)
        else:
            mean = bits.to(tl.float16, bitcast=True).to(tl.float32)
        qk_mean = tl.sum(original_query * mean, axis=2)

    dim_block = tl.arange(0, BLOCK_D)
    for dimension_base in range(HEAD_SIZE * NUM_KV_HEADS // BLOCK_D):
        symbol_rows = tl.zeros((BLOCK_D, BATCH * BLOCK_T), dtype=IO_DTYPE)
        for column in tl.static_range(BLOCK_D):
            key_state, key_cursor, symbol = _rans_step(
                key_state,
                key_cursor,
                key_payload_flat,
                key_lut,
                KEY_CONTAINER_BYTES,
                TABLE_SIZE,
                SCALE_BITS,
                LOWER_BOUND,
            )
            symbol_rows += tl.where(
                dim_block[:, None] == column,
                symbol[None, :].to(IO_DTYPE),
                0.0,
            )
        dimensions = dimension_base * BLOCK_D + dim_block
        kv: tl.constexpr = dimension_base * BLOCK_D // HEAD_SIZE
        gather_index = dim_block[:, None] * 0 + tl.arange(0, BLOCK_T)[None, :]
        for batch_index in tl.static_range(BATCH):
            symbols = tl.gather(
                symbol_rows,
                gather_index + batch_index * BLOCK_T,
                axis=1,
            )
            query_tile = tl.load(
                query_ptr + batch_index * NUM_Q_HEADS * HEAD_SIZE + head_ids[:, None] * HEAD_SIZE + dimensions[None, :],
                mask=head_mask[:, None] & (head_ids[:, None] // heads_per_kv == kv),
                other=0.0,
            ).to(IO_DTYPE)
            score = tl.dot(
                query_tile,
                symbols,
                out_dtype=tl.float32,
            )
            qk += tl.where(
                batch_ids[:, None, None] == batch_index,
                score[None, :, :],
                0.0,
            )

    qk = qk * key_scale[:, None, :] + query_sum[:, :, None] * key_offset[:, None, :] + qk_mean[:, :, None]
    qk = tl.where(
        valid[:, None, :] & head_mask[None, :, None],
        qk * (SCALE * LOG2E),
        float("-inf"),
    )
    maximum = tl.max(qk, axis=2)
    maximum = tl.where(active[:, None], maximum, 0.0)
    probabilities = tl.exp2(qk - maximum[:, :, None])
    denominator = tl.sum(probabilities, axis=2)
    lse = maximum + tl.log2(denominator)
    inverse = 1.0 / denominator

    value_block = tl.load(value_table_ptr + block_offsets_flat, mask=valid_flat, other=0).to(tl.int32)
    value_slot_flat = value_block * BLOCK_SIZE + safe_flat % BLOCK_SIZE
    value_slot = tl.reshape(value_slot_flat, (BATCH, BLOCK_T))
    value_payload_flat = value_storage_ptr + value_slot_flat * VALUE_ROW_BYTES
    value_state = (
        tl.load(value_payload_flat, mask=valid_flat, other=0).to(tl.int32)
        | (tl.load(value_payload_flat + 1, mask=valid_flat, other=0).to(tl.int32) << 8)
        | (tl.load(value_payload_flat + 2, mask=valid_flat, other=0).to(tl.int32) << 16)
        | (tl.load(value_payload_flat + 3, mask=valid_flat, other=0).to(tl.int32) << 24)
    )
    value_cursor = tl.full((BATCH * BLOCK_T,), 4, dtype=tl.int32)
    value_lut = tl.load(value_lut_ptr + tl.arange(0, TABLE_SIZE)).to(tl.float32, bitcast=True)
    value_scale = tl.load(
        value_scales_ptr + value_slot * VALUE_METADATA_STRIDE,
        mask=valid,
        other=0,
    ).to(tl.float32)
    value_offset = tl.load(
        value_offsets_ptr + value_slot * VALUE_METADATA_STRIDE,
        mask=valid,
        other=0,
    ).to(tl.float32)
    weighted = probabilities * value_scale[:, None, :]
    offset_part = tl.sum(probabilities * value_offset[:, None, :], axis=2)
    for dimension_base in range(HEAD_SIZE * NUM_KV_HEADS // BLOCK_D):
        symbol_rows = tl.zeros((BLOCK_D, BATCH * BLOCK_T), dtype=IO_DTYPE)
        for column in tl.static_range(BLOCK_D):
            value_state, value_cursor, symbol = _rans_step(
                value_state,
                value_cursor,
                value_payload_flat,
                value_lut,
                VALUE_CONTAINER_BYTES,
                TABLE_SIZE,
                SCALE_BITS,
                LOWER_BOUND,
            )
            symbol_rows += tl.where(
                dim_block[:, None] == column,
                symbol[None, :].to(IO_DTYPE),
                0.0,
            )
        dimensions = dimension_base * BLOCK_D + dim_block
        gather_index = dim_block[:, None] * 0 + tl.arange(0, BLOCK_T)[None, :]
        for batch_index in tl.static_range(BATCH):
            symbols = tl.gather(
                symbol_rows,
                gather_index + batch_index * BLOCK_T,
                axis=1,
            )
            weighted_batch = tl.sum(
                weighted * (batch_ids[:, None, None] == batch_index),
                axis=0,
            )
            contribution = tl.dot(
                weighted_batch.to(IO_DTYPE),
                tl.trans(symbols),
                out_dtype=tl.float32,
            )
            offset_batch = tl.sum(
                offset_part * (batch_ids[:, None] == batch_index),
                axis=0,
            )
            contribution += offset_batch[:, None]
            output_mask = head_ids[:, None] // heads_per_kv == dimensions[None, :] // HEAD_SIZE
            inverse_batch = tl.sum(
                inverse * (batch_ids[:, None] == batch_index),
                axis=0,
            )
            active_batch = tl.sum(
                active.to(tl.int32) * (batch_ids == batch_index),
                axis=0,
            )
            tl.store(
                output_ptr
                + batch_index * MAX_CHUNKS * NUM_Q_HEADS * HEAD_SIZE
                + chunk * NUM_Q_HEADS * HEAD_SIZE
                + head_ids[:, None] * HEAD_SIZE
                + (dimensions % HEAD_SIZE)[None, :],
                contribution * inverse_batch[:, None],
                mask=output_mask & (active_batch != 0),
            )

    tl.store(
        lse_ptr + batch_ids[:, None] * MAX_CHUNKS * NUM_Q_HEADS + chunk * NUM_Q_HEADS + head_ids[None, :],
        lse * 0.6931471805599453,
        mask=active[:, None] & head_mask[None, :],
    )
