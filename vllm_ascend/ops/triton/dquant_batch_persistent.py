# SPDX-License-Identifier: Apache-2.0
"""Batched one-window rANS attention specialization."""

from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.dquant_fused_window import _rans_step
from vllm_ascend.ops.triton.triton_utils import extract_slice, insert_slice


@triton.jit(do_not_specialize=["MAX_CHUNKS", "BLOCK_TABLE_STRIDE"])
def _paged_dquant_batch_persistent_window_attention_kernel(
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
    PARTITIONS: tl.constexpr = 128,
):
    """Decode actual device lengths with an optional bounded graph grid."""
    program = tl.program_id(0)
    if PARTITIONS:
        batch = program // PARTITIONS
        first = program % PARTITIONS
        step = PARTITIONS
    else:
        batch = program // MAX_CHUNKS
        first = program - batch * MAX_CHUNKS
        step = 1
    compressed = tl.load(compressed_ptr + batch).to(tl.int32)
    end = tl.cdiv(compressed, CHUNK_TOKENS)
    if PARTITIONS == 0:
        end = tl.minimum(first + 1, end)
    # Each iteration reuses the incoming pointers, rather than its predecessor's offsets.
    query_ptr_base = query_ptr
    original_query_ptr_base = original_query_ptr
    key_table_ptr_base = key_table_ptr
    value_table_ptr_base = value_table_ptr
    output_ptr_base = output_ptr
    lse_ptr_base = lse_ptr
    for chunk in range(first, end, step):
        query_ptr = query_ptr_base
        original_query_ptr = original_query_ptr_base
        key_table_ptr = key_table_ptr_base
        value_table_ptr = value_table_ptr_base
        output_ptr = output_ptr_base
        lse_ptr = lse_ptr_base
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
            | tl.load(key_payload + 1, mask=valid, other=0).to(tl.int32) << 8
            | tl.load(key_payload + 2, mask=valid, other=0).to(tl.int32) << 16
            | tl.load(key_payload + 3, mask=valid, other=0).to(tl.int32) << 24
        )
        key_cursor = tl.full((BLOCK_T,), 4, dtype=tl.int32)
        key_lut = tl.load(key_lut_ptr + tl.arange(0, TABLE_SIZE)).to(tl.float32, bitcast=True)
        key_scale = tl.load(key_scales_ptr + key_slot * KEY_METADATA_STRIDE, mask=valid, other=0).to(tl.float32)
        key_offset = tl.load(key_offsets_ptr + key_slot * KEY_METADATA_STRIDE, mask=valid, other=0).to(tl.float32)
        qk = tl.zeros((BLOCK_H, BLOCK_T), dtype=tl.float32)
        qk_mean = tl.zeros((BLOCK_H,), dtype=tl.float32)
        if KEY_WINDOW:
            mean_byte = (head_ids[:, None] // heads_per_kv * HEAD_SIZE + dims[None, :]) * 2
            mean_record = window * KEY_WINDOW + mean_byte // STRIPE_BYTES
            mean_block = tl.load(key_table_ptr + mean_record // BLOCK_SIZE, mask=head_mask[:, None], other=0).to(
                tl.int32
            )
            mean_slot = mean_block * BLOCK_SIZE + mean_record % BLOCK_SIZE
            mean_address = mean_slot * KEY_ROW_BYTES + MEAN_OFFSET + mean_byte % STRIPE_BYTES
            low = tl.load(key_storage_ptr + mean_address, mask=head_mask[:, None], other=0).to(tl.uint32)
            high = tl.load(key_storage_ptr + mean_address + 1, mask=head_mask[:, None], other=0).to(tl.uint32)
            bits = (low | high << 8).to(tl.uint16)
            if IO_DTYPE == tl.bfloat16:  # noqa: SIM300
                mean = bits.to(tl.bfloat16, bitcast=True).to(tl.float32)
            else:
                mean = bits.to(tl.float16, bitcast=True).to(tl.float32)
            qk_mean = tl.sum(original_query.to(tl.float32) * mean, axis=1)
        GROUP_H: tl.constexpr = BLOCK_H // NUM_KV_HEADS
        group_heads = tl.arange(0, GROUP_H)
        for kv in tl.static_range(NUM_KV_HEADS):
            local_qk = tl.zeros((GROUP_H, BLOCK_T), dtype=tl.float32)
            for channel in range(HEAD_SIZE):
                key_state, key_cursor, symbol = _rans_step(
                    key_state,
                    key_cursor,
                    key_payload,
                    key_lut,
                    KEY_CONTAINER_BYTES,
                    TABLE_SIZE,
                    SCALE_BITS,
                    LOWER_BOUND,
                )
                query_element = tl.load(
                    query_ptr + (kv * heads_per_kv + group_heads) * HEAD_SIZE + channel,
                    mask=group_heads < heads_per_kv,
                    other=0.0,
                ).to(tl.float32)
                local_qk += query_element[:, None] * symbol.to(tl.float32)[None, :]
            qk = insert_slice(qk, local_qk, (kv * heads_per_kv, 0), (GROUP_H, BLOCK_T), (1, 1))
        qk = qk * key_scale[None, :] + query_sum[:, None] * key_offset[None, :] + qk_mean[:, None]
        qk = tl.where(valid[None, :] & head_mask[:, None], qk * (SCALE * LOG2E), float("-inf"))
        maximum = tl.max(qk, axis=1)
        probabilities = tl.exp2(qk - maximum[:, None])
        denominator = tl.sum(probabilities, axis=1)
        lse = maximum + tl.log2(denominator)
        inverse = 1.0 / denominator
        # The batched caller passes the same block table for K and V.
        value_slot = key_slot
        value_payload = value_storage_ptr + value_slot * VALUE_ROW_BYTES
        value_state = (
            tl.load(value_payload, mask=valid, other=0).to(tl.int32)
            | tl.load(value_payload + 1, mask=valid, other=0).to(tl.int32) << 8
            | tl.load(value_payload + 2, mask=valid, other=0).to(tl.int32) << 16
            | tl.load(value_payload + 3, mask=valid, other=0).to(tl.int32) << 24
        )
        value_cursor = tl.full((BLOCK_T,), 4, dtype=tl.int32)
        value_lut = tl.load(value_lut_ptr + tl.arange(0, TABLE_SIZE)).to(tl.float32, bitcast=True)
        value_scale = tl.load(value_scales_ptr + value_slot * VALUE_METADATA_STRIDE, mask=valid, other=0).to(tl.float32)
        value_offset = tl.load(value_offsets_ptr + value_slot * VALUE_METADATA_STRIDE, mask=valid, other=0).to(
            tl.float32
        )
        weighted = probabilities * value_scale[None, :]
        offset_part = tl.sum(probabilities * value_offset[None, :], axis=1)
        value_acc = tl.zeros((HEAD_SIZE, BLOCK_H), dtype=tl.float32)
        output_ptr += batch * MAX_CHUNKS * NUM_Q_HEADS * HEAD_SIZE
        lse_ptr += batch * MAX_CHUNKS * NUM_Q_HEADS
        for kv in range(NUM_KV_HEADS):
            local_weighted = extract_slice(weighted, (kv * heads_per_kv, 0), (GROUP_H, BLOCK_T), (1, 1))
            for channel in range(HEAD_SIZE):
                value_state, value_cursor, symbol = _rans_step(
                    value_state,
                    value_cursor,
                    value_payload,
                    value_lut,
                    VALUE_CONTAINER_BYTES,
                    TABLE_SIZE,
                    SCALE_BITS,
                    LOWER_BOUND,
                )
                contribution = tl.sum(local_weighted * symbol.to(tl.float32)[None, :], axis=1)
                value_acc = insert_slice(
                    value_acc, contribution[None, :], (channel, kv * heads_per_kv), (1, GROUP_H), (1, 1)
                )
        tl.store(
            output_ptr + (chunk * NUM_Q_HEADS + head_ids[:, None]) * HEAD_SIZE + dims[None, :],
            (tl.trans(value_acc) + offset_part[:, None]) * inverse[:, None],
            mask=head_mask[:, None],
        )
        tl.store(lse_ptr + chunk * NUM_Q_HEADS + head_ids, lse * 0.6931471805599453, mask=head_mask)
