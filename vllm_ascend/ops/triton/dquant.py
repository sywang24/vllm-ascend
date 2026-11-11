# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ascend Triton kernels for D-Quant preprocessing and decoding."""

import math

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _hadamard_rotate_kernel(
    input_ptr,
    output_ptr,
    row_stride,
    WIDTH: tl.constexpr,
    LOG2_WIDTH: tl.constexpr,
    BLOCK_WIDTH: tl.constexpr,
):
    row = tl.program_id(0)
    indices = tl.arange(0, BLOCK_WIDTH)
    values = tl.load(input_ptr + row * row_stride + indices, mask=indices < WIDTH, other=0).to(tl.float32)

    for stage in tl.static_range(LOG2_WIDTH):
        stride = 1 << stage
        partner = tl.gather(values, indices ^ stride, axis=0)
        values = tl.where((indices & stride) == 0, values + partner, partner - values)

    values = values * (1.0 / tl.sqrt(float(WIDTH)))
    tl.store(output_ptr + row * WIDTH + indices, values, mask=indices < WIDTH)


def hadamard_rotate_npu(values: torch.Tensor) -> torch.Tensor:
    """Apply an orthonormal Hadamard transform to each contiguous last-dim row.

    Rows represent individual attention heads. The last dimension must be a
    power of two, matching the per-head rotation used by D-Quant.
    """
    if values.ndim < 2:
        raise ValueError("values must have at least two dimensions")
    width = values.shape[-1]
    if width <= 0 or width & (width - 1):
        raise ValueError("the last dimension must be a positive power of two")
    if not values.is_contiguous():
        values = values.contiguous()

    rows = values.numel() // width
    output = torch.empty(values.shape, dtype=torch.float32, device=values.device)
    _hadamard_rotate_kernel[(rows,)](
        values,
        output,
        width,
        WIDTH=width,
        LOG2_WIDTH=width.bit_length() - 1,
        BLOCK_WIDTH=width,
        num_warps=4,
    )
    return output.to(values.dtype)


@triton.jit
def _rans_decode_kernel(
    payload_ptr,
    symbol_lut_ptr,
    frequency_lut_ptr,
    cumulative_lut_ptr,
    output_ptr,
    CONTAINER_BYTES: tl.constexpr,
    SYMBOL_COUNT: tl.constexpr,
    TABLE_SIZE: tl.constexpr,
    SCALE_BITS: tl.constexpr,
    LOWER_BOUND: tl.constexpr,
):
    stream = tl.program_id(0)
    payload = payload_ptr + stream * CONTAINER_BYTES
    state = (
        tl.load(payload + 0).to(tl.uint32)
        | (tl.load(payload + 1).to(tl.uint32) << 8)
        | (tl.load(payload + 2).to(tl.uint32) << 16)
        | (tl.load(payload + 3).to(tl.uint32) << 24)
    )
    cursor = 4

    for index in range(int(SYMBOL_COUNT)):
        slot = state & (TABLE_SIZE - 1)
        symbol = tl.load(symbol_lut_ptr + slot).to(tl.int16)
        frequency = tl.load(frequency_lut_ptr + slot).to(tl.uint32)
        cumulative = tl.load(cumulative_lut_ptr + slot).to(tl.uint32)
        state = frequency * (state >> SCALE_BITS) + slot - cumulative
        while state < LOWER_BOUND and cursor < CONTAINER_BYTES:
            byte = tl.load(payload + cursor, mask=cursor < CONTAINER_BYTES, other=0).to(tl.uint32)
            state = (state << 8) | byte
            cursor += 1
        tl.store(output_ptr + stream * SYMBOL_COUNT + index, symbol)


@triton.jit
def _rans_encode_kernel(
    symbols_ptr,
    frequency_ptr,
    cumulative_ptr,
    output_ptr,
    status_ptr,
    CONTAINER_BYTES: tl.constexpr,
    BYTE_BLOCK: tl.constexpr,
    SYMBOL_COUNT: tl.constexpr,
    ALPHABET_SIZE: tl.constexpr,
    SCALE_BITS: tl.constexpr,
    LOWER_BOUND: tl.constexpr,
):
    stream = tl.program_id(0)
    symbol_row = symbols_ptr + stream * SYMBOL_COUNT
    output_row = output_ptr + stream * CONTAINER_BYTES
    byte_offsets = tl.arange(0, BYTE_BLOCK)
    tl.store(output_row + byte_offsets, 0, mask=byte_offsets < CONTAINER_BYTES)

    state = tl.full((), LOWER_BOUND, dtype=tl.uint32)
    cursor = tl.full((), CONTAINER_BYTES, dtype=tl.int32)
    invalid_symbol = tl.full((), 0, dtype=tl.int32)
    renormalization_limit = (LOWER_BOUND >> SCALE_BITS) << 8
    for index in range(int(SYMBOL_COUNT)):
        symbol = tl.load(symbol_row + SYMBOL_COUNT - index - 1).to(tl.int32)
        valid_symbol = (symbol >= 0) & (symbol < ALPHABET_SIZE)
        frequency = tl.load(frequency_ptr + symbol, mask=valid_symbol, other=1).to(tl.uint32)
        cumulative = tl.load(cumulative_ptr + symbol, mask=valid_symbol, other=0).to(tl.uint32)
        invalid_symbol |= (~valid_symbol | (frequency == 0)).to(tl.int32)
        frequency = tl.maximum(frequency, 1)
        maximum_state = renormalization_limit * frequency
        while state >= maximum_state:
            cursor -= 1
            tl.store(output_row + cursor, (state & 0xFF).to(tl.uint8), mask=cursor >= 4)
            state >>= 8
        state = ((state // frequency) << SCALE_BITS) + (state % frequency) + cumulative

    state = state.to(tl.uint32)
    for byte_index in tl.static_range(4):
        state_byte = ((state >> (8 * byte_index)) & 0xFF).to(tl.uint8)
        tl.store(output_row + byte_index, state_byte)

    emitted_bytes = CONTAINER_BYTES - cursor
    # Overflow is a reported error, never permission to read before the buffer
    # or copy past its end. The caller must discard records with nonzero status.
    if cursor >= 4:
        copy_index = tl.full((), 0, dtype=tl.int32)
        while copy_index < emitted_bytes:
            byte = tl.load(output_row + cursor + copy_index)
            tl.store(output_row + 4 + copy_index, byte)
            copy_index += 1
        padding_index = 4 + emitted_bytes
        while padding_index < CONTAINER_BYTES:
            tl.store(output_row + padding_index, 0)
            padding_index += 1
    tl.store(status_ptr + stream, (cursor < 4).to(tl.int32) | (invalid_symbol * 2))


@triton.jit
def _dquant_assign(values, scale, offset, rate_ptr, penalty, LEVELS: tl.constexpr, BLOCK_WIDTH: tl.constexpr):
    best_cost = tl.full((BLOCK_WIDTH,), float("inf"), tl.float32)
    assigned = tl.full((BLOCK_WIDTH,), 0, tl.int32)
    assigned_rates = tl.full((BLOCK_WIDTH,), 0, tl.float32)
    for symbol in range(int(LEVELS)):
        rate = tl.load(rate_ptr + symbol).to(tl.float32)
        error = values - (offset + scale * symbol)
        cost = error * error + penalty * rate
        better = cost < best_cost
        assigned = tl.where(better, symbol, assigned)
        assigned_rates = tl.where(better, rate, assigned_rates)
        best_cost = tl.minimum(best_cost, cost)
    return assigned, assigned_rates


@triton.jit
def _dquant_affine_assign(values, scale, offset, rate_ptr, LEVELS: tl.constexpr, BLOCK_WIDTH: tl.constexpr):
    normalized = (values - offset) / scale
    base = tl.floor(normalized).to(tl.int32)
    assigned = tl.minimum(LEVELS - 1, tl.maximum(0, base - 1))
    error = values - (offset + scale * assigned)
    best_cost = error * error
    for delta in tl.static_range(3):
        symbol = tl.minimum(LEVELS - 1, tl.maximum(0, base + delta))
        error = values - (offset + scale * symbol)
        cost = error * error
        better = cost < best_cost
        assigned = tl.where(better, symbol, assigned)
        best_cost = tl.minimum(best_cost, cost)
    # Gather within unified buffer instead of issuing per-value HBM lookups.
    indices = tl.arange(0, BLOCK_WIDTH)
    rates = tl.load(rate_ptr + indices, mask=indices < LEVELS, other=0)
    # Near a midpoint, compiler rounding can change the exhaustive scan's tie.
    # Preserve that scan for the entire row, including the metadata refit.
    FP32_ROUNDING_MARGIN: tl.constexpr = 4.0 * 2.0**-23
    margin = (tl.abs(offset / scale) + LEVELS) * FP32_ROUNDING_MARGIN
    ambiguous = tl.abs(normalized - base.to(tl.float32) - 0.5) <= margin
    return assigned, tl.gather(rates, assigned, axis=0), ambiguous


@triton.jit
def _dquant_drift_reference(
    values,
    rate_ptr,
    scale,
    offset,
    WIDTH: tl.constexpr,
    LEVELS: tl.constexpr,
    BLOCK_WIDTH: tl.constexpr,
    TARGET_RATE_BITS: tl.constexpr,
    DRIFT_STEPS: tl.constexpr,
):
    indices = tl.arange(0, BLOCK_WIDTH)
    # Walk the alphabet instead of materializing [1024, 8] cost matrices.
    # Keeping only per-channel extrema fits the 910B's unified buffer.
    row_min = tl.full((BLOCK_WIDTH,), float("inf"), tl.float32)
    row_max = tl.full((BLOCK_WIDTH,), float("-inf"), tl.float32)
    initial_cost_min = tl.full((BLOCK_WIDTH,), float("inf"), tl.float32)
    assigned = tl.full((BLOCK_WIDTH,), 0, tl.int32)
    assigned_rates = tl.full((BLOCK_WIDTH,), 0, tl.float32)
    for symbol in range(int(LEVELS)):
        error = values - (offset + scale * symbol)
        distortion = error * error
        rate = tl.load(rate_ptr + symbol).to(tl.float32)
        initial_cost = distortion + 0.0 * rate
        better = initial_cost < initial_cost_min
        assigned = tl.where(better, symbol, assigned)
        assigned_rates = tl.where(better, rate, assigned_rates)
        initial_cost_min = tl.minimum(initial_cost_min, initial_cost)
        row_min = tl.minimum(row_min, distortion)
        row_max = tl.maximum(row_max, distortion)
    max_distortion = tl.max(tl.where(indices < WIDTH, row_max, float("-inf")), axis=0)
    min_distortion = tl.min(tl.where(indices < WIDTH, row_min, float("inf")), axis=0)
    if max_distortion > min_distortion:
        # The unconstrained minimum is optimal when it already fits the rate
        # budget. High-fidelity profiles commonly avoid the entire search.
        assigned_rate = tl.sum(tl.where(indices < WIDTH, assigned_rates, 0.0), axis=0)
        if assigned_rate > TARGET_RATE_BITS:
            high = tl.log2(8.0 * (max_distortion - min_distortion))
            low = high - 40.0
            # A feasible last iteration already computed the final high bound.
            last_feasible = tl.full((), False, tl.int1)
            for _ in range(int(DRIFT_STEPS)):
                midpoint = (low + high) * 0.5
                penalty = tl.exp2(midpoint)
                assigned, assigned_rates = _dquant_assign(values, scale, offset, rate_ptr, penalty, LEVELS, BLOCK_WIDTH)
                assigned_rate = tl.sum(tl.where(indices < WIDTH, assigned_rates, 0.0), axis=0)
                last_feasible = assigned_rate <= TARGET_RATE_BITS
                if assigned_rate <= TARGET_RATE_BITS:
                    high = midpoint
                else:
                    low = midpoint
            if not last_feasible:
                assigned, _fallback_rates = _dquant_assign(
                    values, scale, offset, rate_ptr, tl.exp2(high), LEVELS, BLOCK_WIDTH
                )
    else:
        levels = tl.arange(0, BLOCK_WIDTH)
        rates = tl.load(rate_ptr + levels, mask=levels < LEVELS, other=float("inf"))
        cheapest = tl.argmin(rates, axis=0)
        assigned = tl.full((BLOCK_WIDTH,), cheapest, tl.int32)
    return assigned


@triton.jit
def _dquant_drift(
    values,
    rate_ptr,
    scale,
    offset,
    WIDTH: tl.constexpr,
    LEVELS: tl.constexpr,
    BLOCK_WIDTH: tl.constexpr,
    TARGET_RATE_BITS: tl.constexpr,
    DRIFT_STEPS: tl.constexpr,
    FAST_AFFINE: tl.constexpr = False,
):
    if FAST_AFFINE:
        # Avoid underflowed costs, overflow and grids whose spacing is lost
        # to FP32 offset rounding; retain exhaustive behavior for those rows.
        if (scale >= 1e-15) & (scale <= 1e15) & (tl.abs(offset) <= scale * 1048576.0):
            assigned, rates, ambiguous = _dquant_affine_assign(values, scale, offset, rate_ptr, LEVELS, BLOCK_WIDTH)
            indices = tl.arange(0, BLOCK_WIDTH)
            total = tl.sum(tl.where(indices < WIDTH, rates, 0.0), axis=0)
            has_midpoint = tl.sum(tl.where(indices < WIDTH, ambiguous.to(tl.int32), 0), axis=0) > 0
            if (total > TARGET_RATE_BITS) | has_midpoint:
                assigned = _dquant_drift_reference(
                    values, rate_ptr, scale, offset, WIDTH, LEVELS, BLOCK_WIDTH, TARGET_RATE_BITS, DRIFT_STEPS
                )
        else:
            assigned = _dquant_drift_reference(
                values, rate_ptr, scale, offset, WIDTH, LEVELS, BLOCK_WIDTH, TARGET_RATE_BITS, DRIFT_STEPS
            )
    else:
        assigned = _dquant_drift_reference(
            values, rate_ptr, scale, offset, WIDTH, LEVELS, BLOCK_WIDTH, TARGET_RATE_BITS, DRIFT_STEPS
        )
    return assigned


@triton.jit
def _dquant_quantize_kernel(
    input_ptr,
    rate_ptr,
    symbol_ptr,
    scale_ptr,
    offset_ptr,
    WIDTH: tl.constexpr,
    LEVELS: tl.constexpr,
    BLOCK_WIDTH: tl.constexpr,
    TARGET_RATE_BITS: tl.constexpr,
    DRIFT_STEPS: tl.constexpr,
    FAST_AFFINE: tl.constexpr = False,
):
    stream = tl.program_id(0)
    indices = tl.arange(0, BLOCK_WIDTH)
    values = tl.load(input_ptr + stream * WIDTH + indices, mask=indices < WIDTH, other=0).to(tl.float32)
    minimum = tl.min(tl.where(indices < WIDTH, values, float("inf")), axis=0)
    maximum = tl.max(tl.where(indices < WIDTH, values, float("-inf")), axis=0)
    scale = (maximum - minimum) / (LEVELS - 1)
    offset = minimum
    assigned = _dquant_drift(
        values, rate_ptr, scale, offset, WIDTH, LEVELS, BLOCK_WIDTH, TARGET_RATE_BITS, DRIFT_STEPS, FAST_AFFINE
    )
    assigned_f = assigned.to(tl.float32)
    valid = indices < WIDTH
    count = tl.sum(valid.to(tl.float32), axis=0)
    mean_symbol = tl.sum(tl.where(valid, assigned_f, 0.0), axis=0) / count
    mean_value = tl.sum(tl.where(valid, values, 0.0), axis=0) / count
    centered_symbol = assigned_f - mean_symbol
    variance = tl.sum(tl.where(valid, centered_symbol * centered_symbol, 0.0), axis=0)
    covariance = tl.sum(tl.where(valid, centered_symbol * (values - mean_value), 0.0), axis=0)
    refit_scale = tl.maximum(0.0, covariance / tl.maximum(variance, 1e-20))
    refit_offset = mean_value - refit_scale * mean_symbol
    scale = tl.where(variance > 0, refit_scale, scale).to(tl.float16).to(tl.float32)
    offset = tl.where(variance > 0, refit_offset, offset).to(tl.float16).to(tl.float32)

    tl.store(scale_ptr + stream, scale)
    tl.store(offset_ptr + stream, offset)


@triton.jit
def _dquant_finalize_kernel(
    input_ptr,
    rate_ptr,
    symbol_ptr,
    scale_ptr,
    offset_ptr,
    WIDTH: tl.constexpr,
    LEVELS: tl.constexpr,
    BLOCK_WIDTH: tl.constexpr,
    TARGET_RATE_BITS: tl.constexpr,
    DRIFT_STEPS: tl.constexpr,
    FAST_AFFINE: tl.constexpr = False,
):
    stream = tl.program_id(0)
    indices = tl.arange(0, BLOCK_WIDTH)
    values = tl.load(input_ptr + stream * WIDTH + indices, mask=indices < WIDTH, other=0).to(tl.float32)
    # Reload the stored FP16 metadata: the compiler may fold an in-register
    # FP32 -> FP16 -> FP32 cast, producing symbols inconsistent with the record.
    scale = tl.load(scale_ptr + stream).to(tl.float32)
    offset = tl.load(offset_ptr + stream).to(tl.float32)
    assigned = _dquant_drift(
        values, rate_ptr, scale, offset, WIDTH, LEVELS, BLOCK_WIDTH, TARGET_RATE_BITS, DRIFT_STEPS, FAST_AFFINE
    )
    tl.store(symbol_ptr + stream * WIDTH + indices, assigned.to(tl.int16), mask=indices < WIDTH)


@triton.jit
def _dquant_finalize_encode_kernel(
    input_ptr,
    rate_ptr,
    scale_ptr,
    offset_ptr,
    frequency_ptr,
    cumulative_ptr,
    symbol_ptr,
    output_ptr,
    status_ptr,
    WIDTH: tl.constexpr,
    LEVELS: tl.constexpr,
    BLOCK_WIDTH: tl.constexpr,
    TARGET_RATE_BITS: tl.constexpr,
    DRIFT_STEPS: tl.constexpr,
    CONTAINER_BYTES: tl.constexpr,
    BYTE_BLOCK: tl.constexpr,
    SCALE_BITS: tl.constexpr,
    LOWER_BOUND: tl.constexpr,
    FAST_AFFINE: tl.constexpr = False,
):
    """Refit-aware final drift and rANS encoding in one stream program."""
    stream = tl.program_id(0)
    indices = tl.arange(0, BLOCK_WIDTH)
    values = tl.load(input_ptr + stream * WIDTH + indices, mask=indices < WIDTH, other=0).to(tl.float32)
    scale = tl.load(scale_ptr + stream).to(tl.float32)
    offset = tl.load(offset_ptr + stream).to(tl.float32)
    assigned = _dquant_drift(
        values,
        rate_ptr,
        scale,
        offset,
        WIDTH,
        LEVELS,
        BLOCK_WIDTH,
        TARGET_RATE_BITS,
        DRIFT_STEPS,
        FAST_AFFINE,
    )
    tl.store(symbol_ptr + stream * WIDTH + indices, assigned.to(tl.int16), mask=indices < WIDTH)

    output_row = output_ptr + stream * CONTAINER_BYTES
    byte_offsets = tl.arange(0, BYTE_BLOCK)
    tl.store(output_row + byte_offsets, 0, mask=byte_offsets < CONTAINER_BYTES)
    state = tl.full((), LOWER_BOUND, dtype=tl.uint32)
    cursor = tl.full((), CONTAINER_BYTES, dtype=tl.int32)
    invalid_symbol = tl.full((), 0, dtype=tl.int32)
    renormalization_limit = (LOWER_BOUND >> SCALE_BITS) << 8
    for index in range(int(WIDTH)):
        symbol_index = tl.full((1,), WIDTH - index - 1, dtype=tl.int32)
        symbol = tl.sum(tl.gather(assigned.to(tl.float32), symbol_index, axis=0), axis=0).to(tl.int32)
        valid_symbol = (symbol >= 0) & (symbol < LEVELS)
        safe_symbol = tl.minimum(LEVELS - 1, tl.maximum(0, symbol))
        frequency = tl.load(frequency_ptr + safe_symbol).to(tl.uint32)
        cumulative = tl.load(cumulative_ptr + safe_symbol).to(tl.uint32)
        invalid_symbol |= (~valid_symbol | (frequency == 0)).to(tl.int32)
        frequency = tl.maximum(frequency, 1)
        maximum_state = renormalization_limit * frequency
        while state >= maximum_state:
            cursor -= 1
            tl.store(output_row + cursor, (state & 0xFF).to(tl.uint8), mask=cursor >= 4)
            state >>= 8
        state = ((state // frequency) << SCALE_BITS) + (state % frequency) + cumulative

    for byte_index in tl.static_range(4):
        tl.store(output_row + byte_index, ((state >> (8 * byte_index)) & 0xFF).to(tl.uint8))
    emitted_bytes = CONTAINER_BYTES - cursor
    if cursor >= 4:
        packed = tl.load(output_row + cursor + byte_offsets, mask=byte_offsets < emitted_bytes, other=0)
        tl.store(output_row + 4 + byte_offsets, packed, mask=byte_offsets < CONTAINER_BYTES - 4)
    tl.store(status_ptr + stream, (cursor < 4).to(tl.int32) | (invalid_symbol * 2))


@triton.jit
def _dquant_dequantize_kernel(
    symbol_ptr,
    scale_ptr,
    offset_ptr,
    output_ptr,
    WIDTH: tl.constexpr,
    BLOCK_WIDTH: tl.constexpr,
):
    stream = tl.program_id(0)
    indices = tl.program_id(1) * BLOCK_WIDTH + tl.arange(0, BLOCK_WIDTH)
    symbols = tl.load(symbol_ptr + stream * WIDTH + indices, mask=indices < WIDTH, other=0).to(tl.float32)
    scale = tl.load(scale_ptr + stream).to(tl.float32)
    offset = tl.load(offset_ptr + stream).to(tl.float32)
    values = offset + scale * symbols
    tl.store(output_ptr + stream * WIDTH + indices, values, mask=indices < WIDTH)


@triton.jit(do_not_specialize=["START_TOKEN"])
def _paged_rans_dequantize_kernel(
    storage_ptr,
    table_ptr,
    symbol_lut_ptr,
    frequency_lut_ptr,
    cumulative_lut_ptr,
    scales_ptr,
    offsets_ptr,
    output_ptr,
    BLOCK_SIZE: tl.constexpr,
    ROW_BYTES: tl.constexpr,
    METADATA_STRIDE: tl.constexpr,
    CONTAINER_BYTES: tl.constexpr,
    SYMBOL_COUNT: tl.constexpr,
    TABLE_SIZE: tl.constexpr,
    SCALE_BITS: tl.constexpr,
    START_TOKEN,
):
    output_token = tl.program_id(0)
    token = START_TOKEN + output_token
    block = tl.load(table_ptr + token // BLOCK_SIZE).to(tl.int64)
    slot_index = block * BLOCK_SIZE + token % BLOCK_SIZE
    payload = storage_ptr + slot_index * ROW_BYTES
    state = (
        tl.load(payload).to(tl.uint32)
        | (tl.load(payload + 1).to(tl.uint32) << 8)
        | (tl.load(payload + 2).to(tl.uint32) << 16)
        | (tl.load(payload + 3).to(tl.uint32) << 24)
    )
    scale = tl.load(scales_ptr + slot_index * METADATA_STRIDE).to(tl.float32)
    offset = tl.load(offsets_ptr + slot_index * METADATA_STRIDE).to(tl.float32)
    cursor = 4
    for index in range(int(SYMBOL_COUNT)):
        lut_slot = state & (TABLE_SIZE - 1)
        symbol = tl.load(symbol_lut_ptr + lut_slot).to(tl.float32)
        frequency = tl.load(frequency_lut_ptr + lut_slot).to(tl.uint32)
        cumulative = tl.load(cumulative_lut_ptr + lut_slot).to(tl.uint32)
        state = frequency * (state >> SCALE_BITS) + lut_slot - cumulative
        while state < (1 << 23) and cursor < CONTAINER_BYTES:
            byte = tl.load(payload + cursor, mask=cursor < CONTAINER_BYTES, other=0).to(tl.uint32)
            state = (state << 8) | byte
            cursor += 1
        # Keep affine rounding identical to the reference dequantization path.
        value = offset + scale * symbol
        tl.store(output_ptr + output_token * SYMBOL_COUNT + index, value)


@triton.jit(do_not_specialize=["START_TOKEN"])
def _paged_rans_dequantize_buffered_kernel(
    storage_ptr,
    table_ptr,
    symbol_lut_ptr,
    frequency_lut_ptr,
    cumulative_lut_ptr,
    scales_ptr,
    offsets_ptr,
    output_ptr,
    BLOCK_SIZE: tl.constexpr,
    ROW_BYTES: tl.constexpr,
    METADATA_STRIDE: tl.constexpr,
    CONTAINER_BYTES: tl.constexpr,
    SYMBOL_COUNT: tl.constexpr,
    TABLE_SIZE: tl.constexpr,
    SCALE_BITS: tl.constexpr,
    OUTPUT_CHUNK: tl.constexpr,
    START_TOKEN,
):
    output_token = tl.program_id(0)
    token = START_TOKEN + output_token
    block = tl.load(table_ptr + token // BLOCK_SIZE).to(tl.int64)
    slot = block * BLOCK_SIZE + token % BLOCK_SIZE
    payload = storage_ptr + slot * ROW_BYTES
    state = (
        tl.load(payload).to(tl.uint32)
        | (tl.load(payload + 1).to(tl.uint32) << 8)
        | (tl.load(payload + 2).to(tl.uint32) << 16)
        | (tl.load(payload + 3).to(tl.uint32) << 24)
    )
    scale = tl.load(scales_ptr + slot * METADATA_STRIDE).to(tl.float32)
    offset = tl.load(offsets_ptr + slot * METADATA_STRIDE).to(tl.float32)
    cursor = 4
    lanes = tl.arange(0, OUTPUT_CHUNK)
    # Keep the entropy recurrence serial, but coalesce output stores in UB.
    for chunk in range(triton.cdiv(SYMBOL_COUNT, OUTPUT_CHUNK)):
        values = tl.full((OUTPUT_CHUNK,), 0, tl.float32)
        for item in range(int(OUTPUT_CHUNK)):
            index = chunk * OUTPUT_CHUNK + item
            if index < SYMBOL_COUNT:
                lut_slot = state & (TABLE_SIZE - 1)
                symbol = tl.load(symbol_lut_ptr + lut_slot).to(tl.float32)
                frequency = tl.load(frequency_lut_ptr + lut_slot).to(tl.uint32)
                cumulative = tl.load(cumulative_lut_ptr + lut_slot).to(tl.uint32)
                state = frequency * (state >> SCALE_BITS) + lut_slot - cumulative
                for _renorm in range(3):
                    if state < (1 << 23):
                        byte = tl.load(payload + cursor, mask=cursor < CONTAINER_BYTES, other=0).to(tl.uint32)
                        state = (state << 8) | byte
                        cursor += 1
                values = tl.where(lanes == item, offset + scale * symbol, values)
        indices = chunk * OUTPUT_CHUNK + lanes
        tl.store(output_ptr + output_token * SYMBOL_COUNT + indices, values, mask=indices < SYMBOL_COUNT)


def paged_rans_dequantize_npu(
    storage: torch.Tensor,
    block_table: torch.Tensor,
    length: int,
    block_size: int,
    container_bytes: int,
    stream_length: int,
    scales: torch.Tensor,
    offsets: torch.Tensor,
    decode_luts: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    output_dtype: torch.dtype,
    *,
    start_token: int = 0,
    output_chunk: int = 1,
) -> torch.Tensor:
    """Decode scheduler-owned records without payload gathers or host syncs.

    Block IDs must already be validated by the scheduler. This internal read
    path deliberately avoids the public slot API's device-to-host checks.
    output_chunk > 1 opts into buffered stores; scalar decoding remains the
    default until representative end-to-end measurements justify changing it.
    """
    if storage.dtype != torch.uint8 or storage.ndim != 2 or not storage.is_contiguous():
        raise ValueError("Paged D-Quant storage must be contiguous uint8 rows")
    if block_size <= 0 or length < 0 or start_token < 0 or stream_length <= 0:
        raise ValueError("Invalid paged D-Quant shape")
    if block_table.ndim != 1 or block_table.dtype not in (torch.int32, torch.int64):
        raise ValueError("Block table must be a one-dimensional integer tensor")
    if not block_table.is_contiguous() or block_table.numel() * block_size < start_token + length:
        raise ValueError("Block table does not cover the D-Quant sequence")
    if not output_dtype.is_floating_point:
        raise ValueError("Paged D-Quant output must use a floating-point dtype")
    if storage.shape[1] < ((container_bytes + 1) & ~1) + 4 or container_bytes < 4:
        raise ValueError("D-Quant row cannot hold its payload and metadata")
    if any(tensor.device != storage.device for tensor in (block_table, scales, offsets, *decode_luts)):
        raise ValueError("Paged D-Quant tensors must be on the same device")
    table_size = decode_luts[0].numel()
    if table_size <= 0 or table_size & (table_size - 1):
        raise ValueError("Invalid D-Quant decoder table size")
    if any(tensor.ndim != 1 or not tensor.is_contiguous() or tensor.numel() != table_size for tensor in decode_luts):
        raise ValueError("Invalid D-Quant decoder tables")
    if (
        scales.ndim != 1
        or offsets.ndim != 1
        or scales.shape != offsets.shape
        or scales.numel() != storage.shape[0]
        or scales.stride() != offsets.stride()
    ):
        raise ValueError("Invalid D-Quant metadata views")
    if output_chunk not in (1, 16, 32, 64):
        raise ValueError("D-Quant output_chunk must be one of 1, 16, 32, 64")
    output = torch.empty((length, stream_length), dtype=output_dtype, device=storage.device)
    if length:
        if output_chunk > 1:
            _paged_rans_dequantize_buffered_kernel[(length,)](
                storage,
                block_table,
                *decode_luts,
                scales,
                offsets,
                output,
                BLOCK_SIZE=block_size,
                ROW_BYTES=storage.shape[1],
                METADATA_STRIDE=scales.stride(0),
                CONTAINER_BYTES=container_bytes,
                SYMBOL_COUNT=stream_length,
                TABLE_SIZE=table_size,
                SCALE_BITS=table_size.bit_length() - 1,
                OUTPUT_CHUNK=output_chunk,
                START_TOKEN=start_token,
                num_warps=4,
                enable_fp_fusion=False,
            )
            return output
        _paged_rans_dequantize_kernel[(length,)](
            storage,
            block_table,
            *decode_luts,
            scales,
            offsets,
            output,
            BLOCK_SIZE=block_size,
            ROW_BYTES=storage.shape[1],
            METADATA_STRIDE=scales.stride(0),
            CONTAINER_BYTES=container_bytes,
            SYMBOL_COUNT=stream_length,
            TABLE_SIZE=table_size,
            SCALE_BITS=table_size.bit_length() - 1,
            START_TOKEN=start_token,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return output


def rans_decode_npu(
    payloads: torch.Tensor,
    symbols_per_stream: int,
    symbol_lut: torch.Tensor,
    frequency_lut: torch.Tensor,
    cumulative_lut: torch.Tensor,
) -> torch.Tensor:
    """Decode fixed-size rANS containers to symbol indices on an Ascend NPU.

    The three lookup tensors expand the frequency table to one entry per
    rANS state slot. Their values must be resident on the same NPU as payloads.
    """
    if payloads.ndim != 2 or payloads.dtype != torch.uint8:
        raise ValueError("payloads must be a 2D uint8 tensor")
    if not payloads.is_contiguous():
        payloads = payloads.contiguous()
    if payloads.shape[1] < 4:
        raise ValueError("each payload must contain the 4-byte rANS state")
    if symbols_per_stream <= 0:
        raise ValueError("symbols_per_stream must be positive")
    table_size = symbol_lut.numel()
    if table_size <= 0 or table_size & (table_size - 1):
        raise ValueError("rANS lookup table size must be a positive power of two")
    if frequency_lut.numel() != table_size or cumulative_lut.numel() != table_size:
        raise ValueError("all rANS lookup tensors must have the same length")
    if any(tensor.ndim != 1 or not tensor.is_contiguous() for tensor in (symbol_lut, frequency_lut, cumulative_lut)):
        raise ValueError("rANS lookup tensors must be contiguous one-dimensional tensors")
    if any(tensor.device != payloads.device for tensor in (symbol_lut, frequency_lut, cumulative_lut)):
        raise ValueError("payload and lookup tensors must be on the same device")

    output = torch.empty((payloads.shape[0], symbols_per_stream), dtype=torch.int16, device=payloads.device)
    if payloads.shape[0] == 0:
        return output
    _rans_decode_kernel[(payloads.shape[0],)](
        payloads,
        symbol_lut,
        frequency_lut,
        cumulative_lut,
        output,
        CONTAINER_BYTES=payloads.shape[1],
        SYMBOL_COUNT=symbols_per_stream,
        TABLE_SIZE=table_size,
        SCALE_BITS=table_size.bit_length() - 1,
        LOWER_BOUND=1 << 23,
        num_warps=4,
    )
    return output


def rans_encode_npu(
    symbols: torch.Tensor,
    container_bytes: int,
    frequency_by_symbol: torch.Tensor,
    cumulative_by_symbol: torch.Tensor,
    table_total: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode symbol streams to fixed-size rANS containers on an Ascend NPU.

    Returns the byte containers and one overflow flag per stream. The caller
    must reject streams whose flag is non-zero because their rANS state and
    renormalization bytes do not fit the requested container.
    """
    if symbols.ndim != 2 or symbols.dtype not in (torch.int16, torch.int32, torch.int64):
        raise ValueError("symbols must be a 2D integer tensor")
    if container_bytes < 4:
        raise ValueError("container must fit the 4-byte rANS state")
    if frequency_by_symbol.ndim != 1 or cumulative_by_symbol.ndim != 1:
        raise ValueError("rANS frequency and cumulative tables must be one-dimensional")
    alphabet_size = frequency_by_symbol.numel()
    if alphabet_size <= 0 or cumulative_by_symbol.numel() != alphabet_size:
        raise ValueError("rANS frequency and cumulative tables must have the same non-zero length")
    if table_total <= 0 or table_total & (table_total - 1):
        raise ValueError("rANS table total must be a positive power of two")
    if any(not tensor.is_contiguous() for tensor in (symbols, frequency_by_symbol, cumulative_by_symbol)):
        raise ValueError("rANS inputs must be contiguous")
    if any(tensor.device != symbols.device for tensor in (frequency_by_symbol, cumulative_by_symbol)):
        raise ValueError("rANS inputs must be on the same device")

    payloads = torch.empty((symbols.shape[0], container_bytes), dtype=torch.uint8, device=symbols.device)
    status = torch.empty((symbols.shape[0],), dtype=torch.int32, device=symbols.device)
    if symbols.shape[0] == 0:
        return payloads, status
    _rans_encode_kernel[(symbols.shape[0],)](
        symbols,
        frequency_by_symbol,
        cumulative_by_symbol,
        payloads,
        status,
        CONTAINER_BYTES=container_bytes,
        BYTE_BLOCK=triton.next_power_of_2(container_bytes),
        SYMBOL_COUNT=symbols.shape[1],
        ALPHABET_SIZE=alphabet_size,
        SCALE_BITS=table_total.bit_length() - 1,
        LOWER_BOUND=1 << 23,
        num_warps=4,
    )
    return payloads, status


def dquant_quantize_npu(
    values: torch.Tensor,
    symbol_rates: torch.Tensor,
    target_rate_bits: float,
    drift_steps: int = 12,
    *,
    minimum_symbol_rate: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply per-stream affine quantization and D-Quant drift on an Ascend NPU."""
    if values.ndim != 2 or not values.is_floating_point():
        raise ValueError("values must be a 2D floating-point tensor")
    if not values.is_contiguous():
        values = values.contiguous()
    if values.shape[1] <= 0 or not symbol_rates.ndim == 1 or symbol_rates.numel() <= 1:
        raise ValueError("values need a non-empty width and symbol_rates must be a 1D alphabet")
    if not symbol_rates.is_contiguous() or symbol_rates.device != values.device:
        raise ValueError("symbol_rates must be contiguous and on the same device as values")
    if not math.isfinite(target_rate_bits) or target_rate_bits <= 0:
        raise ValueError("target_rate_bits must be finite and positive")
    if drift_steps <= 0:
        raise ValueError("drift_steps must be positive")
    levels = symbol_rates.numel()
    minimum_rate = float(symbol_rates.min().item()) if minimum_symbol_rate is None else minimum_symbol_rate
    if not math.isfinite(minimum_rate) or minimum_rate < 0:
        raise ValueError("minimum symbol rate must be finite and nonnegative")
    if target_rate_bits < values.shape[1] * minimum_rate:
        raise ValueError("target rate is below the minimum symbol rate for this stream width")
    output_symbols = torch.empty(values.shape, dtype=torch.int16, device=values.device)
    scales = torch.empty((values.shape[0],), dtype=torch.float16, device=values.device)
    offsets = torch.empty((values.shape[0],), dtype=torch.float16, device=values.device)
    if values.shape[0] == 0:
        return output_symbols, scales, offsets
    _dquant_quantize_kernel[(values.shape[0],)](
        values,
        symbol_rates,
        output_symbols,
        scales,
        offsets,
        WIDTH=values.shape[1],
        LEVELS=levels,
        BLOCK_WIDTH=triton.next_power_of_2(values.shape[1]),
        TARGET_RATE_BITS=target_rate_bits,
        DRIFT_STEPS=drift_steps,
        FAST_AFFINE=values.dtype in (torch.float32, torch.bfloat16) and levels == 128 and values.shape[1] >= 128,
        num_warps=4,
    )
    _dquant_finalize_kernel[(values.shape[0],)](
        values,
        symbol_rates,
        output_symbols,
        scales,
        offsets,
        WIDTH=values.shape[1],
        LEVELS=levels,
        BLOCK_WIDTH=triton.next_power_of_2(values.shape[1]),
        TARGET_RATE_BITS=target_rate_bits,
        DRIFT_STEPS=drift_steps,
        FAST_AFFINE=values.dtype in (torch.float32, torch.bfloat16) and levels == 128 and values.shape[1] >= 128,
        num_warps=4,
    )
    return output_symbols, scales, offsets


def dquant_quantize_encode_npu(
    values: torch.Tensor,
    symbol_rates: torch.Tensor,
    container_bytes: int,
    frequency_by_symbol: torch.Tensor,
    cumulative_by_symbol: torch.Tensor,
    table_total: int,
    drift_steps: int = 12,
    *,
    minimum_symbol_rate: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize, refit, final-drift and rANS-encode each stream on one launch pair.

    The initial min/max assignment remains separate because the refit metadata
    must be rounded to FP16 before the final drift. The final drift and rANS
    state machine share one stream program and never materialize a second
    device-side symbol-to-payload handoff.
    """
    if values.ndim != 2 or not values.is_floating_point():
        raise ValueError("values must be a 2D floating-point tensor")
    if not values.is_contiguous():
        values = values.contiguous()
    if values.shape[1] <= 0 or symbol_rates.ndim != 1 or symbol_rates.numel() <= 1:
        raise ValueError("values need a non-empty width and symbol_rates must be a 1D alphabet")
    if not symbol_rates.is_contiguous() or symbol_rates.device != values.device:
        raise ValueError("symbol_rates must be contiguous and on the same device as values")
    if frequency_by_symbol.ndim != 1 or cumulative_by_symbol.ndim != 1:
        raise ValueError("rANS frequency and cumulative tables must be one-dimensional")
    if frequency_by_symbol.numel() != symbol_rates.numel() or cumulative_by_symbol.numel() != symbol_rates.numel():
        raise ValueError("rANS tables must match the quantization alphabet")
    if any(t.device != values.device or not t.is_contiguous() for t in (frequency_by_symbol, cumulative_by_symbol)):
        raise ValueError("rANS tables must be contiguous on the values device")
    if container_bytes < 4 or table_total <= 0 or table_total & (table_total - 1):
        raise ValueError("container and rANS table parameters are invalid")
    if drift_steps <= 0 or container_bytes * 8 - 32 <= 0:
        raise ValueError("drift_steps and target rate must be valid")
    target_rate_bits = container_bytes * 8 - 32
    levels = symbol_rates.numel()
    minimum_rate = float(symbol_rates.min().item()) if minimum_symbol_rate is None else minimum_symbol_rate
    if not math.isfinite(minimum_rate) or minimum_rate < 0 or target_rate_bits < values.shape[1] * minimum_rate:
        raise ValueError("target rate is below the minimum symbol rate for this stream width")
    rows, width = values.shape
    symbols = torch.empty((rows, width), dtype=torch.int16, device=values.device)
    scales = torch.empty((rows,), dtype=torch.float16, device=values.device)
    offsets = torch.empty((rows,), dtype=torch.float16, device=values.device)
    payloads = torch.empty((rows, container_bytes), dtype=torch.uint8, device=values.device)
    status = torch.empty((rows,), dtype=torch.int32, device=values.device)
    if rows == 0:
        return symbols, scales, offsets, payloads, status
    block_width = triton.next_power_of_2(width)
    fast_affine = values.dtype in (torch.float32, torch.bfloat16) and levels == 128 and width >= 128
    common = dict(
        WIDTH=width,
        LEVELS=levels,
        BLOCK_WIDTH=block_width,
        TARGET_RATE_BITS=target_rate_bits,
        DRIFT_STEPS=drift_steps,
        FAST_AFFINE=fast_affine,
        num_warps=4,
    )
    _dquant_quantize_kernel[(rows,)](
        values,
        symbol_rates,
        symbols,
        scales,
        offsets,
        **common,
    )
    _dquant_finalize_encode_kernel[(rows,)](
        values,
        symbol_rates,
        scales,
        offsets,
        frequency_by_symbol,
        cumulative_by_symbol,
        symbols,
        payloads,
        status,
        **common,
        CONTAINER_BYTES=container_bytes,
        BYTE_BLOCK=triton.next_power_of_2(container_bytes),
        SCALE_BITS=table_total.bit_length() - 1,
        LOWER_BOUND=1 << 23,
    )
    return symbols, scales, offsets, payloads, status


def dquant_dequantize_npu(
    symbols: torch.Tensor,
    scales: torch.Tensor,
    offsets: torch.Tensor,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Restore affine-quantized streams from symbols and per-stream metadata."""
    if symbols.ndim != 2 or symbols.dtype not in (torch.int16, torch.int32, torch.int64):
        raise ValueError("symbols must be a 2D integer tensor")
    if (
        scales.ndim != 1
        or offsets.ndim != 1
        or scales.numel() != symbols.shape[0]
        or offsets.numel() != symbols.shape[0]
    ):
        raise ValueError("scales and offsets must have one entry per symbol stream")
    if any(not tensor.is_contiguous() for tensor in (symbols, scales, offsets)):
        raise ValueError("D-Quant inputs must be contiguous")
    if any(tensor.device != symbols.device for tensor in (scales, offsets)):
        raise ValueError("D-Quant inputs must be on the same device")
    if not scales.is_floating_point() or not offsets.is_floating_point() or not output_dtype.is_floating_point:
        raise ValueError("scales, offsets, and output must use floating-point dtypes")

    output = torch.empty(symbols.shape, dtype=output_dtype, device=symbols.device)
    if symbols.shape[0] == 0 or symbols.shape[1] == 0:
        return output
    block_width = min(1024, triton.next_power_of_2(symbols.shape[1]))
    _dquant_dequantize_kernel[(symbols.shape[0], triton.cdiv(symbols.shape[1], block_width))](
        symbols,
        scales,
        offsets,
        output,
        WIDTH=symbols.shape[1],
        BLOCK_WIDTH=block_width,
        num_warps=4,
    )
    return output
