# SPDX-License-Identifier: Apache-2.0
"""Restore striped D-Quant key means without dense indexing temporaries."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["START_TOKEN"])
def _restore_key_means_kernel(
    input_ptr,
    storage_ptr,
    table_ptr,
    output_ptr,
    START_TOKEN,
    WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    WINDOW: tl.constexpr,
    ROW_BYTES: tl.constexpr,
    MEAN_OFFSET: tl.constexpr,
    STRIPE_BYTES: tl.constexpr,
    MEAN_BYTES: tl.constexpr,
    MEAN_TYPE: tl.constexpr,
    BLOCK_WIDTH: tl.constexpr,
):
    row = tl.program_id(0)
    dimensions = tl.program_id(1) * BLOCK_WIDTH + tl.arange(0, BLOCK_WIDTH)
    token = START_TOKEN + row
    window_start = token // WINDOW * WINDOW
    bits = tl.full((BLOCK_WIDTH,), 0, tl.uint32)
    for byte in tl.static_range(MEAN_BYTES):
        byte_index = dimensions * MEAN_BYTES + byte
        record_token = window_start + byte_index // STRIPE_BYTES
        block = tl.load(table_ptr + record_token // BLOCK_SIZE, mask=dimensions < WIDTH, other=0).to(tl.int64)
        slot = block * BLOCK_SIZE + record_token % BLOCK_SIZE
        offset = slot * ROW_BYTES + MEAN_OFFSET + byte_index % STRIPE_BYTES
        value = tl.load(storage_ptr + offset, mask=dimensions < WIDTH, other=0).to(tl.uint32)
        bits = bits | (value << (8 * byte))
    if MEAN_BYTES == 2:
        mean = bits.to(tl.uint16).to(MEAN_TYPE, bitcast=True).to(tl.float32)
    else:
        mean = bits.to(MEAN_TYPE, bitcast=True)
    values = tl.load(input_ptr + row * WIDTH + dimensions, mask=dimensions < WIDTH, other=0).to(tl.float32)
    tl.store(output_ptr + row * WIDTH + dimensions, values + mean, mask=dimensions < WIDTH)


def restore_key_means_npu(
    values,
    storage,
    block_table,
    *,
    start_token,
    block_size,
    window,
    mean_offset,
    stripe_bytes,
    mean_dtype,
):
    """Add centered-window means with the reference FP32 addition/rounding."""
    mean_types = {torch.float16: tl.float16, torch.bfloat16: tl.bfloat16, torch.float32: tl.float32}
    if mean_dtype not in mean_types or values.dtype not in mean_types:
        raise ValueError("D-Quant means and decoded keys must use FP16, BF16 or FP32")
    if values.ndim != 3 or not values.is_contiguous():
        raise ValueError("Decoded keys must be contiguous [tokens, heads, head_size]")
    if storage.ndim != 2 or storage.dtype != torch.uint8 or not storage.is_contiguous():
        raise ValueError("D-Quant records must be contiguous uint8 rows")
    if block_size <= 0 or window <= 0 or start_token < 0 or stripe_bytes <= 0 or mean_offset < 0:
        raise ValueError("Invalid D-Quant mean layout")
    width = values.shape[1] * values.shape[2]
    if width <= 0:
        raise ValueError("Decoded keys must have positive head dimensions")
    mean_bytes = torch.empty((), dtype=mean_dtype).element_size()
    if width * mean_bytes > window * stripe_bytes or mean_offset + stripe_bytes > storage.shape[1]:
        raise ValueError("Striped records cannot hold the D-Quant means")
    required_tokens = (start_token + values.shape[0] + window - 1) // window * window if values.shape[0] else 0
    if (
        block_table.ndim != 1
        or not block_table.is_contiguous()
        or block_table.dtype not in (torch.int32, torch.int64)
        or block_table.numel() * block_size < required_tokens
    ):
        raise ValueError("Block table does not cover the decoded keys")
    if values.device != storage.device or values.device != block_table.device:
        raise ValueError("D-Quant mean tensors must share a device")
    output = torch.empty_like(values)
    if values.numel():
        _restore_key_means_kernel[(values.shape[0], triton.cdiv(width, 256))](
            values,
            storage,
            block_table,
            output,
            start_token,
            WIDTH=width,
            BLOCK_SIZE=block_size,
            WINDOW=window,
            ROW_BYTES=storage.shape[1],
            MEAN_OFFSET=mean_offset,
            STRIPE_BYTES=stripe_bytes,
            MEAN_BYTES=mean_bytes,
            MEAN_TYPE=mean_types[mean_dtype],
            BLOCK_WIDTH=256,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return output
