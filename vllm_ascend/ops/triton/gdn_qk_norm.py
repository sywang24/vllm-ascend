# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from vllm.third_party.flash_linear_attention.ops.l2norm import USE_DEFAULT_FLA_NORM, l2norm_fwd
from vllm.triton_utils import tl, triton

QK_HEAD_WIDTH = 128
QK_ROW_BLOCK = 32


@triton.jit
def _qk_norm_kernel(
    query_ptr,
    key_ptr,
    out_query_ptr,
    out_key_ptr,
    rows,
    WIDTH: tl.constexpr,
    BLOCK_WIDTH: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)[:, None]
    column = tl.arange(0, BLOCK_WIDTH)[None, :]
    valid_row = row < rows
    mask = valid_row & (column < WIDTH)
    values = tl.load(query_ptr + row * WIDTH + column, mask=mask, other=0).to(tl.float32)
    squared = tl.broadcast_to(values * values, [ROW_BLOCK, BLOCK_WIDTH])
    total = tl.sum(tl.where(valid_row, squared, 0), axis=1)[:, None]
    normalized = values * tl.rsqrt(total + 1e-6)
    tl.store(out_query_ptr + row * WIDTH + column, normalized, mask=mask)
    keys = tl.load(key_ptr + row * WIDTH + column, mask=mask, other=0).to(tl.float32)
    squared_keys = tl.broadcast_to(keys * keys, [ROW_BLOCK, BLOCK_WIDTH])
    total_keys = tl.sum(tl.where(valid_row, squared_keys, 0), axis=1)[:, None]
    normalized_keys = keys * tl.rsqrt(total_keys + 1e-6)
    tl.store(out_key_ptr + row * WIDTH + column, normalized_keys, mask=mask)


def l2norm_qk_fwd(query, key):
    """Normalize contiguous decode Q/K together with the FLA arithmetic."""
    if (
        USE_DEFAULT_FLA_NORM
        or query.shape != key.shape
        or query.dtype != key.dtype
        or query.device != key.device
        or not query.is_contiguous()
        or not key.is_contiguous()
        or query.dtype not in (torch.bfloat16, torch.float32)
        or query.shape[-1] != QK_HEAD_WIDTH
        or query.numel() == 0
    ):
        return l2norm_fwd(query), l2norm_fwd(key)
    out_query = torch.empty_like(query)
    out_key = torch.empty_like(key)
    rows = query.numel() // query.shape[-1]
    _qk_norm_kernel[(triton.cdiv(rows, QK_ROW_BLOCK),)](
        query,
        key,
        out_query,
        out_key,
        rows,
        WIDTH=QK_HEAD_WIDTH,
        BLOCK_WIDTH=QK_HEAD_WIDTH,
        ROW_BLOCK=QK_ROW_BLOCK,
        num_warps=4,
    )
    return out_query, out_key
