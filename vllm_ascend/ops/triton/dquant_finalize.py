# SPDX-License-Identifier: Apache-2.0
"""Finalize rotated D-Quant partials and the raw tail without intermediate copies."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["CHUNKS", "TAIL_TOKENS"])
def _finalize_dquant_attention_kernel(
    parts_ptr,
    lses_ptr,
    query_ptr,
    tail_key_ptr,
    tail_value_ptr,
    current_key_ptr,
    current_value_ptr,
    output_ptr,
    CHUNKS,
    TAIL_TOKENS,
    SCALE,
    HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    WIDTH: tl.constexpr,
    LOG2_WIDTH: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_T: tl.constexpr,
    HAS_CURRENT: tl.constexpr,
):
    head = tl.program_id(0)
    kv_head = head // (HEADS // KV_HEADS)
    dims = tl.arange(0, WIDTH)
    chunk_offsets = tl.arange(0, BLOCK_C)
    maximum = tl.full((), float("-inf"), tl.float32)
    denominator = tl.full((), 0.0, tl.float32)
    numerator = tl.zeros((WIDTH,), tl.float32)
    for start in range(tl.cdiv(CHUNKS, BLOCK_C)):
        chunks = start * BLOCK_C + chunk_offsets
        lse = tl.load(lses_ptr + chunks * HEADS + head, mask=chunks < CHUNKS, other=float("-inf"))
        next_maximum = tl.maximum(maximum, tl.max(lse, axis=0))
        alpha = tl.exp(maximum - next_maximum)
        weights = tl.exp(lse - next_maximum)
        partials = tl.load(
            parts_ptr + (chunks[:, None] * HEADS + head) * WIDTH + dims[None, :],
            mask=chunks[:, None] < CHUNKS,
            other=0.0,
        )
        numerator = numerator * alpha + tl.sum(weights[:, None] * partials, axis=0)
        denominator = denominator * alpha + tl.sum(weights, axis=0)
        maximum = next_maximum
    rotated = numerator / tl.maximum(denominator, 1e-30)
    for stage in tl.static_range(LOG2_WIDTH):
        stride = 1 << stage
        partner = tl.gather(rotated, dims ^ stride, axis=0)
        rotated = tl.where((dims & stride) == 0, rotated + partner, partner - rotated)
    # Prefix mass is represented by its LSE and normalized original-basis output.
    numerator = rotated * (1.0 / tl.sqrt(float(WIDTH)))
    maximum = maximum + tl.log(denominator)
    denominator = tl.full((), 1.0, tl.float32)
    query = tl.load(query_ptr + head * WIDTH + dims).to(tl.float32)
    token_offsets = tl.arange(0, BLOCK_T)
    for start in range(tl.cdiv(TAIL_TOKENS, BLOCK_T)):
        tokens = start * BLOCK_T + token_offsets
        offsets = (tokens[:, None] * KV_HEADS + kv_head) * WIDTH + dims[None, :]
        keys = tl.load(tail_key_ptr + offsets, mask=tokens[:, None] < TAIL_TOKENS, other=0.0).to(tl.float32)
        scores = tl.sum(keys * query[None, :], axis=1) * SCALE
        scores = tl.where(tokens < TAIL_TOKENS, scores, float("-inf"))
        next_maximum = tl.maximum(maximum, tl.max(scores, axis=0))
        alpha = tl.exp(maximum - next_maximum)
        probabilities = tl.exp(scores - next_maximum)
        values = tl.load(tail_value_ptr + offsets, mask=tokens[:, None] < TAIL_TOKENS, other=0.0).to(tl.float32)
        numerator = numerator * alpha + tl.sum(probabilities[:, None] * values, axis=0)
        denominator = denominator * alpha + tl.sum(probabilities, axis=0)
        maximum = next_maximum
    if HAS_CURRENT:
        key = tl.load(current_key_ptr + kv_head * WIDTH + dims).to(tl.float32)
        value = tl.load(current_value_ptr + kv_head * WIDTH + dims).to(tl.float32)
        score = tl.sum(key * query, axis=0) * SCALE
        next_maximum = tl.maximum(maximum, score)
        alpha = tl.exp(maximum - next_maximum)
        weight = tl.exp(score - next_maximum)
        numerator = numerator * alpha + value * weight
        denominator = denominator * alpha + weight
    tl.store(output_ptr + head * WIDTH + dims, numerator / denominator)


@triton.jit(do_not_specialize=["MAX_CHUNKS", "CHUNK_TOKENS", "TAIL_METADATA_STRIDE"])
def _finalize_dquant_batch_attention_kernel(
    parts_ptr,
    lses_ptr,
    query_ptr,
    tail_key_ptr,
    tail_value_ptr,
    current_key_ptr,
    current_value_ptr,
    output_ptr,
    compressed_ptr,
    tail_metadata_ptr,
    MAX_CHUNKS,
    CHUNK_TOKENS,
    TAIL_METADATA_STRIDE,
    SCALE,
    HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    WIDTH: tl.constexpr,
    LOG2_WIDTH: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """Finalize one request/head from batched compressed-prefix partials."""
    program = tl.program_id(0)
    batch = program // HEADS
    head = program - batch * HEADS
    kv_head = head // (HEADS // KV_HEADS)
    compressed = tl.load(compressed_ptr + batch).to(tl.int32)
    chunks = tl.cdiv(compressed, CHUNK_TOKENS)
    dims = tl.arange(0, WIDTH)
    chunk_offsets = tl.arange(0, BLOCK_C)
    maximum = tl.full((), float("-inf"), tl.float32)
    denominator = tl.full((), 0.0, tl.float32)
    numerator = tl.zeros((WIDTH,), tl.float32)
    part_base = batch * MAX_CHUNKS
    for start in range(tl.cdiv(chunks, BLOCK_C)):
        local_chunks = start * BLOCK_C + chunk_offsets
        lse = tl.load(
            lses_ptr + (part_base + local_chunks) * HEADS + head,
            mask=local_chunks < chunks,
            other=float("-inf"),
        )
        next_maximum = tl.maximum(maximum, tl.max(lse, axis=0))
        alpha = tl.exp(maximum - next_maximum)
        weights = tl.exp(lse - next_maximum)
        partials = tl.load(
            parts_ptr + ((part_base + local_chunks[:, None]) * HEADS + head) * WIDTH + dims[None, :],
            mask=local_chunks[:, None] < chunks,
            other=0.0,
        )
        numerator = numerator * alpha + tl.sum(weights[:, None] * partials, axis=0)
        denominator = denominator * alpha + tl.sum(weights, axis=0)
        maximum = next_maximum
    rotated = numerator / tl.maximum(denominator, 1e-30)
    for stage in tl.static_range(LOG2_WIDTH):
        stride = 1 << stage
        partner = tl.gather(rotated, dims ^ stride, axis=0)
        rotated = tl.where((dims & stride) == 0, rotated + partner, partner - rotated)
    numerator = rotated * (1.0 / tl.sqrt(float(WIDTH)))
    maximum = tl.where(compressed > 0, maximum + tl.log(tl.maximum(denominator, 1e-30)), float("-inf"))
    denominator = tl.where(compressed > 0, 1.0, 0.0)
    query_ptr += batch * HEADS * WIDTH
    query = tl.load(query_ptr + head * WIDTH + dims).to(tl.float32)
    tail_start = tl.load(tail_metadata_ptr + batch).to(tl.int32)
    tail_tokens = tl.load(tail_metadata_ptr + TAIL_METADATA_STRIDE + batch).to(tl.int32)
    token_offsets = tl.arange(0, BLOCK_T)
    for start in range(tl.cdiv(tail_tokens, BLOCK_T)):
        tokens = start * BLOCK_T + token_offsets
        offsets = ((tail_start + tokens[:, None]) * KV_HEADS + kv_head) * WIDTH + dims[None, :]
        keys = tl.load(tail_key_ptr + offsets, mask=tokens[:, None] < tail_tokens, other=0.0).to(tl.float32)
        scores = tl.sum(keys * query[None, :], axis=1) * SCALE
        scores = tl.where(tokens < tail_tokens, scores, float("-inf"))
        next_maximum = tl.maximum(maximum, tl.max(scores, axis=0))
        alpha = tl.exp(maximum - next_maximum)
        probabilities = tl.exp(scores - next_maximum)
        values = tl.load(tail_value_ptr + offsets, mask=tokens[:, None] < tail_tokens, other=0.0).to(tl.float32)
        numerator = numerator * alpha + tl.sum(probabilities[:, None] * values, axis=0)
        denominator = denominator * alpha + tl.sum(probabilities, axis=0)
        maximum = next_maximum
    current_key_ptr += batch * KV_HEADS * WIDTH
    current_value_ptr += batch * KV_HEADS * WIDTH
    key = tl.load(current_key_ptr + kv_head * WIDTH + dims).to(tl.float32)
    value = tl.load(current_value_ptr + kv_head * WIDTH + dims).to(tl.float32)
    score = tl.sum(key * query, axis=0) * SCALE
    next_maximum = tl.maximum(maximum, score)
    alpha = tl.exp(maximum - next_maximum)
    weight = tl.exp(score - next_maximum)
    numerator = numerator * alpha + value * weight
    denominator = denominator * alpha + weight
    output_ptr += batch * HEADS * WIDTH
    tl.store(output_ptr + head * WIDTH + dims, numerator / denominator)


def finalize_dquant_batch_attention_npu(
    parts,
    lses,
    query,
    tail_key,
    tail_value,
    compressed,
    scale,
    current,
    tail_metadata,
    chunk_tokens,
    output=None,
):
    """Merge batched prefix partials with raw tails and current KV."""
    if parts.ndim != 4 or lses.shape != parts.shape[:3] or not parts.shape[0]:
        raise ValueError("Batched partials require [batch, max_chunks, heads, width]")
    batch, max_chunks, heads, width = parts.shape
    if batch < 1 or query.shape != (batch, heads, width):
        raise ValueError("Batched query layout does not match the partials")
    if query.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("Batched finalize requires BF16 or FP16 queries")
    if parts.dtype != torch.float32 or lses.dtype != torch.float32:
        raise ValueError("Batched partials and LSE must use FP32")
    if compressed.shape != (batch,) or compressed.dtype not in (torch.int32, torch.int64):
        raise ValueError("Batched compressed lengths must contain one integer per request")
    if tail_metadata.shape != (2, batch) or tail_metadata.dtype != torch.int32:
        raise ValueError("Tail metadata must be an int32 [2, batch] tensor")
    if tail_key.ndim != 3 or tail_key.shape != tail_value.shape or tail_key.shape[-1] != width:
        raise ValueError("Batched raw tails must be [tokens, kv_heads, width]")
    kv_heads = tail_key.shape[1]
    if not kv_heads or heads % kv_heads:
        raise ValueError("Query heads must be divisible by KV heads")
    current_key, current_value = current
    if current_key.shape != (batch, kv_heads, width) or current_value.shape != current_key.shape:
        raise ValueError("Current K/V must contain one token per request")
    tensors = [
        parts,
        lses,
        query,
        tail_key,
        tail_value,
        compressed,
        tail_metadata,
        current_key,
        current_value,
    ]
    if any(t.device != query.device or (t is not tail_metadata and not t.is_contiguous()) for t in tensors):
        raise ValueError("Batched finalizer inputs must be contiguous on one device")
    if tail_metadata.stride(1) != 1:
        raise ValueError("Tail metadata rows must be contiguous")
    if any(t.dtype != query.dtype for t in (tail_key, tail_value, current_key, current_value)):
        raise ValueError("Raw K/V must share the query dtype")
    if output is None:
        output = torch.empty_like(query)
    elif (
        output.shape != query.shape
        or output.device != query.device
        or not output.is_contiguous()
        or output.dtype != query.dtype
    ):
        raise ValueError("Batched output must match the query layout and dtype")
    _finalize_dquant_batch_attention_kernel[(batch * heads,)](
        parts,
        lses,
        query,
        tail_key,
        tail_value,
        current_key,
        current_value,
        output,
        compressed,
        tail_metadata,
        max_chunks,
        chunk_tokens,
        tail_metadata.stride(0),
        scale,
        HEADS=heads,
        KV_HEADS=kv_heads,
        WIDTH=width,
        LOG2_WIDTH=width.bit_length() - 1,
        BLOCK_C=16,
        BLOCK_T=32,
        num_warps=4,
    )
    return output


def finalize_dquant_attention_npu(parts, lses, query, tail_key, tail_value, scale, current=None, *, output=None):
    """Merge compressed prefix, inverse rotation, raw tail and optional current KV."""
    if parts.ndim != 3 or not parts.shape[0] or lses.shape != parts.shape[:2]:
        raise ValueError("Expected nonempty [chunks, heads, width] partials and matching LSE")
    chunks, heads, width = parts.shape
    if query.shape != (1, heads, width) or query.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("Expected one BF16 or FP16 query token matching the partials")
    if width <= 0 or width & (width - 1):
        raise ValueError("Head width must be a positive power of two")
    if parts.dtype != torch.float32 or lses.dtype != torch.float32:
        raise ValueError("Partial outputs and LSE must use FP32")
    if tail_key.ndim != 3 or tail_key.shape != tail_value.shape or tail_key.shape[-1] != width:
        raise ValueError("Raw tail K/V shape must be [tokens, kv_heads, width]")
    kv_heads = tail_key.shape[1]
    if not kv_heads or heads % kv_heads:
        raise ValueError("Query head count must be divisible by KV head count")
    tensors = [parts, lses, query, tail_key, tail_value]
    current_key, current_value = tail_key, tail_value
    if current is not None:
        current_key, current_value = current
        if current_key.shape != (1, kv_heads, width) or current_value.shape != current_key.shape:
            raise ValueError("Current K/V must contain one token with matching KV heads")
        tensors.extend(current)
    if any(t.device != query.device or not t.is_contiguous() for t in tensors):
        raise ValueError("Finalizer inputs must be contiguous on one device")
    if any(t.dtype != query.dtype for t in tensors[3:]):
        raise ValueError("Raw K/V must share the query dtype")
    if output is None:
        output = torch.empty_like(query)
    elif (
        output.shape != query.shape
        or output.device != query.device
        or not output.is_contiguous()
        or output.dtype not in (query.dtype, torch.float32)
    ):
        raise ValueError("Output must match the query layout and use query dtype or FP32")
    _finalize_dquant_attention_kernel[(heads,)](
        parts,
        lses,
        query,
        tail_key,
        tail_value,
        current_key,
        current_value,
        output,
        chunks,
        tail_key.shape[0],
        scale,
        HEADS=heads,
        KV_HEADS=kv_heads,
        WIDTH=width,
        LOG2_WIDTH=width.bit_length() - 1,
        BLOCK_C=16,
        BLOCK_T=32,
        HAS_CURRENT=current is not None,
        num_warps=4,
    )
    return output
