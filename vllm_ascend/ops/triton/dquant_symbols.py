# SPDX-License-Identifier: Apache-2.0
"""Attention over cached packed quantizer symbols, without rANS replay."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _symbol_attention_kernel(
    query_ptr,
    original_ptr,
    ks_ptr,
    vs_ptr,
    kscale_ptr,
    koff_ptr,
    vscale_ptr,
    voff_ptr,
    table_ptr,
    key_storage_ptr,
    output_ptr,
    lse_ptr,
    SCALE,
    HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_T: tl.constexpr,
    MEAN_TOKENS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    KEY_ROW_BYTES: tl.constexpr,
    MEAN_OFFSET: tl.constexpr,
    STRIPE_BYTES: tl.constexpr,
    KSTRIDE: tl.constexpr,
    VSTRIDE: tl.constexpr,
    IS_BF16: tl.constexpr,
    KS_ROW: tl.constexpr,
    VS_ROW: tl.constexpr,
):
    chunk = tl.program_id(0)
    head = tl.program_id(1)
    kv = head // (HEADS // KV_HEADS)
    d = tl.arange(0, WIDTH // 2)
    t = chunk * BLOCK_T + tl.arange(0, BLOCK_T)
    blocks = tl.load(table_ptr + t // BLOCK_SIZE).to(tl.int32)
    slots = blocks * BLOCK_SIZE + t % BLOCK_SIZE
    qe = tl.load(query_ptr + head * WIDTH + 2 * d).to(tl.float32)
    qo = tl.load(query_ptr + head * WIDTH + 2 * d + 1).to(tl.float32)
    oe = tl.load(original_ptr + head * WIDTH + 2 * d).to(tl.float32)
    oo = tl.load(original_ptr + head * WIDTH + 2 * d + 1).to(tl.float32)
    mean_start = chunk * BLOCK_T // MEAN_TOKENS * MEAN_TOKENS
    mean_byte = (kv * WIDTH + 2 * d) * 2
    mean_record = mean_start + mean_byte // STRIPE_BYTES
    mean_block = tl.load(table_ptr + mean_record // BLOCK_SIZE).to(tl.int32)
    mean_slot = mean_block * BLOCK_SIZE + mean_record % BLOCK_SIZE
    mean_address = mean_slot * KEY_ROW_BYTES + MEAN_OFFSET + mean_byte % STRIPE_BYTES
    b0 = tl.load(key_storage_ptr + mean_address).to(tl.uint32)
    b1 = tl.load(key_storage_ptr + mean_address + 1).to(tl.uint32)
    b2 = tl.load(key_storage_ptr + mean_address + 2).to(tl.uint32)
    b3 = tl.load(key_storage_ptr + mean_address + 3).to(tl.uint32)
    even_bits = (b0 | (b1 << 8)).to(tl.uint16)
    odd_bits = (b2 | (b3 << 8)).to(tl.uint16)
    if IS_BF16:
        even_mean = even_bits.to(tl.bfloat16, bitcast=True).to(tl.float32)
        odd_mean = odd_bits.to(tl.bfloat16, bitcast=True).to(tl.float32)
    else:
        even_mean = even_bits.to(tl.float16, bitcast=True).to(tl.float32)
        odd_mean = odd_bits.to(tl.float16, bitcast=True).to(tl.float32)
    mean_term = tl.sum(oe * even_mean + oo * odd_mean, axis=0)
    columns = kv * WIDTH // 2 + d[None, :]
    kbytes = tl.load(ks_ptr + slots[:, None] * KS_ROW + columns).to(tl.int32)
    ke = (kbytes & 15).to(tl.float32)
    ko = (kbytes >> 4).to(tl.float32)
    kscale = tl.load(kscale_ptr + slots * KSTRIDE).to(tl.float32)
    koff = tl.load(koff_ptr + slots * KSTRIDE).to(tl.float32)
    scores = (
        tl.sum(ke * qe[None, :] + ko * qo[None, :], axis=1) * kscale + tl.sum(qe + qo, axis=0) * koff + mean_term
    ) * SCALE
    maximum = tl.max(scores, axis=0)
    probs = tl.exp(scores - maximum)
    denom = tl.sum(probs, axis=0)
    vbytes = tl.load(vs_ptr + slots[:, None] * VS_ROW + columns).to(tl.int32)
    ve = (vbytes & 15).to(tl.float32)
    vo = (vbytes >> 4).to(tl.float32)
    vscale = tl.load(vscale_ptr + slots * VSTRIDE).to(tl.float32)
    voff = tl.load(voff_ptr + slots * VSTRIDE).to(tl.float32)
    weights = probs * vscale
    offset = tl.sum(probs * voff, axis=0)
    outlo = (tl.sum(weights[:, None] * ve, axis=0) + offset) / denom
    outhi = (tl.sum(weights[:, None] * vo, axis=0) + offset) / denom
    base = (chunk * HEADS + head) * WIDTH + 2 * d
    tl.store(output_ptr + base, outlo)
    tl.store(output_ptr + base + 1, outhi)
    tl.store(lse_ptr + chunk * HEADS + head, maximum + tl.log(denom))


def pack_symbols(symbols):
    codes = symbols.to(torch.uint8)
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()


def symbol_dquant_attention_npu(
    query,
    rotated_query,
    key_symbols,
    value_symbols,
    key_scale,
    key_offset,
    value_scale,
    value_offset,
    block_table,
    key_storage,
    *,
    scale,
    compressed,
    block_size,
    key_window,
    mean_offset,
    stripe_bytes,
    chunk_tokens=16,
):
    num_q_heads, head_size = query.shape[1:]
    num_kv_heads = key_symbols.shape[1] * 2 // head_size
    chunk_tokens = int(chunk_tokens)
    if chunk_tokens <= 0 or key_window % chunk_tokens:
        raise ValueError("Symbol attention chunk must divide the key window")
    if compressed <= 0 or compressed % chunk_tokens:
        raise ValueError("Symbol attention requires complete aligned key windows")
    if key_window % 2 or block_size % key_window or stripe_bytes % 4:
        raise ValueError("Symbol attention requires page-aligned windows and four-byte mean stripes")
    chunks = compressed // chunk_tokens
    output = torch.empty((chunks, num_q_heads, head_size), dtype=torch.float32, device=query.device)
    lse = torch.empty((chunks, num_q_heads), dtype=torch.float32, device=query.device)
    _symbol_attention_kernel[(chunks, num_q_heads)](
        rotated_query.contiguous(),
        query.contiguous(),
        key_symbols,
        value_symbols,
        key_scale,
        key_offset,
        value_scale,
        value_offset,
        block_table,
        key_storage,
        output,
        lse,
        scale,
        HEADS=num_q_heads,
        KV_HEADS=num_kv_heads,
        WIDTH=head_size,
        BLOCK_T=chunk_tokens,
        MEAN_TOKENS=key_window,
        BLOCK_SIZE=block_size,
        KEY_ROW_BYTES=key_storage.stride(0),
        MEAN_OFFSET=mean_offset,
        STRIPE_BYTES=stripe_bytes,
        KSTRIDE=key_scale.stride(0),
        VSTRIDE=value_scale.stride(0),
        IS_BF16=query.dtype == torch.bfloat16,
        KS_ROW=key_symbols.stride(0),
        VS_ROW=value_symbols.stride(0),
    )
    return output, lse
