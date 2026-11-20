# SPDX-License-Identifier: Apache-2.0
"""Exact-window attention using physical row order for long contexts."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _exact_physical(
    Q,
    SLOTS,
    LENGTHS,
    KEY,
    VALUE,
    COLD,
    COLD_LSE,
    OUT,
    OUT_LSE,
    H: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    SINK: tl.constexpr,
    RING: tl.constexpr,
    LIMIT: tl.constexpr,
    SCALE: tl.constexpr,
    T: tl.constexpr,
    EXACT_ONLY: tl.constexpr,
):
    bh, group = tl.program_id(0), tl.program_id(1)
    out_head = bh * G + group
    slot, length = tl.load(SLOTS + bh // H), tl.load(LENGTHS + bh // H)
    active = (slot >= 0) & (slot < LIMIT) & (length > 0)
    safe_slot = tl.where(active, slot, LIMIT)
    dim = tl.arange(0, D)
    query = tl.load(Q + out_head * D + dim)
    if EXACT_ONLY:
        peak = tl.full((), -float("inf"), tl.float32)
        norm = tl.full((), 0.0, tl.float32)
        acc = tl.zeros((D,), tl.float32)
    else:
        peak = tl.load(COLD_LSE + out_head)
        norm = tl.where(peak > -float("inf"), 1.0, 0.0)
        acc = tl.load(COLD + out_head * D + dim) * norm
    tail = (length - 1) % RING
    for start in range(0, SINK + RING, T):
        row = start + tl.arange(0, T)
        valid = active & (row < SINK + RING)
        if length < SINK + RING:
            # For partial rings, map a physical row to its last absolute position.
            back = (tail - (row - SINK) + RING) % RING
            valid &= tl.where(row < SINK, row < length, length - 1 - back >= SINK)
        address = ((safe_slot * (SINK + RING) + row) * H + bh % H) * D
        keys = tl.load(KEY + address[:, None] + dim[None, :], valid[:, None], 0).to(tl.float32)
        logits = tl.where(valid, tl.sum(keys * query[None, :], 1) * SCALE, -float("inf"))
        next_peak = tl.maximum(peak, tl.max(logits, 0))
        safe_peak = tl.where(next_peak > -float("inf"), next_peak, 0.0)
        rescale = tl.exp(peak - safe_peak)
        prob = tl.exp(logits - safe_peak)
        values = tl.load(VALUE + address[:, None] + dim[None, :], valid[:, None], 0).to(tl.float32)
        acc = acc * rescale + tl.sum(values * prob[:, None], 0)
        norm = norm * rescale + tl.sum(prob, 0)
        peak = next_peak
    tl.store(OUT + out_head * D + dim, acc / tl.maximum(norm, 1.0e-30))
    tl.store(OUT_LSE + out_head, tl.where(norm > 0, peak + tl.log(norm), -float("inf")))


def exact_physical(window, query, slots, lengths, scale, cold, lse, tile=64, buffers=None):
    if buffers is None:
        buffers = torch.empty_like(query), torch.empty_like(lse)
    output, final_lse = buffers
    batch, heads, groups, dim = query.shape
    _exact_physical[(batch * heads, groups)](
        query.contiguous(),
        slots.contiguous(),
        lengths.contiguous(),
        window.key,
        window.value,
        cold.contiguous(),
        lse.contiguous(),
        output,
        final_lse,
        heads,
        groups,
        dim,
        window.sink,
        window.ring,
        window.limit,
        scale,
        tile,
        False,
        compile_mode="simd",
        num_stages=1,
        enable_auto_bind_sub_block=False,
    )
    return output, final_lse, buffers


def exact_only(window, query, slots, lengths, scale, out, lse, tile):
    batch, heads, groups, dim = query.shape
    _exact_physical[(batch * heads, groups)](
        query,
        slots,
        lengths,
        window.key,
        window.value,
        out,
        lse,
        out,
        lse,
        heads,
        groups,
        dim,
        window.sink,
        window.ring,
        window.limit,
        scale,
        tile,
        True,
        compile_mode="simd",
        num_stages=1,
        enable_auto_bind_sub_block=False,
    )


@triton.jit
def _merge_exact_and_cold(
    COLD,
    CLSE,
    EXACT,
    ELSE,
    OUT,
    D: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    SB: tl.constexpr,
    SH: tl.constexpr,
    SG: tl.constexpr,
    SD: tl.constexpr,
):
    head = tl.program_id(0)
    cold = tl.load(CLSE + head)
    exact = tl.load(ELSE + head)
    highest = tl.maximum(cold, exact)
    safe_peak = tl.where(highest > -float("inf"), highest, 0.0)
    wc = tl.exp(cold - safe_peak)
    we = tl.exp(exact - safe_peak)
    lane = tl.arange(0, D)
    cold_base = (head // (H * G)) * SB + (head // G % H) * SH + (head % G) * SG
    dim = head * D + lane
    a = tl.load(COLD + cold_base + lane * SD)
    b = tl.load(EXACT + dim)
    result = (wc * a + we * b) / tl.maximum(wc + we, 1e-30)
    tl.store(OUT + dim, result)


def merge_into(cold, cold_lse, exact, exact_lse, out):
    _, heads, groups, dim = cold.shape
    _merge_exact_and_cold[(cold_lse.numel(),)](
        cold,
        cold_lse,
        exact,
        exact_lse,
        out,
        dim,
        heads,
        groups,
        *cold.stride(),
        compile_mode="simd",
        num_stages=1,
        enable_auto_bind_sub_block=False,
    )
