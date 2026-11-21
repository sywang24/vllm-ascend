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
    # The ring is a permutation of the complete recent set once the window
    # is full, so physical order is sufficient for an attention reduction.
    # Keep the logical-to-ring mapping only for short requests.
    full_window = length >= SINK + RING
    for start in range(0, SINK + RING, T):
        row = start + tl.arange(0, T)
        if full_window:
            valid = active & (row < SINK + RING)
            index = row
        else:
            tail = (length - 1) % RING
            back = (tail - (row - SINK) + RING) % RING
            valid = active & (row < SINK + RING)
            valid &= tl.where(row < SINK, row < length, length - 1 - back >= SINK)
            index = tl.where(row < SINK, row, SINK + (length - 1 - back) % RING)
        address = ((safe_slot * (SINK + RING) + index) * H + bh % H) * D
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
def _exact_tiles(
    Q,
    SLOTS,
    LENGTHS,
    KEY,
    VALUE,
    PARTIAL,
    LSE,
    H: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    SINK: tl.constexpr,
    RING: tl.constexpr,
    LIMIT: tl.constexpr,
    SCALE: tl.constexpr,
    T: tl.constexpr,
    PARTS: tl.constexpr,
):
    bh, group, part = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    head = bh * G + group
    slot = tl.load(SLOTS + bh // H)
    length = tl.load(LENGTHS + bh // H)
    active = (slot >= 0) & (slot < LIMIT) & (length > 0)
    slot = tl.where(active, slot, LIMIT)
    row = part * T + tl.arange(0, T)
    valid = active & (row < SINK + RING)
    if length < SINK + RING:
        tail = (length - 1) % RING
        back = (tail - (row - SINK) + RING) % RING
        valid &= tl.where(row < SINK, row < length, length - 1 - back >= SINK)
    dim = tl.arange(0, D)
    query = tl.load(Q + head * D + dim).to(tl.float32)
    address = ((slot * (SINK + RING) + row) * H + bh % H) * D
    keys = tl.load(KEY + address[:, None] + dim[None, :], valid[:, None], 0).to(tl.float32)
    logits = tl.where(valid, tl.sum(keys * query[None, :], 1) * SCALE, -float("inf"))
    peak = tl.max(logits, 0)
    safe_peak = tl.where(peak > -float("inf"), peak, 0.0)
    weights = tl.exp(logits - safe_peak)
    denom = tl.sum(weights, 0)
    values = tl.load(VALUE + address[:, None] + dim[None, :], valid[:, None], 0).to(tl.float32)
    output = tl.sum(values * weights[:, None], 0) / tl.maximum(denom, 1.0e-30)
    state = head * PARTS + part
    tl.store(PARTIAL + state * D + dim, output)
    tl.store(LSE + state, tl.where(denom > 0, safe_peak + tl.log(denom), -float("inf")))


def exact_partial(window, query, slots, lengths, scale, tile=64, buffers=None):
    batch, heads, groups, dim = query.shape
    parts = triton.cdiv(window.capacity, tile)
    if buffers is None:
        buffers = (
            torch.empty((*query.shape[:-1], parts, dim), device=query.device, dtype=torch.float32),
            torch.empty((*query.shape[:-1], parts), device=query.device, dtype=torch.float32),
        )
    partial, lse = buffers
    _exact_tiles[(batch * heads, groups, parts)](
        query,
        slots,
        lengths,
        window.key,
        window.value,
        partial,
        lse,
        heads,
        groups,
        dim,
        window.sink,
        window.ring,
        window.limit,
        scale,
        tile,
        parts,
        compile_mode="simd",
        num_stages=1,
        enable_auto_bind_sub_block=False,
    )
    return partial, lse


@triton.jit
def _merge_exact_tiles(
    COLD,
    CLSE,
    PARTIAL,
    ELSE,
    OUT,
    D: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    PARTS: tl.constexpr,
    P: tl.constexpr,
    SB: tl.constexpr,
    SH: tl.constexpr,
    SG: tl.constexpr,
    SD: tl.constexpr,
):
    head = tl.program_id(0)
    part = tl.arange(0, P)
    dim = tl.arange(0, D)
    cold_lse = tl.load(CLSE + head)
    exact_lse = tl.load(ELSE + head * PARTS + part, part < PARTS, -float("inf"))
    peak = tl.maximum(cold_lse, tl.max(exact_lse, 0))
    safe_peak = tl.where(peak > -float("inf"), peak, 0.0)
    wc = tl.exp(cold_lse - safe_peak)
    we = tl.exp(exact_lse - safe_peak)
    cold_base = head // (H * G) * SB + (head // G % H) * SH + head % G * SG
    cold = tl.load(COLD + cold_base + dim * SD)
    exact = tl.load(PARTIAL + (head * PARTS + part[:, None]) * D + dim[None, :], part[:, None] < PARTS, 0)
    output = (wc * cold + tl.sum(we[:, None] * exact, 0)) / tl.maximum(wc + tl.sum(we, 0), 1.0e-30)
    tl.store(OUT + head * D + dim, output)


def merge_partial_into(cold, cold_lse, partial, out):
    values, lse = partial
    _, heads, groups, dim = cold.shape
    parts = lse.shape[-1]
    _merge_exact_tiles[(cold_lse.numel(),)](
        cold,
        cold_lse,
        values,
        lse,
        out,
        dim,
        heads,
        groups,
        parts,
        triton.next_power_of_2(parts),
        *cold.stride(),
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
