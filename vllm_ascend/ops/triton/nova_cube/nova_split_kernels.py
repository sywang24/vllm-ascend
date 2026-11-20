# SPDX-License-Identifier: Apache-2.0
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import extract_slice, insert_slice


@triton.jit
def _split_weights(
    SCORES,
    SC,
    TABLE,
    STARTS,
    ENDS,
    WEIGHTS,
    BIAS,
    MAXIMA,
    SUMS,
    G: tl.constexpr,
    CAP: tl.constexpr,
    PAGE: tl.constexpr,
    SS: tl.constexpr,
    SPLITS: tl.constexpr,
    SPLIT: tl.constexpr,
    FEED: tl.constexpr,
    ALIGN: tl.constexpr,
    H: tl.constexpr = 0,
    TS: tl.constexpr = 0,
):
    query_head, split = tl.program_id(0), tl.program_id(1)
    head = query_head // G
    batch = head // H if H else 0
    head = head % H if H else head
    TABLE += batch * TS
    begin, end = tl.load(STARTS + batch), tl.load(ENDS + batch)
    first = begin // ALIGN * ALIGN + split * SPLIT
    n = tl.arange(0, SPLIT)
    valid = (first + n >= begin) & (first + n < end)
    scores = tl.load(SCORES + query_head * CAP + split * SPLIT + n, valid, -1.0e30)
    maximum = tl.max(scores, 0)
    if PAGE >= SPLIT:
        # A split touches at most two physical pages, each with contiguous DMA.
        first_page = first // PAGE
        page = tl.load(TABLE + first_page, first < end, 0).to(tl.int64)
        within = first % PAGE + n
        base = SC + page * SS + head * 3 * PAGE
        step = tl.load(base + PAGE + within, valid & (within < PAGE), 0).to(tl.float32)
        low = tl.load(base + 2 * PAGE + within, valid & (within < PAGE), 0).to(tl.float32)
        boundary = (first_page + 1) * PAGE
        if (first + SPLIT > boundary) & (end > boundary):
            next_page = tl.load(TABLE + first_page + 1).to(tl.int64)
            next_base = SC + next_page * SS + head * 3 * PAGE
            step += tl.load(next_base + PAGE + within - PAGE, valid & (within >= PAGE), 0).to(tl.float32)
            low += tl.load(next_base + 2 * PAGE + within - PAGE, valid & (within >= PAGE), 0).to(tl.float32)
        prob = tl.where(valid, tl.exp(scores - maximum), 0.0)
        tl.store(WEIGHTS + query_head * CAP + split * SPLIT + n, (prob * step).to(tl.float16))
        total, bias = tl.sum(prob, 0), tl.sum(prob * low, 0)
    else:
        total, bias = tl.full((), 0.0, tl.float32), tl.full((), 0.0, tl.float32)
        lanes = tl.arange(0, FEED)
        for offset in range(0, tl.minimum(SPLIT, end - first), FEED):
            pos = first + offset + lanes
            active = (pos >= begin) & (pos < end)
            score = tl.load(SCORES + query_head * CAP + split * SPLIT + offset + lanes, active, -1.0e30)
            prob = tl.where(active, tl.exp(score - maximum), 0.0)
            page = tl.load(TABLE + (first + offset) // PAGE).to(tl.int64)
            within = (first + offset) % PAGE + lanes
            scale_base = SC + page * SS + head * 3 * PAGE + within
            step = tl.load(scale_base + PAGE).to(tl.float32)
            low = tl.load(scale_base + 2 * PAGE).to(tl.float32)
            weights = (prob * step).to(tl.float16)
            tl.store(WEIGHTS + query_head * CAP + split * SPLIT + offset + lanes, weights)
            bias += tl.sum(prob * low, 0)
            total += tl.sum(prob, 0)
    state = query_head * SPLITS + split
    tl.store(MAXIMA + state, maximum)
    tl.store(SUMS + state, total)
    tl.store(BIAS + state, bias)


@triton.jit
def _pv_prepared(
    WEIGHTS,
    VC,
    TABLE,
    STARTS,
    ENDS,
    BIAS,
    PART,
    G: tl.constexpr,
    CAP: tl.constexpr,
    PAGE: tl.constexpr,
    VS: tl.constexpr,
    SPLITS: tl.constexpr,
    SPLIT: tl.constexpr,
    T: tl.constexpr,
    FEED: tl.constexpr,
    VD: tl.constexpr,
    ALIGN: tl.constexpr,
    H: tl.constexpr = 0,
    TS: tl.constexpr = 0,
):
    task = tl.program_id(0)
    head, split = task // SPLITS, task % SPLITS
    batch = head // H if H else 0
    local_head = head % H if H else head
    TABLE += batch * TS
    begin, end = tl.load(STARTS + batch), tl.load(ENDS + batch)
    first = begin // ALIGN * ALIGN + split * SPLIT
    row, n, vg = tl.arange(0, 16), tl.arange(0, FEED), tl.arange(0, 64)
    accum = tl.zeros((16, VD), tl.float32)
    for offset in range(0, tl.minimum(SPLIT, end - first), FEED):
        weights = tl.load(
            WEIGHTS + (head * G + row[:, None]) * CAP + split * SPLIT + offset + n[None, :], row[:, None] < G, 0
        )
        page = tl.load(TABLE + (first + offset) // PAGE).to(tl.int64)
        within = (first + offset) % PAGE + n
        packed = tl.load(VC + page * VS + (local_head * 64 + vg[:, None]) * PAGE + within[None, :])
        for block in tl.static_range(FEED // T):
            codes = extract_slice(packed, (0, block * T), (64, T), (1, 1)).to(tl.int16)
            a = extract_slice(weights, (0, block * T), (16, T), (1, 1))
            value = tl.zeros((VD, T), tl.float16)
            for local_component in tl.static_range(VD // 64):
                component = tl.program_id(1) * (VD // 64) + local_component
                part = ((codes >> (2 * component)) & 3).to(tl.float16)
                value = insert_slice(value, part, (local_component * 64, 0), (64, T), (1, 1))
            accum = tl.dot(a, tl.trans(value), accum)
    state = (head * G + row) * SPLITS + split
    bias = tl.load(BIAS + state, row < G, 0)
    dims = tl.program_id(1) * VD + tl.arange(0, VD)
    tl.store(PART + state[:, None] * 256 + dims[None, :], accum + bias[:, None], row[:, None] < G)
