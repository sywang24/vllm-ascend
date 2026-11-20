# SPDX-License-Identifier: Apache-2.0
"""Reduce all split states in one pass when there are at most 128 splits."""

from vllm.triton_utils import tl, triton


@triton.jit
def merge_one_pass(PART, MAXIMA, SUMS, OUT, LSE, SPLITS: tl.constexpr, B: tl.constexpr, N: tl.constexpr):
    head, part = tl.program_id(0), tl.program_id(1)
    dims = part * N + tl.arange(0, N)
    split = tl.arange(0, B)
    state = head * SPLITS + split
    maximum = tl.load(MAXIMA + state, split < SPLITS, -1.0e30)
    total = tl.load(SUMS + state, split < SPLITS, 0)
    peak = tl.max(maximum, 0)
    weight = tl.exp(maximum - peak)
    denom = tl.sum(total * weight, 0)
    subtotal = tl.load(PART + state[:, None] * 256 + dims[None, :], split[:, None] < SPLITS, 0)
    value = tl.sum(subtotal * weight[:, None], 0) / tl.maximum(denom, 1.0e-30)
    out_dim = dims % 64 * 4 + dims // 64
    tl.store(OUT + head * 256 + out_dim, value)
    if part == 0:
        tl.store(LSE + head, tl.where(denom > 0, peak + tl.log(denom), -float("inf")))
