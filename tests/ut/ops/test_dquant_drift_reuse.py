# SPDX-License-Identifier: Apache-2.0
"""Keep drift assignments identical when avoiding repeated alphabet scans."""

import pytest
import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.dquant import _dquant_assign, _dquant_drift_reference
from vllm_ascend.quantization.methods.kv_cache.dquant_npu_cache import DQuantNPUCache

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="Ascend NPU is unavailable")


@triton.jit
def _legacy_drift(
    values,
    rates,
    scale,
    offset,
    WIDTH: tl.constexpr,
    LEVELS: tl.constexpr,
    BLOCK: tl.constexpr,
    budget,
    STEPS: tl.constexpr,
):
    """Frozen search before reuse, including the unconditional final assignment."""
    lane = tl.arange(0, BLOCK)
    row_min = tl.full((BLOCK,), float("inf"), tl.float32)
    row_max = tl.full((BLOCK,), float("-inf"), tl.float32)
    for symbol in range(int(LEVELS)):
        error = values - (offset + scale * symbol)
        distortion = error * error
        row_min = tl.minimum(row_min, distortion)
        row_max = tl.maximum(row_max, distortion)
    maximum = tl.max(tl.where(lane < WIDTH, row_max, float("-inf")), 0)
    minimum = tl.min(tl.where(lane < WIDTH, row_min, float("inf")), 0)
    if maximum > minimum:
        assigned, assigned_rates = _dquant_assign(values, scale, offset, rates, 0.0, LEVELS, BLOCK)
        total = tl.sum(tl.where(lane < WIDTH, assigned_rates, 0.0), 0)
        if total > budget:
            high = tl.log2(8.0 * (maximum - minimum))
            low = high - 40.0
            for _ in range(int(STEPS)):
                midpoint = (low + high) * 0.5
                assigned, assigned_rates = _dquant_assign(
                    values, scale, offset, rates, tl.exp2(midpoint), LEVELS, BLOCK
                )
                total = tl.sum(tl.where(lane < WIDTH, assigned_rates, 0.0), 0)
                if total <= budget:
                    high = midpoint
                else:
                    low = midpoint
            assigned, _ = _dquant_assign(values, scale, offset, rates, tl.exp2(high), LEVELS, BLOCK)
    else:
        alphabet_rates = tl.load(rates + lane, lane < LEVELS, float("inf"))
        assigned = tl.full((BLOCK,), tl.argmin(alphabet_rates, 0), tl.int32)
    return assigned


@triton.jit
def _drift_rows(
    values,
    rates,
    scales,
    offsets,
    budgets,
    output,
    WIDTH: tl.constexpr,
    LEVELS: tl.constexpr,
    BLOCK: tl.constexpr,
    STEPS: tl.constexpr,
    LEGACY: tl.constexpr,
):
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    data = tl.load(values + row * WIDTH + lane, lane < WIDTH, 0).to(tl.float32)
    scale = tl.load(scales + row).to(tl.float32)
    offset = tl.load(offsets + row).to(tl.float32)
    budget = tl.load(budgets + row)
    if LEGACY:
        symbols = _legacy_drift(data, rates, scale, offset, WIDTH, LEVELS, BLOCK, budget, STEPS)
    else:
        symbols = _dquant_drift_reference(data, rates, scale, offset, WIDTH, LEVELS, BLOCK, budget, STEPS)
    tl.store(output + row * WIDTH + lane, symbols, lane < WIDTH)


@pytest.mark.parametrize("steps", [0, 1, 12])
@pytest.mark.parametrize("levels,width,container", [(8, 512, 155), (6, 257, 109)])
@pytest.mark.parametrize("sparse_rates", [False, True])
def test_reuse_matches_legacy(steps, levels, width, container, sparse_rates):
    torch.manual_seed(8191 + steps)
    rows = 192
    cache = DQuantNPUCache(1, 512, container, levels, device="npu", alpha=1.4)
    rates = cache.rates.clone()
    if sparse_rates:
        rates[1] = float("inf")
    values = torch.randn(rows, width, device="npu", dtype=torch.float32)
    magnitudes = torch.tensor([1e-16, 1e-8, 1e-4, 1.0, 100.0, 1e8, 1e16, 0.0], device="npu")
    values *= magnitudes.repeat(rows // magnitudes.numel())[:, None]
    low, high = values.amin(-1), values.amax(-1)
    scales = ((high - low) / (levels - 1)).contiguous()
    offsets = low.contiguous()
    budgets = torch.tensor([0, container * 8 - 32, width * 64], device="npu", dtype=torch.int32).repeat(rows // 3)
    outputs = [torch.empty(rows, width, device="npu", dtype=torch.int16) for _ in range(2)]

    def call():
        for legacy, output in zip((True, False), outputs):
            _drift_rows[(rows,)](
                values,
                rates,
                scales,
                offsets,
                budgets,
                output,
                width,
                levels,
                triton.next_power_of_2(width),
                steps,
                legacy,
                num_warps=4,
            )

    call()
    torch.npu.synchronize()
    assert torch.equal(*outputs)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="global"):
        call()
    values.neg_()
    offsets.copy_(-high)
    graph.replay()
    torch.npu.synchronize()
    assert torch.equal(*outputs)
