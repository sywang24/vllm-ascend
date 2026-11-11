# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.ops.triton.dquant import dquant_quantize_encode_npu, dquant_quantize_npu, rans_encode_npu
from vllm_ascend.quantization.methods.kv_cache.dquant_npu_cache import DQuantNPUCache

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="Ascend NPU is unavailable")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("levels,container", [(8, 155), (6, 109)])
@pytest.mark.parametrize("rows", [1, 64, 257])
def test_fused_encoder_matches_separate_path(dtype, levels, container, rows):
    torch.manual_seed(10731 + rows + levels)
    cache = DQuantNPUCache(rows, 512, container, levels, device="npu")
    values = torch.randn((rows, 512), dtype=dtype, device="npu")
    if rows > 1:
        values[0] = 0
        values[1] = 1
        values[2] = torch.linspace(-16, 16, 512, device="npu").to(dtype)
        values[3] *= 0.0001
        values[4] *= 128
        values[5] = 0
        values[5, :16] = torch.arange(16, device="npu").to(dtype)
    symbols, scales, offsets = dquant_quantize_npu(
        values, cache.rates, container * 8 - 32, minimum_symbol_rate=cache.minimum_symbol_rate
    )
    payloads, status = rans_encode_npu(
        symbols, container, cache.frequency_by_symbol, cache.cumulative_by_symbol, cache.table.total
    )
    actual = dquant_quantize_encode_npu(
        values,
        cache.rates,
        container,
        cache.frequency_by_symbol,
        cache.cumulative_by_symbol,
        cache.table.total,
        minimum_symbol_rate=cache.minimum_symbol_rate,
    )
    for expected, result in zip((symbols, scales, offsets, status), (actual[0], actual[1], actual[2], actual[4])):
        torch.testing.assert_close(expected, result, atol=0, rtol=0)
    valid = status == 0
    torch.testing.assert_close(payloads[valid], actual[3][valid], atol=0, rtol=0)


@pytest.mark.parametrize("levels,container", [(8, 155), (6, 109)])
def test_fused_encoder_overflow_rejected_before_commit(levels, container):
    cache = DQuantNPUCache(4, 512, container, levels, device="npu")
    before = cache.storage.clone()
    values = torch.randn((4, 512), dtype=torch.bfloat16, device="npu")
    symbols = torch.zeros_like(values, dtype=torch.int16)
    metadata = torch.zeros((4,), dtype=torch.float16, device="npu")
    payload = torch.zeros((4, container), dtype=torch.uint8, device="npu")
    status = torch.tensor([0, 1, 0, 0], dtype=torch.int32, device="npu")
    with (
        patch(
            "vllm_ascend.quantization.methods.kv_cache.dquant_npu_cache.dquant_quantize_encode_npu",
            return_value=(symbols, metadata, metadata, payload, status),
        ),
        pytest.raises(RuntimeError, match="refusing to commit"),
    ):
        cache.write(torch.arange(4, device="npu"), values)
    assert torch.equal(cache.storage, before)
    assert not cache.initialized.any().item()
