import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton import dquant
from vllm_ascend.quantization.methods.kv_cache.dquant_npu_cache import DQuantNPUCache


def exhaustive(values, rates, budget):
    symbols = torch.empty_like(values, dtype=torch.int16)
    scales = torch.empty(values.shape[0], device=values.device, dtype=torch.float16)
    offsets = torch.empty_like(scales)
    kwargs = dict(
        WIDTH=values.shape[1],
        LEVELS=rates.numel(),
        BLOCK_WIDTH=dquant.triton.next_power_of_2(values.shape[1]),
        TARGET_RATE_BITS=budget,
        DRIFT_STEPS=12,
        FAST_AFFINE=False,
        num_warps=4,
    )
    for kernel in (dquant._dquant_quantize_kernel, dquant._dquant_finalize_kernel):
        kernel[(values.shape[0],)](values, rates, symbols, scales, offsets, **kwargs)
    return symbols, scales, offsets


@pytest.mark.parametrize("width,budget", [(128, 954), (512, 3200), (512, 3816), (513, 3824), (1024, 7632)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_affine_fast_path_matches_exhaustive_quantizer(width, budget, dtype):
    torch.manual_seed(20261026)
    cache = DQuantNPUCache(128, width, 481, 128, alpha=1.0)
    values = torch.randn((128, width), device="npu", dtype=torch.float32)
    values[0].zero_()
    values[1].fill_(2)
    values[2] *= 1e-17
    values[3] += 30
    values[4] = torch.linspace(-1, 1, width, device="npu")
    values[5] = torch.arange(width, device="npu").float() % 128 + 0.5
    values[6] = values[6] * 0.125 + 1e6
    values = values.to(dtype)
    expected = exhaustive(values, cache.rates, budget)
    actual = dquant.dquant_quantize_npu(values, cache.rates, budget, minimum_symbol_rate=cache.minimum_symbol_rate)
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert torch.all(cache.rates[actual[0].long()].sum(1) <= budget).item()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fast_quantizer_keeps_record_bytes_and_overflow_status(dtype):
    torch.manual_seed(20261026)
    cache = DQuantNPUCache(8192, 512, 481, 128, alpha=1.0)
    values = torch.randn((8192, 512), device="npu", dtype=dtype)
    expected = exhaustive(values, cache.rates, 3816)
    actual = dquant.dquant_quantize_npu(values, cache.rates, 3816, minimum_symbol_rate=cache.minimum_symbol_rate)
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    args = (481, cache.frequency_by_symbol, cache.cumulative_by_symbol, cache.table.total)
    expected_payload, expected_status = dquant.rans_encode_npu(expected[0], *args)
    payload, status = dquant.rans_encode_npu(actual[0], *args)
    torch.testing.assert_close(payload, expected_payload, rtol=0, atol=0)
    torch.testing.assert_close(status, expected_status, rtol=0, atol=0)
    assert not status.any().item()
