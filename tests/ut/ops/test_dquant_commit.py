# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.ops.triton.dquant_commit import commit_records_npu
from vllm_ascend.ops.triton.dquant_symbols import pack_symbols
from vllm_ascend.quantization.methods.kv_cache.dquant_npu_cache import DQuantNPUCache

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="Ascend NPU is unavailable")


@pytest.mark.parametrize("rows", [0, 1, 65])
@pytest.mark.parametrize(
    "stream,container,symbol_cache", [(37, 9, False), (512, 109, False), (512, 155, True), (1024, 331, True)]
)
def test_record_commit_preserves_padding_means_and_unused_slots(rows, stream, container, symbol_cache):
    capacity = 136
    symbol_bytes = stream // 2 if symbol_cache else 0
    metadata = (container + 1) & ~1
    width = metadata + 4 + 32 + symbol_bytes
    storage = torch.full((capacity, width), 173, device="npu", dtype=torch.uint8)
    cache = DQuantNPUCache(capacity, stream, container, 8, storage=storage)
    if symbol_cache:
        cache.symbols = storage[:, -symbol_bytes:]
    before = storage.clone()
    slots = torch.arange(rows, device="npu", dtype=torch.int64) * 2 + 3
    payload = torch.randint(0, 256, (rows, container), device="npu", dtype=torch.uint8)
    scales = torch.randn(rows, device="npu", dtype=torch.float16)
    offsets = torch.randn_like(scales)
    symbols = torch.randint(0, 8, (rows, stream), device="npu", dtype=torch.int16)
    expected = before.clone()
    expected[slots, :container] = payload
    expected[slots, metadata : metadata + 2] = scales.view(torch.uint8).view(rows, 2)
    expected[slots, metadata + 2 : metadata + 4] = offsets.view(torch.uint8).view(rows, 2)
    if symbol_cache:
        expected[slots, -symbol_bytes:] = pack_symbols(symbols)
    initialized = torch.zeros_like(cache.initialized)
    initialized[slots] = True
    commit_records_npu(cache, slots, payload, scales, offsets, symbols)
    torch.testing.assert_close(storage, expected, rtol=0, atol=0)
    torch.testing.assert_close(cache.initialized, initialized, rtol=0, atol=0)


def test_overflow_never_reaches_record_commit():
    cache = DQuantNPUCache(4, 512, 155, 8)
    values = torch.randn((4, 512), device="npu", dtype=torch.bfloat16)
    result = (
        torch.zeros((4, 512), device="npu", dtype=torch.int16),
        torch.ones(4, device="npu", dtype=torch.float16),
        torch.zeros(4, device="npu", dtype=torch.float16),
        torch.zeros((4, 155), device="npu", dtype=torch.uint8),
        torch.tensor([0, 1, 0, 0], device="npu", dtype=torch.int32),
    )
    module = "vllm_ascend.quantization.methods.kv_cache.dquant_npu_cache."
    with (
        patch(module + "dquant_quantize_encode_npu", return_value=result),
        patch(module + "commit_records_npu") as commit,
        pytest.raises(RuntimeError, match="refusing to commit"),
    ):
        cache.write(torch.arange(4, device="npu"), values)
    commit.assert_not_called()
