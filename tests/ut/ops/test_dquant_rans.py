# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from vllm_ascend.quantization.methods.kv_cache.dquant_npu_cache import DQuantNPUCache, _record_views

repo_root = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "vllm_ascend/quantization/methods/kv_cache/dquant_codec.py").is_file()
)


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


codec = _load_module("dquant_codec", repo_root / "vllm_ascend/quantization/methods/kv_cache/dquant_codec.py")
dquant = _load_module("dquant_triton", repo_root / "vllm_ascend/ops/triton/dquant.py")


def _build_decode_lut(table):
    symbols = []
    frequencies = []
    cumulative = []
    for symbol, (frequency, start, end) in enumerate(zip(table.frequencies, table.cumulative, table.cumulative[1:])):
        count = end - start
        symbols.extend([symbol] * count)
        frequencies.extend([frequency] * count)
        cumulative.extend([start] * count)
    return tuple(torch.tensor(values, dtype=torch.int32, device="npu") for values in (symbols, frequencies, cumulative))


def test_dquant_record_views_preserve_page_stride_and_alias_storage():
    storage = torch.zeros((3, 128), dtype=torch.uint8)
    payloads, scales, offsets = _record_views(storage, 3, 95, 96, 128)

    payloads[1, 4] = 17
    scales[1] = 0.5
    offsets[1] = -1.25

    assert storage[1, 4].item() == 17
    assert scales.stride(0) == 64
    assert offsets.stride(0) == 64
    assert scales[1].item() == 0.5
    assert offsets[1].item() == -1.25


def test_dquant_owned_store_allocates_compact_records():
    cache = DQuantNPUCache(
        capacity=8,
        stream_length=1024,
        container_bytes=331,
        levels=8,
        device="cpu",
    )
    assert cache.storage_bytes == 8 * 336
    assert cache.record_storage_bytes == cache.storage_bytes


def test_dquant_attention_defaults_match_paper_layout():
    config = codec.DQuantCodecConfig.for_attention(8, 128)
    assert config.stream_length == 1024
    assert config.key_container_bytes == 331
    assert config.value_container_bytes == 231
    assert config.key_levels == 8
    assert config.value_levels == 6
    assert config.page_bytes(128) == (128 * 336, 128 * 236)
    assert config.storage_bits_per_value() == pytest.approx(2.234375)
    assert config.storage_bytes_with_recent_window(128) == 128 * 4096
    assert config.storage_bytes_with_recent_window(4096) < 4096 * 4096


@pytest.mark.skipif(not torch.npu.is_available(), reason="Ascend NPU is unavailable")
def test_rans_decode_npu_matches_cpu_reference():
    table = codec.frequencies_from_probabilities([0.28, 0.22, 0.16, 0.12, 0.10, 0.07, 0.04, 0.01])
    expected = [
        [0, 1, 2, 3, 4, 5, 6, 7] * 16,
        [3, 3, 2, 2, 1, 1, 0, 0] * 16,
        [7, 0, 6, 1, 5, 2, 4, 3] * 16,
    ]
    container_bytes = 96
    payloads = []
    for symbols in expected:
        encoded = codec._rans_encode(symbols, table)
        assert len(encoded) <= container_bytes
        payloads.append(encoded + bytes(container_bytes - len(encoded)))
    payload_tensor = torch.tensor([list(payload) for payload in payloads], dtype=torch.uint8, device="npu")
    actual = dquant.rans_decode_npu(payload_tensor, len(expected[0]), *_build_decode_lut(table)).cpu().tolist()
    assert actual == expected


@pytest.mark.skipif(not torch.npu.is_available(), reason="Ascend NPU is unavailable")
def test_rans_encode_npu_matches_cpu_reference():
    table = codec.frequencies_from_probabilities([0.28, 0.22, 0.16, 0.12, 0.10, 0.07, 0.04, 0.01])
    symbols = [
        [0, 1, 2, 3, 4, 5, 6, 7] * 16,
        [3, 3, 2, 2, 1, 1, 0, 0] * 16,
        [7, 0, 6, 1, 5, 2, 4, 3] * 16,
    ]
    container_bytes = 96
    symbol_tensor = torch.tensor(symbols, dtype=torch.int16, device="npu")
    frequencies = torch.tensor(table.frequencies, dtype=torch.int32, device="npu")
    cumulative = torch.tensor(table.cumulative[:-1], dtype=torch.int32, device="npu")
    payloads, status = dquant.rans_encode_npu(symbol_tensor, container_bytes, frequencies, cumulative, table.total)

    assert status.cpu().tolist() == [0] * len(symbols)
    expected = []
    for stream in symbols:
        encoded = codec._rans_encode(stream, table)
        assert len(encoded) <= container_bytes
        expected.append(encoded + bytes(container_bytes - len(encoded)))
    actual = payloads.cpu().tolist()
    assert [bytes(stream) for stream in actual] == expected


@pytest.mark.skipif(not torch.npu.is_available(), reason="Ascend NPU is unavailable")
@pytest.mark.parametrize("padding", [0, 1, 17, 257])
def test_rans_encoder_overlapping_compaction_matches_reference(padding):
    table = codec.frequencies_from_probabilities([0.28, 0.22, 0.16, 0.12, 0.10, 0.07, 0.04, 0.01])
    generator = torch.Generator().manual_seed(20261026)
    symbols = torch.randint(0, 8, (8, 512), generator=generator).tolist()
    symbols[0] = [0] * 512
    symbols[1] = [7] * 512
    symbols[2] = [0, 7] * 256
    symbols[3] = list(range(8)) * 64
    encoded = [codec._rans_encode(stream, table) for stream in symbols]
    container_bytes = max(map(len, encoded)) + padding
    payloads, status = dquant.rans_encode_npu(
        torch.tensor(symbols, dtype=torch.int16, device="npu"),
        container_bytes,
        torch.tensor(table.frequencies, dtype=torch.int32, device="npu"),
        torch.tensor(table.cumulative[:-1], dtype=torch.int32, device="npu"),
        table.total,
    )
    assert status.cpu().tolist() == [0] * len(symbols)
    expected = [stream + bytes(container_bytes - len(stream)) for stream in encoded]
    assert [bytes(row) for row in payloads.cpu().tolist()] == expected


@pytest.mark.skipif(not torch.npu.is_available(), reason="Ascend NPU is unavailable")
@pytest.mark.parametrize("levels", [6, 8])
def test_npu_quantize_encode_decode_pipeline(levels):
    width = 128
    table = codec.gaussian_order_statistic_table(width, levels)
    rates = torch.tensor(table.rate_bits, dtype=torch.float32, device="npu")
    target_rate = 512.0
    torch.manual_seed(23)
    values = torch.randn((2, width), dtype=torch.float16, device="npu")

    symbols, scales, offsets = dquant.dquant_quantize_npu(values, rates, target_rate)
    cpu_values = values.cpu().float().tolist()
    expected_symbols = []
    expected_scales = []
    expected_offsets = []
    for stream in cpu_values:
        scale = (max(stream) - min(stream)) / (levels - 1)
        offset = min(stream)
        assigned = codec._drift(stream, scale, offset, table, target_rate)
        scale, offset = codec._refit(stream, assigned, scale, offset)
        scale = codec._round_fp16(scale)
        offset = codec._round_fp16(offset)
        assigned = codec._drift(stream, scale, offset, table, target_rate)
        expected_symbols.append(assigned)
        expected_scales.append(scale)
        expected_offsets.append(offset)
    assert symbols.cpu().tolist() == expected_symbols
    assert scales.cpu().tolist() == expected_scales
    assert offsets.cpu().tolist() == expected_offsets

    frequency = torch.tensor(table.frequencies, dtype=torch.int32, device="npu")
    cumulative = torch.tensor(table.cumulative[:-1], dtype=torch.int32, device="npu")
    container_bytes = 96
    payloads, overflow = dquant.rans_encode_npu(symbols, container_bytes, frequency, cumulative, table.total)
    decoded_symbols = dquant.rans_decode_npu(payloads, width, *_build_decode_lut(table))

    assert overflow.cpu().tolist() == [0, 0]
    assert torch.equal(decoded_symbols, symbols)
    symbol_rates = rates[symbols.long()].sum(dim=1)
    assert bool(torch.all(symbol_rates <= target_rate).item())
    reconstructed = dquant.dquant_dequantize_npu(decoded_symbols, scales, offsets)
    assert bool(torch.isfinite(reconstructed).all().item())
    assert float((values.float() - reconstructed).square().mean().item()) < 0.5


@pytest.mark.skipif(not torch.npu.is_available(), reason="Ascend NPU is unavailable")
@pytest.mark.parametrize("levels", [6, 8])
def test_constant_stream_never_selects_padding_symbols(levels):
    table = codec.gaussian_order_statistic_table(128, levels)
    rates = torch.tensor(table.rate_bits, dtype=torch.float32, device="npu")
    values = torch.zeros((2, 128), dtype=torch.float16, device="npu")
    symbols, scales, offsets = dquant.dquant_quantize_npu(values, rates, 512.0)
    assert symbols.min().item() >= 0
    assert symbols.max().item() < levels
    expected = min(range(levels), key=table.rate_bits.__getitem__)
    assert torch.all(symbols == expected).item()
    assert torch.isfinite(scales).all().item()
    assert torch.isfinite(offsets).all().item()


@pytest.mark.skipif(not torch.npu.is_available(), reason="Ascend NPU is unavailable")
def test_rans_overflow_and_invalid_symbol_return_status_without_device_fault():
    table = codec.frequencies_from_probabilities([0.5, 0.5])
    symbols = torch.tensor([[0, 1] * 64, [0] * 127 + [99]], dtype=torch.int16, device="npu")
    frequencies = torch.tensor(table.frequencies, dtype=torch.int32, device="npu")
    cumulative = torch.tensor(table.cumulative[:-1], dtype=torch.int32, device="npu")
    payloads, status = dquant.rans_encode_npu(symbols, 4, frequencies, cumulative, table.total)
    statuses = status.cpu().tolist()
    assert statuses[0] & 1
    assert statuses[1] & 2
    assert payloads.shape == (2, 4)
    assert torch.ones(1, device="npu").cpu().item() == 1


@pytest.mark.skipif(not torch.npu.is_available(), reason="Ascend NPU is unavailable")
def test_npu_slot_cache_round_trip_and_copy():
    dense_cache = torch.zeros((2, 4, 4, 32), dtype=torch.float16, device="npu")
    cache = DQuantNPUCache.from_dense_cache(dense_cache, container_bytes=96, levels=8)
    slots = torch.tensor([1, 3], dtype=torch.int64, device="npu")
    torch.manual_seed(31)
    values = torch.randn((2, 128), dtype=torch.float16, device="npu")
    cache.write(slots, values)

    expected = cache.read(slots)
    cache.copy_slots(torch.tensor([[1, 2], [3, 4]], dtype=torch.int64, device="npu"))
    actual = cache.read(torch.tensor([2, 4], dtype=torch.int64, device="npu"))

    assert torch.equal(actual, expected)
    assert cache.storage_bytes == 8 * 256
    assert cache.record_storage_bytes == 8 * 100
    assert cache.bits_per_value == 6.25
    assert torch.equal(dense_cache.view(torch.uint8).view(8, 256)[:, :96], cache.payloads)
