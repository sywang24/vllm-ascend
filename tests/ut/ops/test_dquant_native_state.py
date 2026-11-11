# SPDX-License-Identifier: Apache-2.0
"""Check every decoded symbol and final integer state against a CPU reference."""

import ctypes
from pathlib import Path

import pytest
import torch
import torch_npu  # noqa: F401

import vllm_ascend
from vllm_ascend.ops.triton.dquant_fused import make_packed_decode_lut
from vllm_ascend.quantization.methods.kv_cache.dquant_npu_cache import DQuantNPUCache


def integer_reference(records, packed_lut, container):
    data = records.cpu().long()
    lut = packed_lut.cpu().long()
    rows = torch.arange(data.shape[0])
    state = data[:, 0] | data[:, 1] << 8 | data[:, 2] << 16 | data[:, 3] << 24
    cursor = torch.full_like(state, 4)
    symbols = torch.empty((data.shape[0], 512), dtype=torch.int32)
    for channel in range(512):
        slot = state & 255
        entry = lut[slot]
        symbols[:, channel] = (entry & 255).int()
        state = ((entry >> 8) & 511) * (state >> 8) + slot - ((entry >> 17) & 255)
        refill = (state < (1 << 23)) & (cursor < container)
        byte = data[rows, cursor.clamp_max(container - 1)]
        state = torch.where(refill, (state << 8) | byte, state)
        cursor += refill.long()
    return symbols, state.int(), cursor.int()


@pytest.mark.skipif(not torch.npu.is_available(), reason="Ascend NPU unavailable")
@pytest.mark.parametrize("rows", [1, 65, 129, 1024])
@pytest.mark.parametrize("levels,container,stride", [(8, 155, 176), (6, 109, 114)])
def test_native_integer_state(rows, levels, container, stride):
    torch.npu.set_device(0)
    if torch.npu.get_device_name() != "Ascend910B3":
        pytest.skip("Native specialization requires Ascend910B3")
    library = Path(vllm_ascend.__file__).parent.parent / "build_dquant_native/lib/libdquant_native_probe.so"
    if not library.is_file():
        pytest.skip("Build csrc/dquant with -DDQUANT_BUILD_TEST_KERNELS=ON")
    torch.manual_seed(913 + rows + levels)
    records = torch.zeros((rows + 2, stride), dtype=torch.uint8, device="npu")
    cache = DQuantNPUCache(rows + 2, 512, container, levels, storage=records)
    slots = torch.arange(rows, device="npu")
    cache.write(slots, torch.randn((rows, 512), dtype=torch.bfloat16, device="npu"))
    lut = make_packed_decode_lut(cache.table.frequencies, records.device)
    tiles = (rows + 63) // 64
    symbols = torch.full((tiles + 1, 512, 64), 173.0, device="npu")
    states = torch.full((tiles + 1, 2, 64), 173, dtype=torch.int32, device="npu")
    torch.npu.synchronize()
    handle = ctypes.CDLL(str(library))
    launch = handle.launch_dquant_probe
    launch.argtypes = [ctypes.c_uint32, ctypes.c_void_p] + [ctypes.c_void_p] * 4 + [ctypes.c_uint32] * 3
    launch.restype = None

    def call():
        launch(
            40,
            torch.npu.current_stream().npu_stream,
            records.data_ptr(),
            lut.data_ptr(),
            symbols.data_ptr(),
            states.data_ptr(),
            rows,
            stride,
            container,
        )

    def check():
        expected_symbols, expected_state, expected_cursor = integer_reference(records[:rows], lut, container)
        actual_symbols = symbols[:tiles].cpu().permute(0, 2, 1).reshape(-1, 512)[:rows]
        actual_states = states[:tiles].cpu().permute(1, 0, 2).reshape(2, -1)[:, :rows]
        torch.testing.assert_close(actual_symbols.int(), expected_symbols, atol=0, rtol=0)
        torch.testing.assert_close(actual_states[0], expected_state, atol=0, rtol=0)
        torch.testing.assert_close(actual_states[1], expected_cursor, atol=0, rtol=0)
        assert bool((symbols[tiles:].cpu() == 173.0).all())
        assert bool((states[tiles:].cpu() == 173).all())

    call()
    torch.npu.synchronize()
    check()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="global"):
        call()
    cache.write(slots, torch.randn((rows, 512), dtype=torch.bfloat16, device="npu"))
    symbols.fill_(173.0)
    states.fill_(173)
    graph.replay()
    torch.npu.synchronize()
    check()
