# SPDX-License-Identifier: Apache-2.0
"""Optional Ascend910B3 decoder with all temporary symbols kept in UB."""

import ctypes
from pathlib import Path

import torch
from vllm.logger import init_logger


class DQuantNativeAttention:
    def __init__(self, library):
        self.library = ctypes.CDLL(str(library))
        self.launch = self.library.launch_dquant_attention
        self.launch.argtypes = (
            [ctypes.c_uint32, ctypes.c_void_p]
            + [ctypes.c_void_p] * 10
            + [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_float]
        )
        self.launch.restype = None
        self.logger = init_logger("vllm." + __name__)
        self._logged = False

    @staticmethod
    def supports(query, block_tables, chunk_tokens, tile_tokens):
        return (
            query.dtype == torch.bfloat16
            and query.shape[1:] == (12, 256)
            and query.is_contiguous()
            and block_tables.dtype == torch.int32
            and block_tables.is_contiguous()
            and chunk_tokens == tile_tokens == 64
        )

    def __call__(
        self,
        rotated_query,
        query,
        key_storage,
        value_storage,
        block_tables,
        key_lut,
        value_lut,
        output,
        lse,
        compressed,
        scale,
    ):
        if not self._logged:
            self.logger.info("D-Quant native single-stream Ascend C attention active: library=%s", self.library._name)
            self._logged = True
        tensors = (
            rotated_query,
            query,
            key_storage,
            value_storage,
            block_tables,
            key_lut,
            value_lut,
            output,
            lse,
            compressed,
        )
        self.launch(
            40,
            torch.npu.current_stream(query.device).npu_stream,
            *(tensor.data_ptr() for tensor in tensors),
            query.shape[0],
            output.shape[1],
            block_tables.stride(0),
            scale,
        )


def create_dquant_native_attention(spec, device):
    library = Path(__file__).resolve().parent.parent / "libdquant_native.so"
    if (
        device.type != "npu"
        or not library.is_file()
        or spec.num_kv_heads != 2
        or spec.head_size != 256
        or spec.block_size != 128
        or spec.dtype != torch.bfloat16
        or spec.dquant_key_window != 64
        or spec.mean_stripe_bytes != 16
        or spec.key_record_bytes != 176
        or spec.value_record_bytes != 114
        or spec.dquant_key_container_bytes != 155
        or spec.dquant_value_container_bytes != 109
        or spec.dquant_symbol_cache
    ):
        return None
    if torch.npu.get_device_name(device) != "Ascend910B3":
        return None
    with torch.npu.device(device):
        return DQuantNativeAttention(library)
