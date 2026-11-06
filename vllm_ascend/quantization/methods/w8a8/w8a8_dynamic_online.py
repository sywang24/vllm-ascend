# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import torch
import torch_npu

from .w8a8_dynamic import AscendW8A8DynamicLinearMethod

_WEIGHT_QUANT_CHUNK_ROWS = 4096


class AscendW8A8DynamicOnlineLinearMethod(AscendW8A8DynamicLinearMethod):
    """Load floating-point weights and quantize each output channel on NPU."""

    def get_weight(self, input_size: int, output_size: int, params_dtype: torch.dtype) -> dict[str, torch.Tensor]:
        if params_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("Online LM head quantization requires float16 or bfloat16 weights.")
        return {"weight": torch.empty(output_size, input_size, dtype=params_dtype)}

    def get_perchannel_param(self, output_size: int, params_dtype: torch.dtype) -> dict[str, torch.Tensor]:
        # npu_dynamic_quant returns FP32 scales; keep them in FP32 for quant_matmul.
        return super().get_perchannel_param(output_size, torch.float32)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data
        quantized_weight = torch.empty_like(weight, dtype=torch.int8)
        weight_scale = torch.empty(weight.shape[0], device=weight.device, dtype=torch.float32)
        # Bound temporary NPU allocations for large vocabulary shards.
        for start in range(0, weight.shape[0], _WEIGHT_QUANT_CHUNK_ROWS):
            end = min(start + _WEIGHT_QUANT_CHUNK_ROWS, weight.shape[0])
            quantized_chunk, scale = torch_npu.npu_dynamic_quant(weight[start:end].contiguous(), dst_type=torch.int8)
            quantized_weight[start:end].copy_(quantized_chunk)
            weight_scale[start:end].copy_(scale)
        layer.weight.data = quantized_weight
        del weight, quantized_weight
        layer.weight_scale.data = weight_scale
        layer.weight_offset.data.zero_()
        super().process_weights_after_loading(layer)
