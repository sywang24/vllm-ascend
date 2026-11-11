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

import math

import pytest
import torch

from vllm_ascend.ops.triton.dquant import hadamard_rotate_npu


def _hadamard_reference(rows):
    output = []
    width = len(rows[0])
    for row in rows:
        values = [float(value) for value in row]
        stride = 1
        while stride < width:
            for base in range(0, width, 2 * stride):
                for index in range(base, base + stride):
                    left, right = values[index], values[index + stride]
                    values[index] = left + right
                    values[index + stride] = left - right
            stride *= 2
        output.append([value / math.sqrt(width) for value in values])
    return output


@pytest.mark.parametrize("width", [32, 128, 256])
def test_hadamard_rotate_npu_matches_reference(width):
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU is unavailable")
    torch.manual_seed(17)
    values = torch.randn((3, width), dtype=torch.float16, device="npu")

    actual = hadamard_rotate_npu(values).float().cpu().tolist()
    expected = _hadamard_reference(values.float().cpu().tolist())
    for actual_row, expected_row in zip(actual, expected):
        assert max(abs(left - right) for left, right in zip(actual_row, expected_row)) < 0.02


def test_hadamard_rotate_npu_rejects_non_power_of_two_width():
    values = torch.empty((2, 100), dtype=torch.float16)
    with pytest.raises(ValueError, match="power of two"):
        hadamard_rotate_npu(values)
