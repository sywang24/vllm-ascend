#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# This file is mainly Adapted from vllm-project/vllm/vllm/envs.py
# Copyright 2023 The vLLM team.
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
#

import os
from collections.abc import Callable
from typing import Any

# The begin-* and end* here are used by the documentation generator
# to extract the used env vars.

# begin-env-vars-definition


def _strict_binary_env(name: str, default: str = "0") -> bool:
    value = os.getenv(name, default)
    if value not in {"0", "1"}:
        raise ValueError(f"{name} must be either '0' or '1', got {value!r}")
    return value == "1"


env_variables: dict[str, Callable[[], Any]] = {
    # Validation-only contiguous BF16 cache adapter. Default off; 1/true/yes
    # enable it, other values disable it. Non-sensitive.
    "VLLM_ASCEND_DQUANT_DENSE_REFERENCE": lambda: (
        os.getenv("VLLM_ASCEND_DQUANT_DENSE_REFERENCE", "0").lower() in {"1", "true", "yes"}
    ),
    # Raw recent KV tokens; default 0, integer >= 0, non-sensitive.
    "VLLM_ASCEND_DQUANT_RECENT_WINDOW": lambda: int(os.getenv("VLLM_ASCEND_DQUANT_RECENT_WINDOW", "0")),
    # Key mean window; default 0 (off), positive divisor of block size, non-sensitive.
    "VLLM_ASCEND_DQUANT_KEY_WINDOW": lambda: int(os.getenv("VLLM_ASCEND_DQUANT_KEY_WINDOW", "0")),
    # Experimental single-token streaming KV decode. Default 0 (off); otherwise
    # a positive multiple of Key window (or any positive integer without it).
    # Non-sensitive. Bounds dense KV scratch; does not apply to prefill.
    "VLLM_ASCEND_DQUANT_DECODE_CHUNK_SIZE": lambda: int(os.getenv("VLLM_ASCEND_DQUANT_DECODE_CHUNK_SIZE", "0")),
    # Experimental contiguous rANS output stores; 1 keeps scalar behavior.
    # Allowed values: 1, 16, 32, 64. Non-sensitive; opt in after workload profiling.
    "VLLM_ASCEND_DQUANT_OUTPUT_CHUNK": lambda: int(os.getenv("VLLM_ASCEND_DQUANT_OUTPUT_CHUNK", "1")),
    # Incremental dense decode cache (mirror) cap in compressed tokens per
    # request; 0 disables the mirror and restores full re-reads. Non-sensitive.
    "VLLM_ASCEND_DQUANT_MIRROR_TOKENS": lambda: int(os.getenv("VLLM_ASCEND_DQUANT_MIRROR_TOKENS", "8192")),
    # Opt-in GDN core graph replay for a single non-speculative decode token.
    # Default off; 1/true/yes enable it only with D-Quant. Non-sensitive.
    "VLLM_ASCEND_DQUANT_GDN_GRAPH": lambda: (
        os.getenv("VLLM_ASCEND_DQUANT_GDN_GRAPH", "0").lower() in {"1", "true", "yes"}
    ),
    # Experimental Qwen3.5 GDN decoder segments (up to 3 layers per graph).
    # Default off; 1/true/yes enable with D-Quant and eager execution only.
    # Non-sensitive. Prefill, speculation and multi-token batches use eager.
    "VLLM_ASCEND_DQUANT_SEGMENT_GRAPH": lambda: (
        os.getenv("VLLM_ASCEND_DQUANT_SEGMENT_GRAPH", "0").lower() in {"1", "true", "yes"}
    ),
    # Pinned-host decoded overflow mirror, absolute sequence token limit.
    # Non-sensitive integer >=0, default 0 disables; rounded down to key windows.
    # Tokens beyond the HBM mirror are staged over PCIe instead of re-decoded.
    "VLLM_ASCEND_DQUANT_HOST_MIRROR_TOKENS": lambda: int(os.getenv("VLLM_ASCEND_DQUANT_HOST_MIRROR_TOKENS", "0")),
    # Shared dense K/V workspace for overflow single-token decode. Non-sensitive;
    # default off, 1/true/yes enable with eager D-Quant. Reused per device stream.
    "VLLM_ASCEND_DQUANT_DECODE_WORKSPACE": lambda: (
        os.getenv("VLLM_ASCEND_DQUANT_DECODE_WORKSPACE", "0").lower() in {"1", "true", "yes"}
    ),
    # Experimental Qwen3.5 graph regions across GDN/attention decoder boundaries.
    # Non-sensitive; default off, 1/true/yes enable. Requires both decoder graph
    # switches, eager D-Quant, language-only execution and a single PP stage.
    "VLLM_ASCEND_DQUANT_BOUNDARY_GRAPH": lambda: (
        os.getenv("VLLM_ASCEND_DQUANT_BOUNDARY_GRAPH", "0").lower() in {"1", "true", "yes"}
    ),
    # Experimental Qwen3.5 full-attention decoder pre/post graphs in eager D-Quant.
    # Non-sensitive; default off; 1/true/yes enable, other values disable.
    # Single-token decode only; prefill and unsupported inputs use eager.
    "VLLM_ASCEND_DQUANT_ATTENTION_GRAPH": lambda: (
        os.getenv("VLLM_ASCEND_DQUANT_ATTENTION_GRAPH", "0").lower() in {"1", "true", "yes"}
    ),
    # Fused single-token decode over compressed rANS KV records. The kernel
    # decodes levels on chip and folds affine/rotation into attention, so no
    # dense prefix is materialized. Non-sensitive; default off.
    "VLLM_ASCEND_DQUANT_FUSED_ATTENTION": lambda: (
        os.getenv("VLLM_ASCEND_DQUANT_FUSED_ATTENTION", "0").lower() in {"1", "true", "yes"}
    ),
    # Cache exact quantizer symbols at four bits/value alongside rANS records.
    # Non-sensitive; default off; 1/true/yes enable with fused attention only.
    # The extra bytes are included in physical pages and scheduler allocation.
    "VLLM_ASCEND_DQUANT_SYMBOL_CACHE": lambda: (
        os.getenv("VLLM_ASCEND_DQUANT_SYMBOL_CACHE", "0").lower() in {"1", "true", "yes"}
    ),
    # Compressed tokens decoded per fused-attention program; default 16,
    # a positive multiple of the vector tile. Non-sensitive.
    "VLLM_ASCEND_DQUANT_FUSED_CHUNK_TOKENS": lambda: int(os.getenv("VLLM_ASCEND_DQUANT_FUSED_CHUNK_TOKENS", "16")),
    # Tokens decoded per vector tile in the fused rANS attention kernel.
    # Positive power of two, at least 16; default 16. Non-sensitive.
    "VLLM_ASCEND_DQUANT_FUSED_TILE_TOKENS": lambda: int(os.getenv("VLLM_ASCEND_DQUANT_FUSED_TILE_TOKENS", "16")),
    # Non-sensitive alphabet sizes, integers 2..256; defaults K 8, V 6.
    "VLLM_ASCEND_DQUANT_KEY_LEVELS": lambda: int(os.getenv("VLLM_ASCEND_DQUANT_KEY_LEVELS", "8")),
    "VLLM_ASCEND_DQUANT_VALUE_LEVELS": lambda: int(os.getenv("VLLM_ASCEND_DQUANT_VALUE_LEVELS", "6")),
    # Non-sensitive payload bytes, 0 (paper default) or >=4; rounded to odd bytes.
    "VLLM_ASCEND_DQUANT_KEY_BYTES": lambda: int(os.getenv("VLLM_ASCEND_DQUANT_KEY_BYTES", "0")),
    "VLLM_ASCEND_DQUANT_VALUE_BYTES": lambda: int(os.getenv("VLLM_ASCEND_DQUANT_VALUE_BYTES", "0")),
    # Non-sensitive finite positive Gaussian sharpening exponent; default 1.4.
    "VLLM_ASCEND_DQUANT_ALPHA": lambda: float(os.getenv("VLLM_ASCEND_DQUANT_ALPHA", "1.4")),
    # Experimental D-Quant KV bridge. Default off; 1/true/yes enable it.
    # Non-sensitive. Supports TP with unquantized weights; see DQUANT.md.
    "VLLM_ASCEND_ENABLE_DQUANT": lambda: os.getenv("VLLM_ASCEND_ENABLE_DQUANT", "0").lower() in {"1", "true", "yes"},
    # max compile thread number for package building. Usually, it is set to
    # the number of CPU cores. If not set, the default value is None, which
    # means all number of CPU cores will be used.
    "MAX_JOBS": lambda: os.getenv("MAX_JOBS", None),
    # The build type of the package. It can be one of the following values:
    # Release, Debug, RelWithDebugInfo. If not set, the default value is Release.
    "CMAKE_BUILD_TYPE": lambda: os.getenv("CMAKE_BUILD_TYPE"),
    # Whether to compile custom kernels. If not set, the default value is True.
    # If set to False, the custom kernels will not be compiled.
    # This configuration option should only be set to False when running UT
    # scenarios in an environment without an NPU. Do not set it to False in
    # other scenarios.
    "COMPILE_CUSTOM_KERNELS": lambda: bool(int(os.getenv("COMPILE_CUSTOM_KERNELS", "1"))),
    # The CXX compiler used for compiling the package. If not set, the default
    # value is None, which means the system default CXX compiler will be used.
    "CXX_COMPILER": lambda: os.getenv("CXX_COMPILER", None),
    # The C compiler used for compiling the package. If not set, the default
    # value is None, which means the system default C compiler will be used.
    "C_COMPILER": lambda: os.getenv("C_COMPILER", None),
    # The version of the Ascend chip. It's used for package building.
    # If not set, we will query chip info through `npu-smi`.
    # Please make sure that the version is correct.
    "SOC_VERSION": lambda: os.getenv("SOC_VERSION", None),
    # If set, vllm-ascend will print verbose logs during compilation
    "VERBOSE": lambda: bool(int(os.getenv("VERBOSE", "0"))),
    # The home path for CANN toolkit. If not set, the default value is
    # /usr/local/Ascend/ascend-toolkit/latest
    "ASCEND_HOME_PATH": lambda: os.getenv("ASCEND_HOME_PATH", None),
    # The path for HCCL library, it's used by pyhccl communicator backend. If
    # not set, the default value is libhccl.so.
    "HCCL_SO_PATH": lambda: os.getenv("HCCL_SO_PATH", None),
    # The version of vllm is installed. This value is used for developers who
    # installed vllm from source locally. In this case, the version of vllm is
    # usually changed. For example, if the version of vllm is "0.9.0", but when
    # it's installed from source, the version of vllm is usually set to "0.9.1".
    # In this case, developers need to set this value to "0.9.0" to make sure
    # that the correct package is installed.
    "VLLM_VERSION": lambda: os.getenv("VLLM_VERSION", None),
    # Whether to anbale dynamic EPLB
    "DYNAMIC_EPLB": lambda: os.getenv("DYNAMIC_EPLB", "false").lower(),
    # Control the aclrtMemcpyBatchAsync compile path for KV cache offloading.
    # "1": force enable, "0": force disable, None: auto-detect from CANN headers.
    "VLLM_ASCEND_ENABLE_BATCH_MEMCPY": lambda: os.getenv("VLLM_ASCEND_ENABLE_BATCH_MEMCPY", None),
    # Emit per-layer KVPool ranged transfer audit events. Default: 0 (disabled).
    # Valid values: 0 or 1. This configuration is not sensitive.
    "VLLM_ASCEND_KVPOOL_RANGE_DEBUG": lambda: _strict_binary_env("VLLM_ASCEND_KVPOOL_RANGE_DEBUG"),
    # Override the Unified Buffer (UB) size in KB for Triton kernel tile sizing.
    # 0 (default): auto-detect from device properties, falling back to 192 KB
    # (safe for Ascend 910B/A3). Set to a positive value to override when
    # auto-detection is unavailable or for debugging UB overflow issues.
    "VLLM_ASCEND_ROPE_UB_SIZE_KB": lambda: int(os.getenv("VLLM_ASCEND_ROPE_UB_SIZE_KB") or 0),
}

# end-env-vars-definition


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in env_variables:
        return env_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(env_variables.keys())
