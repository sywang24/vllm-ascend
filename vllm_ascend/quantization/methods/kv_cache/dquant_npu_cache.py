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

"""Slot-addressed NPU storage for fixed-container D-Quant streams."""

import torch

from vllm_ascend.ops.triton.dquant import (
    dquant_dequantize_npu,
    dquant_quantize_encode_npu,
    dquant_quantize_npu,
    rans_decode_npu,
    rans_encode_npu,
)
from vllm_ascend.ops.triton.dquant_commit import commit_records_npu
from vllm_ascend.quantization.methods.kv_cache.dquant_codec import (
    gaussian_order_statistic_table,
)


def _record_views(
    storage: torch.Tensor,
    capacity: int,
    container_bytes: int,
    metadata_offset: int,
    row_stride_bytes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    payloads = torch.as_strided(
        storage,
        size=(capacity, container_bytes),
        stride=(row_stride_bytes, 1),
    )
    metadata_bytes = torch.as_strided(
        storage,
        size=(capacity, 4),
        stride=(row_stride_bytes, 1),
        storage_offset=metadata_offset,
    )
    metadata = metadata_bytes.view(torch.float16)
    return payloads, metadata[:, 0], metadata[:, 1]


class DQuantNPUCache:
    """Experimental fixed-stride store for one K or V stream per token.

    This prototype does not integrate with vLLM page allocation or attention
    kernels. Binding it to an existing dense cache preserves that cache's page
    stride and therefore does not reduce allocated memory.
    """

    def __init__(
        self,
        capacity: int,
        stream_length: int,
        container_bytes: int,
        levels: int,
        device: torch.device | str = "npu",
        alpha: float = 1.4,
        table_total: int = 256,
        storage: torch.Tensor | None = None,
    ) -> None:
        if capacity <= 0 or stream_length <= 0 or container_bytes < 4:
            raise ValueError("capacity, stream_length, and container_bytes must be positive")
        if levels <= 1 or levels > table_total:
            raise ValueError("levels must be greater than one and no larger than table_total")

        self.capacity = capacity
        self.stream_length = stream_length
        self.container_bytes = container_bytes
        self.device = torch.device(device)
        if self.device.type == "npu" and self.device.index is None:
            self.device = torch.device("npu", torch.npu.current_device())
        self.table = gaussian_order_statistic_table(stream_length, levels, alpha, table_total)
        self.minimum_symbol_rate = min(self.table.rate_bits)
        self.rates = torch.tensor(self.table.rate_bits, dtype=torch.float32, device=self.device)
        self.frequency_by_symbol = torch.tensor(self.table.frequencies, dtype=torch.int32, device=self.device)
        self.cumulative_by_symbol = torch.tensor(self.table.cumulative[:-1], dtype=torch.int32, device=self.device)
        self.decode_luts = self._make_decode_luts()
        self.metadata_offset = (container_bytes + 1) & ~1
        self.record_bytes = self.metadata_offset + 4
        if storage is None:
            storage = torch.zeros((capacity, self.record_bytes), dtype=torch.uint8, device=self.device)
            self.row_stride_bytes = self.record_bytes
        else:
            if storage.device != self.device or not storage.is_contiguous():
                raise ValueError("storage must be contiguous and on the cache device")
            if storage.dtype != torch.uint8 or storage.ndim != 2 or storage.shape[0] != capacity:
                raise ValueError("storage must be a uint8 tensor with one row per cache slot")
            if storage.shape[1] < self.record_bytes:
                raise ValueError("storage row is too small for the D-Quant record")
            self.row_stride_bytes = storage.shape[1]
        self.storage = storage
        self.payloads, self.scales, self.offsets = _record_views(
            storage,
            capacity,
            container_bytes,
            self.metadata_offset,
            self.row_stride_bytes,
        )
        self.initialized = torch.zeros((capacity,), dtype=torch.bool, device=self.device)
        self.symbols = None

    @classmethod
    def from_dense_cache(
        cls,
        cache: torch.Tensor,
        container_bytes: int,
        levels: int,
        alpha: float = 1.4,
        table_total: int = 256,
    ) -> "DQuantNPUCache":
        """Bind to a dense cache tensor while preserving its physical page stride."""
        if (
            cache.ndim != 4
            or not cache.is_contiguous()
            or cache.dtype not in (torch.float16, torch.bfloat16, torch.uint8)
        ):
            raise ValueError("cache must be contiguous [blocks, block, heads, dim] fp16/bf16/uint8 storage")
        capacity = cache.shape[0] * cache.shape[1]
        stream_length = cache.shape[2] * cache.shape[3]
        row_stride_bytes = cache[0, 0].numel() * cache.element_size()
        storage = cache.view(torch.uint8).view(capacity, row_stride_bytes)
        return cls(
            capacity=capacity,
            stream_length=stream_length,
            container_bytes=container_bytes,
            levels=levels,
            device=cache.device,
            alpha=alpha,
            table_total=table_total,
            storage=storage,
        )

    @classmethod
    def from_byte_storage(
        cls,
        storage: torch.Tensor,
        capacity: int,
        stream_length: int,
        container_bytes: int,
        levels: int,
        alpha: float = 1.4,
        table_total: int = 256,
    ) -> "DQuantNPUCache":
        """Bind compact records to a worker-allocated flat byte buffer."""
        if storage.ndim != 1 or storage.dtype not in (torch.int8, torch.uint8):
            raise ValueError("storage must be a flat int8 or uint8 byte buffer")
        if not storage.is_contiguous():
            raise ValueError("storage must be contiguous")

        record_bytes = ((container_bytes + 1) & ~1) + 4
        expected_bytes = capacity * record_bytes
        if storage.numel() != expected_bytes:
            raise ValueError(
                f"storage has {storage.numel()} bytes; expected {expected_bytes} "
                f"for {capacity} records of {record_bytes} bytes"
            )
        rows = storage.view(torch.uint8).view(capacity, record_bytes)
        return cls(
            capacity=capacity,
            stream_length=stream_length,
            container_bytes=container_bytes,
            levels=levels,
            device=storage.device,
            alpha=alpha,
            table_total=table_total,
            storage=rows,
        )

    @property
    def storage_bytes(self) -> int:
        """Physical backing allocation in bytes, including unused row padding."""
        return self.capacity * self.row_stride_bytes

    @property
    def record_storage_bytes(self) -> int:
        """Bytes occupied by compact records when allocated at record stride."""
        return self.capacity * self.record_bytes

    @property
    def bits_per_value(self) -> float:
        """Logical record cost, not backing allocation cost when bound."""
        return self.record_bytes * 8 / self.stream_length

    def write(self, slots: torch.Tensor, values: torch.Tensor) -> None:
        # Bounds are validated by the caller from scheduler-owned block tables;
        # skipping the device-side check here avoids a host sync every flush.
        slots = self._validate_slots(slots, check_bounds=False)
        if values.shape != (slots.numel(), self.stream_length) or not values.is_floating_point():
            raise ValueError("values must contain one floating-point stream per slot")
        if values.device != self.device:
            raise ValueError("values must be on the cache device")
        if slots.numel() == 0:
            return

        if self.stream_length == 512 and self.rates.numel() in (6, 8) and self.table.total == 256:
            symbols, scales, offsets, payloads, overflow = dquant_quantize_encode_npu(
                values.contiguous(),
                self.rates,
                self.container_bytes,
                self.frequency_by_symbol,
                self.cumulative_by_symbol,
                self.table.total,
                minimum_symbol_rate=self.minimum_symbol_rate,
            )
        else:
            symbols, scales, offsets = dquant_quantize_npu(
                values.contiguous(),
                self.rates,
                self.container_bytes * 8 - 32,
                minimum_symbol_rate=self.minimum_symbol_rate,
            )
            payloads, overflow = rans_encode_npu(
                symbols,
                self.container_bytes,
                self.frequency_by_symbol,
                self.cumulative_by_symbol,
                self.table.total,
            )
        # Reject an invalid record before it can corrupt subsequent attention.
        # The rate budget includes state bytes, but rANS byte rounding still
        # requires checking the encoder's actual status.
        if bool(overflow.any().item()):
            raise RuntimeError("D-Quant rANS container overflow; refusing to commit invalid KV records")
        commit_records_npu(self, slots, payloads, scales, offsets, symbols)

    def overflow_report(self) -> int:
        """Host-side count of deferred rANS container overflows (one sync)."""
        if getattr(self, "overflow_total", None) is None:
            return 0
        total = int(self.overflow_total.item())
        self.overflow_total.zero_()
        return total

    def read(
        self,
        slots: torch.Tensor,
        output_dtype: torch.dtype = torch.float16,
        require_initialized: bool = True,
    ) -> torch.Tensor:
        slots = self._validate_slots(slots)
        if slots.numel() == 0:
            return torch.empty((0, self.stream_length), dtype=output_dtype, device=self.device)
        if require_initialized and not bool(torch.all(self.initialized[slots]).item()):
            raise ValueError("one or more D-Quant cache slots have not been written")
        symbols = rans_decode_npu(self.payloads[slots].contiguous(), self.stream_length, *self.decode_luts)
        return dquant_dequantize_npu(
            symbols,
            self.scales[slots].contiguous(),
            self.offsets[slots].contiguous(),
            output_dtype=output_dtype,
        )

    def copy_slots(self, src_to_dst: torch.Tensor) -> None:
        if src_to_dst.ndim != 2 or src_to_dst.shape[1] != 2:
            raise ValueError("src_to_dst must have shape [copies, 2]")
        src = self._validate_slots(src_to_dst[:, 0])
        dst = self._validate_slots(src_to_dst[:, 1])
        if dst.numel() == 0:
            return
        if not bool(torch.all(self.initialized[src]).item()):
            raise ValueError("one or more source D-Quant cache slots have not been written")
        if torch.unique(dst).numel() != dst.numel():
            raise ValueError("destination D-Quant cache slots must be unique")

        payloads = self.payloads[src].clone()
        scales = self.scales[src].clone()
        offsets = self.offsets[src].clone()
        self.payloads[dst] = payloads
        self.scales[dst] = scales
        self.offsets[dst] = offsets
        if self.symbols is not None:
            self.symbols[dst] = self.symbols[src].clone()
        self.initialized[dst] = True

    def copy_blocks(self, src_to_dst: torch.Tensor, block_size: int) -> None:
        """Copy complete physical pages using vLLM's [source, destination] map."""
        if block_size <= 0 or self.capacity % block_size:
            raise ValueError("block_size must be positive and divide the cache capacity")
        if src_to_dst.ndim != 2 or src_to_dst.shape[1] != 2:
            raise ValueError("src_to_dst must have shape [copies, 2]")
        if src_to_dst.dtype not in (torch.int32, torch.int64) or src_to_dst.device != self.device:
            raise ValueError("src_to_dst must be an int32/int64 tensor on the cache device")
        num_blocks = self.capacity // block_size
        if src_to_dst.numel() and bool(torch.any((src_to_dst < 0) | (src_to_dst >= num_blocks)).item()):
            raise IndexError("D-Quant cache block is outside capacity")
        src, dst = src_to_dst.to(torch.long).unbind(dim=1)
        if torch.unique(dst).numel() != dst.numel():
            raise ValueError("destination cache blocks must be unique")
        if dst.numel() == 0:
            return

        slots = torch.arange(block_size, device=self.device)
        src_slots = (src[:, None] * block_size + slots).reshape(-1)
        dst_slots = (dst[:, None] * block_size + slots).reshape(-1)
        records = self.storage[src_slots].clone()
        initialized = self.initialized[src_slots].clone()
        self.storage[dst_slots] = records
        self.initialized[dst_slots] = initialized

    def _validate_slots(self, slots: torch.Tensor, check_bounds: bool = True) -> torch.Tensor:
        if slots.ndim != 1 or slots.dtype not in (torch.int32, torch.int64):
            raise ValueError("slots must be a one-dimensional int32 or int64 tensor")
        if slots.device != self.device:
            raise ValueError("slots must be on the cache device")
        if check_bounds and slots.numel() and bool(torch.any((slots < 0) | (slots >= self.capacity)).item()):
            raise IndexError("D-Quant cache slot is outside capacity")
        return slots.to(torch.long)

    def _make_decode_luts(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        symbols: list[int] = []
        frequencies: list[int] = []
        cumulative: list[int] = []
        starts = self.table.cumulative[:-1]
        for symbol, (frequency, start) in enumerate(zip(self.table.frequencies, starts)):
            symbols.extend([symbol] * frequency)
            frequencies.extend([frequency] * frequency)
            cumulative.extend([start] * frequency)
        return tuple(
            torch.tensor(values, dtype=torch.int32, device=self.device) for values in (symbols, frequencies, cumulative)
        )
