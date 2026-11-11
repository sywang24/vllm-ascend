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

"""CPU reference for D-Quant's fixed-container stream format.

This module validates the rate-distortion assignment and rANS byte stream
without importing torch or depending on an Ascend runtime.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist

RANS_STATE_BITS = 32
RANS_LOWER_BOUND = 1 << 23
DRIFT_STEPS = 12


@dataclass(frozen=True)
class RansTable:
    frequencies: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.frequencies or any(freq <= 0 for freq in self.frequencies):
            raise ValueError("rANS frequencies must all be positive")
        total = sum(self.frequencies)
        if total & (total - 1):
            raise ValueError("sum of rANS frequencies must be a power of two")

    @property
    def total(self) -> int:
        return sum(self.frequencies)

    @property
    def scale_bits(self) -> int:
        return self.total.bit_length() - 1

    @property
    def cumulative(self) -> tuple[int, ...]:
        result = [0]
        for frequency in self.frequencies:
            result.append(result[-1] + frequency)
        return tuple(result)

    @property
    def rate_bits(self) -> tuple[float, ...]:
        return tuple(math.log2(self.total / frequency) for frequency in self.frequencies)


@dataclass(frozen=True)
class DQuantStream:
    payload: bytes
    scale: float
    offset: float
    value_count: int


@dataclass(frozen=True)
class DQuantCodecConfig:
    """Per-stream K/V layout and entropy model configuration."""

    stream_length: int
    key_container_bytes: int
    value_container_bytes: int
    key_levels: int = 8
    value_levels: int = 6
    alpha: float = 1.4
    table_total: int = 256
    drift_steps: int = DRIFT_STEPS
    recent_window: int = 128

    @classmethod
    def for_attention(
        cls,
        num_kv_heads: int,
        head_size: int,
        key_container_bytes: int = 331,
        value_container_bytes: int = 231,
        key_levels: int = 8,
        value_levels: int = 6,
        alpha: float = 1.4,
        table_total: int = 256,
        drift_steps: int = DRIFT_STEPS,
        recent_window: int = 128,
    ) -> DQuantCodecConfig:
        """Create the paper's default layout for one token's K/V streams.

        The stream spans all local KV heads. Qwen3-family dense attention uses
        8 heads × 128 channels on the target TP layout, yielding 1024 values.
        Other layouts are accepted, but should be validated against a target
        calibration/evaluation before deployment.
        """
        if num_kv_heads <= 0 or head_size <= 0:
            raise ValueError("num_kv_heads and head_size must be positive")
        return cls(
            stream_length=num_kv_heads * head_size,
            key_container_bytes=key_container_bytes,
            value_container_bytes=value_container_bytes,
            key_levels=key_levels,
            value_levels=value_levels,
            alpha=alpha,
            table_total=table_total,
            drift_steps=drift_steps,
            recent_window=recent_window,
        )

    @property
    def key_record_bytes(self) -> int:
        return ((self.key_container_bytes + 1) & ~1) + 4

    @property
    def value_record_bytes(self) -> int:
        return ((self.value_container_bytes + 1) & ~1) + 4

    def page_bytes(self, block_size: int) -> tuple[int, int]:
        """Return separate K/V physical page sizes for a vLLM block."""
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        return block_size * self.key_record_bytes, block_size * self.value_record_bytes

    def storage_bits_per_value(self) -> float:
        return (self.key_record_bytes + self.value_record_bytes) * 8 / (2 * self.stream_length)

    def storage_bytes_with_recent_window(self, context_length: int) -> int:
        if context_length < 0:
            raise ValueError("context_length cannot be negative")
        compressed_tokens = max(context_length - self.recent_window, 0)
        recent_tokens = context_length - compressed_tokens
        compressed_bytes = compressed_tokens * (self.key_record_bytes + self.value_record_bytes)
        recent_bytes = recent_tokens * 2 * self.stream_length * 2
        return compressed_bytes + recent_bytes

    def __post_init__(self) -> None:
        if self.stream_length <= 1:
            raise ValueError("stream_length must be greater than one")
        if self.key_container_bytes < 4 or self.value_container_bytes < 4:
            raise ValueError("K/V containers must fit the 4-byte rANS state")
        if self.key_levels <= 1 or self.value_levels <= 1:
            raise ValueError("K/V symbol levels must be greater than one")
        if not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        if self.table_total <= 0 or self.table_total & (self.table_total - 1):
            raise ValueError("table_total must be a positive power of two")
        if max(self.key_levels, self.value_levels) > self.table_total:
            raise ValueError("table_total must be at least as large as each symbol alphabet")
        if self.drift_steps < 0:
            raise ValueError("drift_steps cannot be negative")
        if self.recent_window < 0:
            raise ValueError("recent_window cannot be negative")


class DQuantCodec:
    """Configured encoder/decoder for fixed-size key and value streams."""

    def __init__(self, config: DQuantCodecConfig):
        self.config = config
        self.key_table = gaussian_order_statistic_table(
            config.stream_length, config.key_levels, config.alpha, config.table_total
        )
        self.value_table = gaussian_order_statistic_table(
            config.stream_length, config.value_levels, config.alpha, config.table_total
        )

    def encode_key(self, values: Sequence[float]) -> DQuantStream:
        return self._encode(values, self.key_table, self.config.key_container_bytes)

    def encode_value(self, values: Sequence[float]) -> DQuantStream:
        return self._encode(values, self.value_table, self.config.value_container_bytes)

    def decode_key(self, stream: DQuantStream) -> list[float]:
        return self._decode(stream, self.key_table)

    def decode_value(self, stream: DQuantStream) -> list[float]:
        return self._decode(stream, self.value_table)

    def _encode(self, values: Sequence[float], table: RansTable, container_bytes: int) -> DQuantStream:
        if len(values) != self.config.stream_length:
            raise ValueError(f"expected a stream of {self.config.stream_length} values, got {len(values)}")
        return encode_stream(values, table, container_bytes, self.config.drift_steps)

    def _decode(self, stream: DQuantStream, table: RansTable) -> list[float]:
        if stream.value_count != self.config.stream_length:
            raise ValueError(f"expected a stream of {self.config.stream_length} values, got {stream.value_count}")
        return decode_stream(stream, table)


class DQuantPagedCache:
    """CPU reference for fixed-stride, slot-addressed compressed KV storage.

    Each slot stores separate fixed-size K/V payloads and FP16 scale/offset
    pairs. This models paged-cache addressing and block copies; it is not an
    Ascend device cache or an attention backend.
    """

    METADATA_BYTES_PER_STREAM = 4

    def __init__(self, codec: DQuantCodec, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.codec = codec
        self.capacity = capacity
        self._keys = bytearray(capacity * codec.config.key_container_bytes)
        self._values = bytearray(capacity * codec.config.value_container_bytes)
        self._key_metadata = bytearray(capacity * self.METADATA_BYTES_PER_STREAM)
        self._value_metadata = bytearray(capacity * self.METADATA_BYTES_PER_STREAM)
        self._initialized = bytearray(capacity)

    @property
    def bytes_per_slot(self) -> int:
        config = self.codec.config
        return config.key_container_bytes + config.value_container_bytes + 2 * self.METADATA_BYTES_PER_STREAM

    @property
    def storage_bytes(self) -> int:
        return self.capacity * self.bytes_per_slot

    @property
    def bits_per_value(self) -> float:
        return self.bytes_per_slot * 8 / (2 * self.codec.config.stream_length)

    def write(self, slot: int, key: Sequence[float], value: Sequence[float]) -> None:
        self._validate_slot(slot)
        key_stream = self.codec.encode_key(key)
        value_stream = self.codec.encode_value(value)
        key_metadata = struct.pack("<ee", key_stream.scale, key_stream.offset)
        value_metadata = struct.pack("<ee", value_stream.scale, value_stream.offset)

        key_bytes = self.codec.config.key_container_bytes
        value_bytes = self.codec.config.value_container_bytes
        key_start = slot * key_bytes
        value_start = slot * value_bytes
        metadata_start = slot * self.METADATA_BYTES_PER_STREAM
        self._keys[key_start : key_start + key_bytes] = key_stream.payload
        self._values[value_start : value_start + value_bytes] = value_stream.payload
        self._key_metadata[metadata_start : metadata_start + len(key_metadata)] = key_metadata
        self._value_metadata[metadata_start : metadata_start + len(value_metadata)] = value_metadata
        self._initialized[slot] = 1

    def read(self, slot: int) -> tuple[list[float], list[float]]:
        self._validate_slot(slot)
        if not self._initialized[slot]:
            raise ValueError(f"cache slot {slot} has not been written")
        key_bytes = self.codec.config.key_container_bytes
        value_bytes = self.codec.config.value_container_bytes
        metadata_bytes = self.METADATA_BYTES_PER_STREAM
        key_start = slot * key_bytes
        value_start = slot * value_bytes
        metadata_start = slot * metadata_bytes
        key_scale, key_offset = struct.unpack_from("<ee", self._key_metadata, metadata_start)
        value_scale, value_offset = struct.unpack_from("<ee", self._value_metadata, metadata_start)
        key_stream = DQuantStream(
            bytes(self._keys[key_start : key_start + key_bytes]),
            key_scale,
            key_offset,
            self.codec.config.stream_length,
        )
        value_stream = DQuantStream(
            bytes(self._values[value_start : value_start + value_bytes]),
            value_scale,
            value_offset,
            self.codec.config.stream_length,
        )
        return self.codec.decode_key(key_stream), self.codec.decode_value(value_stream)

    def copy_slots(self, src_to_dst: Sequence[tuple[int, int]]) -> None:
        """Copy slot records with snapshot semantics, including overlapping copies."""
        pairs = list(src_to_dst)
        destinations = set()
        for src, dst in pairs:
            self._validate_slot(src)
            self._validate_slot(dst)
            if not self._initialized[src]:
                raise ValueError(f"source cache slot {src} has not been written")
            if dst in destinations:
                raise ValueError(f"destination cache slot {dst} appears more than once")
            destinations.add(dst)

        key_bytes = self.codec.config.key_container_bytes
        value_bytes = self.codec.config.value_container_bytes
        metadata_bytes = self.METADATA_BYTES_PER_STREAM
        snapshots = []
        for src, dst in pairs:
            key_start = src * key_bytes
            value_start = src * value_bytes
            metadata_start = src * metadata_bytes
            snapshots.append(
                (
                    dst,
                    bytes(self._keys[key_start : key_start + key_bytes]),
                    bytes(self._values[value_start : value_start + value_bytes]),
                    bytes(self._key_metadata[metadata_start : metadata_start + metadata_bytes]),
                    bytes(self._value_metadata[metadata_start : metadata_start + metadata_bytes]),
                )
            )
        for dst, key, value, key_metadata, value_metadata in snapshots:
            key_start = dst * key_bytes
            value_start = dst * value_bytes
            metadata_start = dst * metadata_bytes
            self._keys[key_start : key_start + key_bytes] = key
            self._values[value_start : value_start + value_bytes] = value
            self._key_metadata[metadata_start : metadata_start + metadata_bytes] = key_metadata
            self._value_metadata[metadata_start : metadata_start + metadata_bytes] = value_metadata
            self._initialized[dst] = 1

    def copy_blocks(self, src_to_dst: Sequence[tuple[int, int]], block_size: int) -> None:
        """Copy complete cache pages using vLLM's source/destination order."""
        if block_size <= 0 or self.capacity % block_size:
            raise ValueError("block_size must be positive and divide cache capacity")
        num_blocks = self.capacity // block_size
        pairs = list(src_to_dst)
        destinations = set()
        for src, dst in pairs:
            if src < 0 or src >= num_blocks or dst < 0 or dst >= num_blocks:
                raise IndexError(f"cache block mapping ({src}, {dst}) is outside capacity")
            if dst in destinations:
                raise ValueError(f"destination cache block {dst} appears more than once")
            destinations.add(dst)
            first_slot = src * block_size
            last_slot = first_slot + block_size
            if not all(self._initialized[first_slot:last_slot]):
                raise ValueError(f"source cache block {src} contains an unwritten slot")

        slot_pairs = [
            (src * block_size + offset, dst * block_size + offset) for src, dst in pairs for offset in range(block_size)
        ]
        self.copy_slots(slot_pairs)

    def _validate_slot(self, slot: int) -> None:
        if slot < 0 or slot >= self.capacity:
            raise IndexError(f"cache slot {slot} is outside capacity {self.capacity}")


def frequencies_from_probabilities(probabilities: Sequence[float], total: int = 256) -> RansTable:
    """Quantize a probability model to positive integer rANS frequencies."""
    if total <= 0 or total & (total - 1):
        raise ValueError("rANS total must be a positive power of two")
    if len(probabilities) == 0 or len(probabilities) > total:
        raise ValueError("probability table must have between 1 and total entries")
    if any(not math.isfinite(prob) or prob <= 0 for prob in probabilities):
        raise ValueError("probabilities must be finite and positive")

    norm = sum(probabilities)
    available = total - len(probabilities)
    exact = [prob / norm * available for prob in probabilities]
    base = [1 + int(value) for value in exact]
    remainder = total - sum(base)
    order = sorted(range(len(exact)), key=lambda idx: exact[idx] - int(exact[idx]), reverse=True)
    for idx in order[:remainder]:
        base[idx] += 1
    return RansTable(tuple(base))


def sharpen_probabilities(probabilities: Sequence[float], alpha: float = 1.4) -> list[float]:
    """Sharpen an analytic symbol model before converting it to rANS counts."""
    if not probabilities or any(not math.isfinite(prob) or prob <= 0 for prob in probabilities):
        raise ValueError("probabilities must be finite and positive")
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    sharpened = [prob**alpha for prob in probabilities]
    normalizer = sum(sharpened)
    return [prob / normalizer for prob in sharpened]


def gaussian_order_statistic_probabilities(
    stream_length: int,
    levels: int,
    alpha: float = 1.4,
) -> list[float]:
    """Estimate affine-grid symbol probabilities from Gaussian order stats.

    Blom plotting positions approximate the expected minimum and maximum of a
    standard-normal stream. Midpoint decision boundaries then integrate the
    Gaussian density over each affine quantization cell. This is a practical
    analytic model; it avoids collecting calibration data.
    """
    if stream_length <= 1:
        raise ValueError("stream_length must be greater than one")
    if levels <= 1:
        raise ValueError("levels must be greater than one")

    normal = NormalDist()
    first_rank = (0.625) / (stream_length + 0.25)
    last_rank = (stream_length - 0.375) / (stream_length + 0.25)
    minimum = normal.inv_cdf(first_rank)
    maximum = normal.inv_cdf(last_rank)
    step = (maximum - minimum) / (levels - 1)

    boundaries = [minimum + step * (index + 0.5) for index in range(levels - 1)]
    cdf = [normal.cdf(boundary) for boundary in boundaries]
    probabilities = [cdf[0]]
    probabilities.extend(cdf[index] - cdf[index - 1] for index in range(1, len(cdf)))
    probabilities.append(1.0 - cdf[-1])
    return sharpen_probabilities(probabilities, alpha)


def gaussian_order_statistic_table(
    stream_length: int,
    levels: int,
    alpha: float = 1.4,
    total: int = 256,
) -> RansTable:
    """Build the calibration-free rANS table for one stream shape."""
    probabilities = gaussian_order_statistic_probabilities(stream_length, levels, alpha)
    return frequencies_from_probabilities(probabilities, total)


def _rans_encode(symbols: Sequence[int], table: RansTable) -> bytes:
    """Encode symbols in forward order using byte-renormalized rANS."""
    cumulative = table.cumulative
    state = RANS_LOWER_BOUND
    emitted = bytearray()
    scale_bits = table.scale_bits

    for symbol in reversed(symbols):
        if symbol < 0 or symbol >= len(table.frequencies):
            raise ValueError(f"symbol {symbol} is outside the rANS alphabet")
        frequency = table.frequencies[symbol]
        max_state = ((RANS_LOWER_BOUND >> scale_bits) << 8) * frequency
        while state >= max_state:
            emitted.append(state & 0xFF)
            state >>= 8
        state = ((state // frequency) << scale_bits) + (state % frequency) + cumulative[symbol]

    return state.to_bytes(4, "little") + bytes(reversed(emitted))


def _rans_decode(encoded: bytes, value_count: int, table: RansTable) -> list[int]:
    if len(encoded) < 4:
        raise ValueError("rANS stream is shorter than its state")
    state = int.from_bytes(encoded[:4], "little")
    if state < RANS_LOWER_BOUND:
        raise ValueError("invalid rANS initial state")

    cumulative = table.cumulative
    symbols: list[int] = []
    cursor = 4
    mask = table.total - 1
    for _ in range(value_count):
        slot = state & mask
        symbol = next(
            (idx for idx, (start, end) in enumerate(zip(cumulative, cumulative[1:])) if start <= slot < end),
            None,
        )
        if symbol is None:
            raise ValueError("rANS state decoded outside the alphabet")
        symbols.append(symbol)
        state = table.frequencies[symbol] * (state >> table.scale_bits) + slot - cumulative[symbol]
        while state < RANS_LOWER_BOUND:
            if cursor >= len(encoded):
                raise ValueError("truncated rANS stream")
            state = (state << 8) | encoded[cursor]
            cursor += 1
    return symbols


def _assign(values: Sequence[float], scale: float, offset: float, table: RansTable, penalty: float) -> list[int]:
    rates = table.rate_bits
    assignments: list[int] = []
    for value in values:
        best_symbol = 0
        best_cost = math.inf
        for symbol in range(len(rates)):
            reconstructed = offset + scale * symbol
            distortion = (value - reconstructed) ** 2
            cost = distortion + penalty * rates[symbol]
            if cost < best_cost:
                best_cost = cost
                best_symbol = symbol
        assignments.append(best_symbol)
    return assignments


def _drift(
    values: Sequence[float],
    scale: float,
    offset: float,
    table: RansTable,
    target_rate_bits: float,
    steps: int = DRIFT_STEPS,
) -> list[int]:
    rates = table.rate_bits
    if target_rate_bits < len(values) * min(rates):
        raise ValueError("container budget is below the minimum symbol rate")
    distortions = [(value - (offset + scale * symbol)) ** 2 for value in values for symbol in range(len(rates))]
    distortion_range = max(distortions) - min(distortions)
    if distortion_range == 0:
        cheapest_symbol = min(range(len(rates)), key=rates.__getitem__)
        return [cheapest_symbol] * len(values)

    assignments = _assign(values, scale, offset, table, 0.0)
    if sum(rates[symbol] for symbol in assignments) <= target_rate_bits:
        return assignments

    high = math.log2(8.0 * distortion_range)
    low = high - 40.0
    for _ in range(steps):
        midpoint = (low + high) / 2
        penalty = 2.0**midpoint
        assignments = _assign(values, scale, offset, table, penalty)
        rate = sum(rates[symbol] for symbol in assignments)
        if rate <= target_rate_bits:
            high = midpoint
        else:
            low = midpoint
    return _assign(values, scale, offset, table, 2.0**high)


def _refit(
    values: Sequence[float], symbols: Sequence[int], initial_scale: float, initial_offset: float
) -> tuple[float, float]:
    mean_symbol = sum(symbols) / len(symbols)
    mean_value = sum(values) / len(values)
    variance = sum((symbol - mean_symbol) ** 2 for symbol in symbols)
    if variance == 0:
        return initial_scale, initial_offset
    covariance = sum((symbol - mean_symbol) * (value - mean_value) for symbol, value in zip(symbols, values))
    scale = max(0.0, covariance / variance)
    return scale, mean_value - scale * mean_symbol


def _round_fp16(value: float) -> float:
    try:
        return struct.unpack("<e", struct.pack("<e", value))[0]
    except OverflowError as exc:
        raise ValueError("D-Quant scale or offset does not fit in FP16") from exc


def encode_stream(
    values: Sequence[float],
    table: RansTable,
    container_bytes: int,
    drift_steps: int = DRIFT_STEPS,
) -> DQuantStream:
    """Quantize one token stream into an exactly fixed-size rANS container."""
    if not values:
        raise ValueError("D-Quant streams cannot be empty")
    if container_bytes < 4:
        raise ValueError("container must fit the 4-byte rANS state")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("D-Quant input values must be finite")

    minimum = min(values)
    maximum = max(values)
    scale = (maximum - minimum) / (len(table.frequencies) - 1) if len(table.frequencies) > 1 else 0.0
    offset = minimum
    rate_budget = container_bytes * 8 - RANS_STATE_BITS
    symbols = _drift(values, scale, offset, table, rate_budget, drift_steps)
    scale, offset = _refit(values, symbols, scale, offset)
    scale, offset = _round_fp16(scale), _round_fp16(offset)
    symbols = _drift(values, scale, offset, table, rate_budget, drift_steps)

    encoded = _rans_encode(symbols, table)
    penalty = 1.0
    while len(encoded) > container_bytes and penalty < 2**32:
        penalty *= 2
        symbols = _assign(values, scale, offset, table, penalty)
        encoded = _rans_encode(symbols, table)
    if len(encoded) > container_bytes:
        raise ValueError("container is too small to hold the constrained rANS stream")

    return DQuantStream(
        payload=encoded + bytes(container_bytes - len(encoded)),
        scale=scale,
        offset=offset,
        value_count=len(values),
    )


def decode_stream(stream: DQuantStream, table: RansTable) -> list[float]:
    symbols = _rans_decode(stream.payload, stream.value_count, table)
    return [stream.offset + stream.scale * symbol for symbol in symbols]


def hadamard_rotate(values: Sequence[float], block_size: int) -> list[float]:
    """Apply an orthonormal Walsh-Hadamard transform to fixed-size blocks."""
    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("Hadamard block size must be a positive power of two")
    if len(values) % block_size:
        raise ValueError("input length must be divisible by the Hadamard block size")

    output: list[float] = []
    normalization = math.sqrt(block_size)
    for start in range(0, len(values), block_size):
        block = [float(value) for value in values[start : start + block_size]]
        stride = 1
        while stride < block_size:
            for base in range(0, block_size, stride * 2):
                for index in range(base, base + stride):
                    left = block[index]
                    right = block[index + stride]
                    block[index] = left + right
                    block[index + stride] = left - right
            stride *= 2
        output.extend(value / normalization for value in block)
    return output


def remove_windowed_key_mean(
    keys: Sequence[Sequence[float]], window_size: int
) -> tuple[list[list[float]], list[list[float]]]:
    """Center key channels independently in fixed token windows."""
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if not keys:
        return [], []
    width = len(keys[0])
    if width == 0 or any(len(row) != width for row in keys):
        raise ValueError("key rows must have one consistent, non-zero width")

    centered: list[list[float]] = []
    means: list[list[float]] = []
    for start in range(0, len(keys), window_size):
        window = keys[start : start + window_size]
        mean = [sum(row[channel] for row in window) / len(window) for channel in range(width)]
        means.append(mean)
        centered.extend([[value - mean[channel] for channel, value in enumerate(row)] for row in window])
    return centered, means
