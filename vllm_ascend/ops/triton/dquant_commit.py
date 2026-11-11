# SPDX-License-Identifier: Apache-2.0
"""Commit a validated entropy-coded stream to its scheduler-owned slot."""

from vllm.triton_utils import tl, triton


@triton.jit
def _commit_records_kernel(
    slots_ptr,
    payload_ptr,
    scales_ptr,
    offsets_ptr,
    symbols_ptr,
    storage_ptr,
    initialized_ptr,
    ROW_BYTES: tl.constexpr,
    CONTAINER: tl.constexpr,
    METADATA: tl.constexpr,
    SYMBOL_BYTES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.load(slots_ptr + row).to(tl.int64)
    columns = tl.arange(0, BLOCK)
    payload = tl.load(payload_ptr + row * CONTAINER + columns, columns < CONTAINER, 0)
    tl.store(storage_ptr + slot * ROW_BYTES + columns, payload, columns < CONTAINER)
    scale = tl.load(scales_ptr + row).to(tl.uint16, bitcast=True)
    offset = tl.load(offsets_ptr + row).to(tl.uint16, bitcast=True)
    bits = tl.where(columns < 2, scale, offset).to(tl.uint32)
    byte = (bits >> ((columns % 2) * 8)).to(tl.uint8)
    tl.store(storage_ptr + slot * ROW_BYTES + METADATA + columns, byte, columns < 4)
    if SYMBOL_BYTES:
        low = tl.load(symbols_ptr + row * (2 * SYMBOL_BYTES) + 2 * columns, columns < SYMBOL_BYTES, 0).to(tl.uint8)
        high = tl.load(symbols_ptr + row * (2 * SYMBOL_BYTES) + 2 * columns + 1, columns < SYMBOL_BYTES, 0).to(tl.uint8)
        tl.store(
            storage_ptr + slot * ROW_BYTES + ROW_BYTES - SYMBOL_BYTES + columns,
            low | (high << 4),
            columns < SYMBOL_BYTES,
        )
    tl.store(initialized_ptr + slot, True)


def commit_records_npu(cache, slots, payloads, scales, offsets, symbols):
    """Caller must reject overflow before invoking this writer."""
    if slots.numel() == 0:
        return
    symbol_bytes = cache.stream_length // 2 if cache.symbols is not None else 0
    _commit_records_kernel[(slots.numel(),)](
        slots,
        payloads,
        scales,
        offsets,
        symbols,
        cache.storage,
        cache.initialized,
        ROW_BYTES=cache.row_stride_bytes,
        CONTAINER=cache.container_bytes,
        METADATA=cache.metadata_offset,
        SYMBOL_BYTES=symbol_bytes,
        BLOCK=triton.next_power_of_2(max(cache.container_bytes, symbol_bytes, 4)),
    )
