# SPDX-License-Identifier: Apache-2.0
"""Persistent recent KV and device-side commits for full decode graphs."""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.dquant import (
    _dquant_finalize_encode_kernel,
    _dquant_quantize_kernel,
)


class DQuantBatchSlots:
    """Keep request identities and lengths at addresses shared by all graphs."""

    def __init__(self, capacity, device):
        self.capacity = capacity
        self.owners = {}
        self.cpu = torch.zeros((2, capacity), dtype=torch.int32, pin_memory=True)
        self.device = torch.zeros_like(self.cpu, device=device)

    def release(self, request_ids):
        for request_id in request_ids:
            self.owners.pop(request_id, None)

    def prepare(self, request_ids, active, lengths, capturing=False):
        self.cpu.zero_()
        self.cpu[0].fill_(-1)
        if not capturing:
            self.owners = {key: slot for key, slot in self.owners.items() if key in active}
            free = sorted(set(range(self.capacity)) - set(self.owners.values()), reverse=True)
            for row, request in enumerate(request_ids):
                if request is None or request not in active:
                    continue
                if request not in self.owners:
                    if not free:
                        raise RuntimeError("D-Quant recent pool has no free request slot")
                    self.owners[request] = free.pop()
                self.cpu[0, row] = self.owners[request]
                self.cpu[1, row] = lengths[row]
        # The pinned source is reused on the next step. Keep this one small
        # transfer synchronous rather than racing a future metadata update.
        self.device.copy_(self.cpu)
        return self.device, tuple(self.cpu[0, : len(request_ids)].tolist())


class DQuantRecentPool:
    def __init__(self, capacity, spec, device):
        self.capacity = capacity
        self.stride = spec.recent_capacity + 1
        shape = (capacity, self.stride, spec.num_kv_heads, spec.head_size)
        self.key = torch.empty(shape, dtype=spec.dtype, device=device)
        self.value = torch.empty_like(self.key)
        self.slots = torch.empty((capacity, self.stride), dtype=torch.int64, device=device)
        self.totals = torch.full((capacity,), -1, dtype=torch.int32, device=device)
        self.owners = {}

    def claim(self, request_id, slot=None):
        if request_id in self.owners:
            actual = self.owners[request_id]
            if slot is not None and actual != slot:
                raise RuntimeError("D-Quant request changed its recent-pool slot")
            return actual
        if slot is None:
            free = sorted(set(range(self.capacity)) - set(self.owners.values()))
            if not free:
                raise RuntimeError("D-Quant recent pool capacity exceeded")
            slot = free[0]
        if not 0 <= slot < self.capacity or slot in self.owners.values():
            raise RuntimeError("D-Quant recent pool slot is already owned")
        self.owners[request_id] = slot
        return slot

    def release(self, request_id):
        self.owners.pop(request_id, None)


class DQuantGraphWorkspace:
    """One set of scratch buffers, shared sequentially by attention layers."""

    def __init__(self, capacity, max_length):
        self.capacity = capacity
        self.max_length = max_length
        self.buffers = None

    def get(self, spec, heads, device, chunk):
        layout = (spec.stream_length, spec.head_size, heads, spec.dtype, spec.dquant_key_window, chunk, device)
        if self.buffers is not None:
            if layout != self.layout:
                raise ValueError("Shared D-Quant graph workspace requires one attention layout")
            return self.buffers
        self.layout = layout
        batch, window, width = self.capacity, spec.dquant_key_window, spec.stream_length
        chunks = triton.cdiv(self.max_length, chunk)
        self.buffers = {
            "compressed": torch.empty(batch, dtype=torch.int32, device=device),
            "tail_metadata": torch.empty((2, batch), dtype=torch.int32, device=device),
            "flush": torch.empty(batch, dtype=torch.int32, device=device),
            "parts": torch.empty((batch, chunks, heads, spec.head_size), dtype=torch.float32, device=device),
            "lses": torch.empty((batch, chunks, heads), dtype=torch.float32, device=device),
            "error": torch.zeros((2, batch), dtype=torch.int32, device=device),
        }
        for name, size in (("key", spec.dquant_key_container_bytes), ("value", spec.dquant_value_container_bytes)):
            self.buffers[name] = {
                "values": torch.empty((batch * window, width), dtype=spec.dtype, device=device),
                "symbols": torch.empty((batch * window, width), dtype=torch.int16, device=device),
                "scales": torch.empty(batch * window, dtype=torch.float16, device=device),
                "offsets": torch.empty(batch * window, dtype=torch.float16, device=device),
                "payloads": torch.empty((batch * window, size), dtype=torch.uint8, device=device),
                "status": torch.empty(batch * window, dtype=torch.int32, device=device),
            }
        return self.buffers

    def check_errors(self):
        if self.buffers is not None:
            code = int(self.buffers["error"].max().item())
            if code:
                raise RuntimeError(f"D-Quant decode graph rejected invalid state/record (code={code})")


@triton.jit
def _decode_metadata(
    batch_ptr,
    totals_ptr,
    compressed_ptr,
    tail_ptr,
    flush_ptr,
    error_ptr,
    CAPACITY: tl.constexpr,
    RECENT: tl.constexpr,
    WINDOW: tl.constexpr,
    STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
    BATCH: tl.constexpr,
):
    row = tl.arange(0, BLOCK)
    slot = tl.load(batch_ptr + row, row < BATCH, -1)
    total = tl.load(batch_ptr + CAPACITY + row, row < BATCH, 0)
    active = (row < BATCH) & (slot >= 0) & (slot < CAPACITY) & (total > 0)
    prefix = total - 1
    previous = tl.load(totals_ptr + slot, active, -1)
    stale = active & (prefix > 0) & (previous != prefix)
    previous_error = tl.load(error_ptr + row, row < BATCH, 0)
    tl.store(error_ptr + row, previous_error | (stale.to(tl.int32) * 2), row < BATCH)
    compressed = tl.maximum(prefix - RECENT, 0) // WINDOW * WINDOW
    after = tl.maximum(total - RECENT, 0) // WINDOW * WINDOW
    tl.store(compressed_ptr + row, tl.where(active, compressed, 0), row < BATCH)
    tl.store(tail_ptr + row, tl.where(active, slot * STRIDE, 0), row < BATCH)
    tl.store(tail_ptr + CAPACITY + row, tl.where(active, prefix - compressed, 0), row < BATCH)
    tl.store(flush_ptr + row, (active & ~stale & (after > compressed)).to(tl.int32), row < BATCH)


@triton.jit
def _append_current(
    key_ptr,
    value_ptr,
    slots_ptr,
    current_k,
    current_v,
    current_slots,
    batch_ptr,
    tail_ptr,
    CAPACITY: tl.constexpr,
    STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.load(batch_ptr + row)
    total = tl.load(batch_ptr + CAPACITY + row)
    if (slot >= 0) & (slot < CAPACITY) & (total > 0):
        length = tl.load(tail_ptr + CAPACITY + row)
        dims = tl.arange(0, BLOCK)
        offset = (slot * STRIDE + length) * WIDTH + dims
        tl.store(key_ptr + offset, tl.load(current_k + row * WIDTH + dims, dims < WIDTH, 0), dims < WIDTH)
        tl.store(value_ptr + offset, tl.load(current_v + row * WIDTH + dims, dims < WIDTH, 0), dims < WIDTH)
        tl.store(slots_ptr + slot * STRIDE + length, tl.load(current_slots + row))


@triton.jit
def _prepare_encode(
    pool_ptr,
    values_ptr,
    batch_ptr,
    flush_ptr,
    STRIDE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    KV_HEADS: tl.constexpr,
    WINDOW: tl.constexpr,
    CENTER: tl.constexpr,
    IO_DTYPE: tl.constexpr,
    LOG2_HEAD: tl.constexpr,
):
    row = tl.program_id(0)
    token = tl.program_id(1)
    if tl.load(flush_ptr + row):
        slot = tl.load(batch_ptr + row)
        dims = tl.arange(0, HEAD_SIZE)
        tokens = tl.arange(0, WINDOW)
        for head in tl.static_range(KV_HEADS):
            ptr = pool_ptr + ((slot * STRIDE + token) * KV_HEADS + head) * HEAD_SIZE + dims
            values = tl.load(ptr).to(tl.float32)
            if CENTER:
                offsets = ((slot * STRIDE + tokens[:, None]) * KV_HEADS + head) * HEAD_SIZE + dims[None, :]
                means = (tl.sum(tl.load(pool_ptr + offsets).to(tl.float32), 0) / WINDOW).to(IO_DTYPE)
                values = (values - means.to(tl.float32)).to(IO_DTYPE).to(tl.float32)
            for stage in tl.static_range(LOG2_HEAD):
                bit = 1 << stage
                partner = tl.gather(values, dims ^ bit, 0)
                values = tl.where((dims & bit) == 0, values + partner, partner - values)
            values *= 1.0 / tl.sqrt(float(HEAD_SIZE))
            out = ((row * WINDOW + token) * KV_HEADS + head) * HEAD_SIZE + dims
            tl.store(values_ptr + out, values)


@triton.jit
def _quantize_if_flushing(
    values,
    rates,
    symbols,
    scales,
    offsets,
    flush,
    WINDOW: tl.constexpr,
    WIDTH: tl.constexpr,
    LEVELS: tl.constexpr,
    TARGET_RATE_BITS: tl.constexpr,
):
    if tl.load(flush + tl.program_id(0) // WINDOW):
        _dquant_quantize_kernel(
            values, rates, symbols, scales, offsets, WIDTH, LEVELS, WIDTH, TARGET_RATE_BITS, 12, False
        )


@triton.jit
def _encode_if_flushing(
    values,
    rates,
    scales,
    offsets,
    frequencies,
    cumulatives,
    symbols,
    payloads,
    status,
    flush,
    WINDOW: tl.constexpr,
    WIDTH: tl.constexpr,
    LEVELS: tl.constexpr,
    CONTAINER: tl.constexpr,
    BYTE_BLOCK: tl.constexpr,
):
    if tl.load(flush + tl.program_id(0) // WINDOW):
        _dquant_finalize_encode_kernel(
            values,
            rates,
            scales,
            offsets,
            frequencies,
            cumulatives,
            symbols,
            payloads,
            status,
            WIDTH,
            LEVELS,
            WIDTH,
            CONTAINER * 8 - 32,
            12,
            CONTAINER,
            BYTE_BLOCK,
            8,
            1 << 23,
            False,
        )


@triton.jit
def _commit_packed_window(
    payloads,
    scales,
    offsets,
    status,
    storage,
    block_tables,
    batch_ptr,
    compressed,
    flush,
    error,
    key_pool,
    CONTAINER: tl.constexpr,
    ROW_BYTES: tl.constexpr,
    METADATA_OFFSET: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    WINDOW: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    KV_HEADS: tl.constexpr,
    POOL_STRIDE: tl.constexpr,
    MEAN_OFFSET: tl.constexpr,
    STRIPE: tl.constexpr,
    CENTER: tl.constexpr,
    BYTE_BLOCK: tl.constexpr,
    IO_DTYPE: tl.constexpr,
    CAPACITY: tl.constexpr,
):
    row = tl.program_id(0)
    if tl.load(flush + row):
        tokens = tl.arange(0, WINDOW)
        failed = tl.max(tl.load(status + row * WINDOW + tokens), 0) != 0
        if failed:
            if tl.program_id(1) == 0:
                tl.store(error + CAPACITY + row, 1)
        else:
            # Two even-sized records form aligned words, including the
            # 114-byte Value format. Byte stores introduce replay-unsafe
            # cross-core locks in the Ascend Triton launcher.
            token = tl.program_id(1) * 2
            position = tl.load(compressed + row) + token
            block = tl.load(block_tables + row * TABLE_STRIDE + position // BLOCK_SIZE)
            slot = block * BLOCK_SIZE + position % BLOCK_SIZE
            words = tl.arange(0, BYTE_BLOCK)
            packed = tl.full((BYTE_BLOCK,), 0, tl.uint32)
            if CENTER:
                owner = tl.load(batch_ptr + row)
                dims = tl.arange(0, HEAD_SIZE)
                addresses = ((owner * POOL_STRIDE + tokens[:, None]) * KV_HEADS) * HEAD_SIZE + dims[None, :]
                mean0 = (tl.sum(tl.load(key_pool + addresses).to(tl.float32), 0) / WINDOW).to(IO_DTYPE)
                mean1 = (tl.sum(tl.load(key_pool + addresses + HEAD_SIZE).to(tl.float32), 0) / WINDOW).to(IO_DTYPE)
            for byte in tl.static_range(4):
                byte_position = words * 4 + byte
                record = byte_position // ROW_BYTES
                local = byte_position % ROW_BYTES
                stream = row * WINDOW + token + record
                valid = byte_position < 2 * ROW_BYTES
                data = tl.load(payloads + stream * CONTAINER + local, valid & (local < CONTAINER), 0).to(tl.uint32)
                scale = tl.load(scales + stream, valid, 0).to(tl.uint16, bitcast=True).to(tl.uint32)
                offset = tl.load(offsets + stream, valid, 0).to(tl.uint16, bitcast=True).to(tl.uint32)
                metadata = scale | (offset << 16)
                shift = tl.maximum(local - METADATA_OFFSET, 0) * 8
                data = tl.where(
                    (local >= METADATA_OFFSET) & (local < METADATA_OFFSET + 4), (metadata >> shift) & 255, data
                )
                if CENTER:
                    mean_byte = (token + record) * STRIPE + local - MEAN_OFFSET
                    channel = tl.maximum(mean_byte // 2, 0)
                    lane = channel % HEAD_SIZE
                    selected = tl.where(channel < HEAD_SIZE, tl.gather(mean0, lane, 0), tl.gather(mean1, lane, 0))
                    bits = selected.to(tl.uint16, bitcast=True).to(tl.uint32)
                    mean_data = (bits >> ((mean_byte % 2) * 8)) & 255
                    data = tl.where((local >= MEAN_OFFSET) & (local < MEAN_OFFSET + STRIPE), mean_data, data)
                packed |= data << (byte * 8)
            tl.store(storage + slot * ROW_BYTES // 4 + words, packed, words < ROW_BYTES // 2)


@triton.jit
def _compact_recent(
    key,
    value,
    slots,
    totals,
    batch_ptr,
    tail_ptr,
    flush,
    CAPACITY: tl.constexpr,
    STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
    WINDOW: tl.constexpr,
    RECENT: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    row = tl.program_id(0)
    channel = tl.program_id(1)
    slot = tl.load(batch_ptr + row)
    total = tl.load(batch_ptr + CAPACITY + row)
    if (slot >= 0) & (slot < CAPACITY) & (total > 0):
        if tl.load(flush + row):
            tokens = tl.arange(0, BLOCK_T)
            dims = channel * 32 + tl.arange(0, 32)
            src = (slot * STRIDE + WINDOW + tokens[:, None]) * WIDTH + dims[None, :]
            dst = (slot * STRIDE + tokens[:, None]) * WIDTH + dims[None, :]
            valid = (tokens[:, None] < RECENT) & (dims[None, :] < WIDTH)
            # Each program owns these channels for the whole tail, so all
            # overlapping source rows are loaded before their destinations.
            kept_key = tl.load(key + src, valid, 0)
            kept_value = tl.load(value + src, valid, 0)
            tl.store(key + dst, kept_key, valid)
            tl.store(value + dst, kept_value, valid)
            if channel == 0:
                kept_slots = tl.load(slots + slot * STRIDE + WINDOW + tokens, tokens < RECENT, 0)
                tl.store(slots + slot * STRIDE + tokens, kept_slots, tokens < RECENT)
        if channel == 0:
            tl.store(totals + slot, total)


def prepare_decode(pool, buffers, batch_metadata, batch, spec):
    _decode_metadata[(1,)](
        batch_metadata,
        pool.totals,
        buffers["compressed"],
        buffers["tail_metadata"],
        buffers["flush"],
        buffers["error"],
        CAPACITY=pool.capacity,
        RECENT=spec.dquant_recent_window,
        WINDOW=spec.dquant_key_window,
        STRIDE=pool.stride,
        BLOCK=triton.next_power_of_2(batch),
        BATCH=batch,
    )


def commit_decode(cache, buffers, batch_metadata, block_tables, updates, batch):
    spec, pool = cache.spec, cache.recent_pool
    window, width = spec.dquant_key_window, spec.stream_length
    dtype = tl.bfloat16 if spec.dtype == torch.bfloat16 else tl.float16
    slots, key, value = updates
    _append_current[(batch,)](
        pool.key,
        pool.value,
        pool.slots,
        key,
        value,
        slots,
        batch_metadata,
        buffers["tail_metadata"],
        CAPACITY=pool.capacity,
        STRIDE=pool.stride,
        WIDTH=width,
        BLOCK=triton.next_power_of_2(width),
    )
    for name, raw, compressed in (("key", pool.key, cache.key), ("value", pool.value, cache.value)):
        scratch = buffers[name]
        center = name == "key"
        _prepare_encode[(batch, window)](
            raw,
            scratch["values"],
            batch_metadata,
            buffers["flush"],
            STRIDE=pool.stride,
            HEAD_SIZE=spec.head_size,
            KV_HEADS=spec.num_kv_heads,
            WINDOW=window,
            CENTER=center,
            IO_DTYPE=dtype,
            num_warps=4,
            LOG2_HEAD=spec.head_size.bit_length() - 1,
        )
        _quantize_if_flushing[(batch * window,)](
            scratch["values"],
            compressed.rates,
            scratch["symbols"],
            scratch["scales"],
            scratch["offsets"],
            buffers["flush"],
            WINDOW=window,
            WIDTH=width,
            LEVELS=compressed.rates.numel(),
            TARGET_RATE_BITS=compressed.container_bytes * 8 - 32,
            num_warps=4,
        )
        _encode_if_flushing[(batch * window,)](
            scratch["values"],
            compressed.rates,
            scratch["scales"],
            scratch["offsets"],
            compressed.frequency_by_symbol,
            compressed.cumulative_by_symbol,
            scratch["symbols"],
            scratch["payloads"],
            scratch["status"],
            buffers["flush"],
            WINDOW=window,
            WIDTH=width,
            LEVELS=compressed.rates.numel(),
            CONTAINER=compressed.container_bytes,
            BYTE_BLOCK=triton.next_power_of_2(compressed.container_bytes),
            num_warps=4,
        )
    # Validate both streams before publishing either record.
    status = buffers["key"]["status"] | buffers["value"]["status"]
    for name, compressed in (("key", cache.key), ("value", cache.value)):
        scratch = buffers[name]
        _commit_packed_window[(batch, window // 2)](
            scratch["payloads"],
            scratch["scales"],
            scratch["offsets"],
            status,
            compressed.storage.flatten().view(torch.int32),
            block_tables,
            batch_metadata,
            buffers["compressed"],
            buffers["flush"],
            buffers["error"],
            pool.key,
            CONTAINER=compressed.container_bytes,
            ROW_BYTES=compressed.storage.shape[1],
            METADATA_OFFSET=compressed.metadata_offset,
            TABLE_STRIDE=block_tables.stride(0),
            BLOCK_SIZE=spec.block_size,
            WINDOW=window,
            HEAD_SIZE=spec.head_size,
            KV_HEADS=spec.num_kv_heads,
            POOL_STRIDE=pool.stride,
            MEAN_OFFSET=cache.key.record_bytes,
            STRIPE=spec.mean_stripe_bytes,
            CENTER=name == "key",
            BYTE_BLOCK=triton.next_power_of_2(compressed.storage.shape[1] // 2),
            IO_DTYPE=dtype,
            CAPACITY=pool.capacity,
            num_warps=4,
        )
    _compact_recent[(batch, triton.cdiv(width, 32))](
        pool.key,
        pool.value,
        pool.slots,
        pool.totals,
        batch_metadata,
        buffers["tail_metadata"],
        buffers["flush"],
        CAPACITY=pool.capacity,
        STRIDE=pool.stride,
        WIDTH=width,
        WINDOW=window,
        RECENT=spec.dquant_recent_window,
        BLOCK_T=triton.next_power_of_2(max(1, spec.dquant_recent_window)),
        num_warps=4,
    )
