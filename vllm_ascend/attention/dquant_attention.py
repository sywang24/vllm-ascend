# SPDX-License-Identifier: Apache-2.0
"""Eager attention bridge for fixed-record D-Quant KV caches.

The optional fused decoder consumes compact records directly in attention.
The compatibility path reconstructs temporary dense K/V tensors for the native
Ascend attention operator, batching single-token decode queries across requests. A bounded decoded-prefix
mirror avoids repeated entropy decoding. Complete key windows leaving a request's raw tail are centered and
encoded once. Means are striped across physical records so page copies retain
all information needed to decode a compressed window.
"""

import logging
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch_npu

from vllm_ascend import envs
from vllm_ascend.core.kv_cache_interface import DQuantAttentionSpec
from vllm_ascend.ops.dquant_native import create_dquant_native_attention
from vllm_ascend.ops.triton.dquant import hadamard_rotate_npu, paged_rans_dequantize_npu
from vllm_ascend.ops.triton.dquant_decode_pool import DQuantRecentPool, commit_decode, prepare_decode
from vllm_ascend.ops.triton.dquant_finalize import (
    finalize_dquant_attention_npu,
    finalize_dquant_batch_attention_npu,
)
from vllm_ascend.ops.triton.dquant_fused import (
    make_packed_decode_lut,
    merge_dquant_attention_partials_npu,
    paged_dquant_attention_npu,
    paged_dquant_batch_attention_npu,
)
from vllm_ascend.ops.triton.dquant_restore import restore_key_means_npu
from vllm_ascend.ops.triton.dquant_symbols import symbol_dquant_attention_npu
from vllm_ascend.quantization.methods.kv_cache.dquant_npu_cache import DQuantNPUCache

_SHORT_PREFIX_PROGRAM_BUDGET = 40


@dataclass
class RecentKVState:
    key: torch.Tensor
    value: torch.Tensor
    slots: torch.Tensor
    length: int = 0
    total: int = 0
    compressed: int = 0
    pool_slot: int = -1


def merge_attention_partials(parts, lses):
    """Merge split-KV outputs without a host loop or dense KV reconstruction."""
    return merge_dquant_attention_partials_npu(parts, lses)


class DQuantDecodeWorkspace:
    """One reusable dense K/V pair per device stream and attention layout."""

    def __init__(self):
        self.buffers = {}

    def get(self, tokens, heads, head_size, dtype, device):
        stream = torch.npu.current_stream(device).npu_stream
        layout = (device, stream, dtype, heads, head_size)
        pair = self.buffers.get(layout)
        if pair is None or pair[0].shape[0] < tokens:
            quantum = 512
            rows = (tokens + quantum - 1) // quantum * quantum
            pair = tuple(torch.empty((rows, heads, head_size), dtype=dtype, device=device) for _ in range(2))
            self.buffers[layout] = pair
        return tuple(t[:tokens] for t in pair)


def install_dquant_decode_workspace(model):
    workspace = DQuantDecodeWorkspace()
    for module in model.modules():
        impl = getattr(module, "impl", None)
        if hasattr(impl, "_dquant_cache"):
            impl._dquant_decode_workspace = workspace
    model._dquant_decode_workspace = workspace
    return workspace


class DQuantAttentionCache:
    def __init__(
        self,
        storage: Sequence[torch.Tensor],
        spec: DQuantAttentionSpec,
        *,
        paged_decode: bool = True,
        decode_chunk_size: int = 0,
        output_chunk: int = 1,
        decode_workspace: DQuantDecodeWorkspace | None = None,
        max_num_seqs: int = 32,
    ) -> None:
        if len(storage) != 2:
            raise ValueError("D-Quant attention requires separate K/V record buffers")
        key, value = storage
        if key.ndim != 3 or value.ndim != 3 or key.shape[:2] != value.shape[:2]:
            raise ValueError("D-Quant buffers must have shape [blocks, block_size, record_bytes]")
        if key.shape[1:] != (spec.block_size, spec.key_record_bytes):
            raise ValueError("D-Quant key buffer does not match its cache spec")
        if value.shape[1:] != (spec.block_size, spec.value_record_bytes):
            raise ValueError("D-Quant value buffer does not match its cache spec")
        if spec.head_size <= 0 or spec.head_size & (spec.head_size - 1):
            raise ValueError("D-Quant rotation requires a power-of-two head size")
        self.spec = spec
        self.paged_decode = paged_decode
        self.decode_workspace = decode_workspace
        self.set_output_chunk(output_chunk)
        self.set_decode_chunk_size(decode_chunk_size)
        self.recent: dict[str, RecentKVState] = {}
        self.recent_pool = DQuantRecentPool(max_num_seqs, spec, key.device) if spec.uses_window_state else None
        self._requested_pool_slots = {}
        self._tail_metadata_cpu = torch.empty((2, max_num_seqs), dtype=torch.int32, pin_memory=True)
        self._tail_metadata = torch.empty((2, max_num_seqs), dtype=torch.int32, device=key.device)
        self._compressed_cpu = torch.empty(max_num_seqs, dtype=torch.int32, pin_memory=True)
        self._compressed_lengths = torch.empty(max_num_seqs, dtype=torch.int32, device=key.device)
        self._flush_scratch = None
        # Incremental dense mirror of the decoded compressed prefix per request.
        # Turns the per-decode-step full rANS re-read (O(prefix)) into an
        # O(64) decode of newly flushed records. Single-token decode appends
        # its raw tail in the same allocation instead of copying the prefix.
        self.mirror: dict[str, tuple[int, int, torch.Tensor, torch.Tensor]] = {}
        self.mirror_tail_end: dict[str, int] = {}
        self.host_mirror: dict[str, tuple[int, int, torch.Tensor, torch.Tensor]] = {}
        self.host_mirror_peak_bytes = 0
        self.mirror_cap = int(envs.VLLM_ASCEND_DQUANT_MIRROR_TOKENS)
        self.host_mirror_cap = int(envs.VLLM_ASCEND_DQUANT_HOST_MIRROR_TOKENS)
        if self.host_mirror_cap < 0:
            raise ValueError("D-Quant host mirror token limit must be nonnegative")
        self.fused_decode = bool(envs.VLLM_ASCEND_DQUANT_FUSED_ATTENTION)
        self.short_prefix_program_budget = 0
        if (
            self.fused_decode
            and key.device.type == "npu"
            and spec.num_kv_heads == 2
            and spec.head_size == 256
            and spec.dquant_key_window == 64
            and torch.npu.get_device_name(key.device) == "Ascend910B3"
        ):
            # Paired 12-head tests favor smaller tiles until the program count
            # exceeds this measured budget; longer prefixes retain 64 tokens.
            self.short_prefix_program_budget = _SHORT_PREFIX_PROGRAM_BUDGET
        self._fused_decode_logged = False
        if self.fused_decode:
            if spec.num_kv_heads != 2 or not spec.uses_window_state or not paged_decode:
                raise ValueError("Fused D-Quant attention requires two local KV heads and paged window state")
            self.mirror_cap = self.host_mirror_cap = 0
        capacity = key.shape[0] * spec.block_size
        self.key = DQuantNPUCache(
            capacity,
            spec.stream_length,
            spec.dquant_key_container_bytes,
            spec.dquant_key_levels,
            alpha=spec.dquant_alpha,
            device=key.device,
            storage=key.view(capacity, spec.key_record_bytes),
        )
        self.value = DQuantNPUCache(
            capacity,
            spec.stream_length,
            spec.dquant_value_container_bytes,
            spec.dquant_value_levels,
            alpha=spec.dquant_alpha,
            device=value.device,
            storage=value.view(capacity, spec.value_record_bytes),
        )
        self.means = self.key.storage[:, self.key.record_bytes : self.key.record_bytes + spec.mean_stripe_bytes]
        if spec.dquant_symbol_cache:
            if not self.fused_decode:
                raise ValueError("D-Quant symbol cache requires fused attention")
            self.key.symbols = self.key.storage[:, -spec.symbol_bytes :]
            self.value.symbols = self.value.storage[:, -spec.symbol_bytes :]
        self.fused_key_lut = self.fused_value_lut = None
        if self.fused_decode:
            self.fused_key_lut = make_packed_decode_lut(self.key.table.frequencies, key.device)
            self.fused_value_lut = make_packed_decode_lut(self.value.table.frequencies, value.device)
        self.native_attention = create_dquant_native_attention(spec, key.device) if self.fused_decode else None

    def release(self, request_ids: Sequence[str]) -> None:
        for request_id in request_ids:
            self.recent.pop(request_id, None)
            if self.recent_pool is not None:
                self.recent_pool.release(request_id)
            self.mirror.pop(request_id, None)
            self.mirror_tail_end.pop(request_id, None)
            self.host_mirror.pop(request_id, None)

    def bind_recent_slots(self, request_ids, slots, prefixes):
        self._requested_pool_slots = dict(zip(request_ids, slots))
        window = self.spec.dquant_key_window or 1
        for request, slot, prefix in zip(request_ids, slots, prefixes):
            if not prefix:
                continue
            state = self.recent.get(request)
            if state is None:
                state = self._new_recent_state(request)
                self.recent[request] = state
            state.total = prefix
            state.compressed = max(0, prefix - self.spec.dquant_recent_window) // window * window
            state.length = prefix - state.compressed

    def _new_recent_state(self, request_id):
        pool = self.recent_pool
        slot = pool.claim(request_id, self._requested_pool_slots.get(request_id))
        return RecentKVState(pool.key[slot], pool.value[slot], pool.slots[slot], pool_slot=slot)

    def graph_decode(self, query, block_tables, batch_metadata, scale, output, updates, workspace):
        if query.dtype != self.spec.dtype or self.recent_pool is None:
            raise ValueError("D-Quant graph query dtype must match its recent pool")
        batch = query.shape[0]
        if batch > self.recent_pool.capacity:
            raise ValueError("D-Quant decode graph batch exceeds its request pool capacity")
        chunk = int(envs.VLLM_ASCEND_DQUANT_FUSED_CHUNK_TOKENS)
        tile = int(envs.VLLM_ASCEND_DQUANT_FUSED_TILE_TOKENS)
        buffers = workspace.get(self.spec, query.shape[1], query.device, chunk)
        prepare_decode(self.recent_pool, buffers, batch_metadata, batch, self.spec)
        parts, lses, chunk = paged_dquant_batch_attention_npu(
            query,
            block_tables,
            buffers["compressed"][:batch],
            scale,
            self.key.storage,
            self.key.scales,
            self.key.offsets,
            self.fused_key_lut,
            self.value.storage,
            self.value.scales,
            self.value.offsets,
            self.fused_value_lut,
            block_size=self.spec.block_size,
            head_size=self.spec.head_size,
            num_kv_heads=self.spec.num_kv_heads,
            key_container_bytes=self.key.container_bytes,
            value_container_bytes=self.value.container_bytes,
            key_window=self.spec.dquant_key_window,
            mean_offset=self.key.record_bytes,
            stripe_bytes=self.spec.mean_stripe_bytes,
            chunk_tokens=chunk,
            tile_tokens=tile,
            compressed_max=workspace.max_length,
            partials=(buffers["parts"][:batch], buffers["lses"][:batch]),
            native_backend=self.native_attention,
        )
        pool = self.recent_pool
        finalize_dquant_batch_attention_npu(
            parts,
            lses,
            query,
            pool.key.flatten(0, 1),
            pool.value.flatten(0, 1),
            buffers["compressed"][:batch],
            scale,
            (updates[1], updates[2]),
            buffers["tail_metadata"][:, :batch],
            chunk,
            output=output,
        )
        commit_decode(self, buffers, batch_metadata, block_tables, updates, batch)
        return output

    def set_output_chunk(self, size: int) -> None:
        if size not in (1, 16, 32, 64):
            raise ValueError("D-Quant output chunk must be 1, 16, 32 or 64")
        self.output_chunk = size

    def set_decode_chunk_size(self, size: int) -> None:
        window = self.spec.dquant_key_window or 1
        if not isinstance(size, int) or size < 0 or size % window:
            raise ValueError("Decode chunk size must be nonnegative and divisible by the Key window")
        self.decode_chunk_size = size

    def _mirror_rows(self, target: int) -> int:
        # Bound unused dense storage to one small growth quantum per request.
        quantum = max(512, self.spec.dquant_key_window or 1)
        return min(self.mirror_cap, (target + quantum - 1) // quantum * quantum)

    def _mirror_buffer_rows(self, rows: int) -> int:
        # Reserve the raw tail and current token beside the decoded prefix.
        return rows + (self.spec.recent_capacity + 1 if self.spec.uses_window_state else 0)

    def _single_decode_kv(self, block_table, prefix, dtype, request_id, current):
        state = self.recent.get(request_id)
        window = self.spec.dquant_key_window or 1
        cap = self.mirror_cap // window * window
        if not self.paged_decode or state is None or not state.compressed:
            return None
        if state.compressed > cap:
            return self._single_decode_overflow(block_table, prefix, dtype, request_id, current)
        # read() refreshes newly compressed windows before replacing the raw tail.
        self.read(block_table, prefix, dtype, request_id, include_recent=False)
        mirror = self.mirror.get(request_id)
        if mirror is None:
            return None
        filled, _, key, value = mirror
        if key.dtype != dtype or key.shape[0] < prefix + 1:
            return None
        # A preceding decode already materialized the raw tail at its absolute
        # sequence positions. read() replaces newly compressed windows in place.
        start = max(filled, min(prefix, self.mirror_tail_end.get(request_id, filled)))
        if start < prefix:
            key[start:prefix].copy_(state.key[start - filled : state.length])
            value[start:prefix].copy_(state.value[start - filled : state.length])
        key[prefix : prefix + 1].copy_(current[0])
        value[prefix : prefix + 1].copy_(current[1])
        self.mirror_tail_end[request_id] = prefix + 1
        return key[: prefix + 1], value[: prefix + 1]

    def _single_decode_overflow(self, block_table, prefix, dtype, request_id, current):
        mirror = self.mirror.get(request_id)
        state = self.recent[request_id]
        cap = self.mirror_cap // (self.spec.dquant_key_window or 1) * (self.spec.dquant_key_window or 1)
        if self.decode_workspace is None or mirror is None or mirror[0] != cap or mirror[2].dtype != dtype:
            return None
        if state.total != prefix or state.compressed > prefix:
            return None
        num_blocks = (prefix + self.spec.block_size - 1) // self.spec.block_size
        if block_table.ndim != 1 or block_table.numel() < num_blocks:
            raise ValueError("Block table does not cover the D-Quant sequence")
        key, value = self.decode_workspace.get(
            prefix + 1, self.spec.num_kv_heads, self.spec.head_size, dtype, block_table.device
        )
        for destination, source in zip((key, value), mirror[2:]):
            destination[:cap].copy_(source[:cap])
        self._read_overflow(
            block_table,
            cap,
            state.compressed,
            dtype,
            request_id,
            destination=(key[cap : state.compressed], value[cap : state.compressed]),
        )
        key[state.compressed : prefix].copy_(state.key[: state.length])
        value[state.compressed : prefix].copy_(state.value[: state.length])
        key[prefix : prefix + 1].copy_(current[0])
        value[prefix : prefix + 1].copy_(current[1])
        return key, value

    def read_compressed_range(self, block_table, start, end, dtype):
        """Restore an aligned compressed range without reading the full prefix."""
        window = self.spec.dquant_key_window or 1
        if start < 0 or end < start or start % window or end % window:
            raise ValueError("Compressed range must be nonnegative and aligned to Key windows")
        shape = (end - start, self.spec.num_kv_heads, self.spec.head_size)

        def decode(cache):
            return paged_rans_dequantize_npu(
                cache.storage,
                block_table,
                end - start,
                self.spec.block_size,
                cache.container_bytes,
                cache.stream_length,
                cache.scales,
                cache.offsets,
                cache.decode_luts,
                dtype,
                start_token=start,
                output_chunk=self.output_chunk,
            ).reshape(shape)

        key, value = decode(self.key), decode(self.value)
        if end == start:
            return key, value
        key, value = hadamard_rotate_npu(key), hadamard_rotate_npu(value)
        if self.spec.dquant_key_window:
            key = self._restore_key_means(key, block_table, start)
        return key, value

    def _restore_key_means(self, key, block_table, start):
        window = self.spec.dquant_key_window
        if key.shape[0] <= window:
            return restore_key_means_npu(
                key,
                self.key.storage,
                block_table,
                start_token=start,
                block_size=self.spec.block_size,
                window=window,
                mean_offset=self.key.record_bytes,
                stripe_bytes=self.spec.mean_stripe_bytes,
                mean_dtype=self.spec.dtype,
            )
        positions = torch.arange(start, start + key.shape[0], device=block_table.device)
        slots = block_table[positions // self.spec.block_size].long() * self.spec.block_size
        slots = slots + positions % self.spec.block_size
        mean_bytes = self.means[slots].reshape(-1, window * self.spec.mean_stripe_bytes)
        width = self.spec.stream_length * torch.empty((), dtype=self.spec.dtype).element_size()
        means = mean_bytes[:, :width].contiguous().view(self.spec.dtype)
        means = means.view(-1, self.spec.num_kv_heads, self.spec.head_size).repeat_interleave(window, 0)
        return (key.float() + means.float()).to(key.dtype)

    def _read_overflow(self, block_table, start, end, dtype, request_id, destination=None):
        window = self.spec.dquant_key_window or 1
        limit = min(self.host_mirror_cap, self.key.capacity) // window * window
        if request_id is None or limit <= start:
            decoded = self.read_compressed_range(block_table, start, end, dtype)
            if destination is None:
                return decoded
            for output, source in zip(destination, decoded):
                output.copy_(source)
            return destination
        target = min(end, limit)
        mirror = self.host_mirror.get(request_id)
        if mirror is None or mirror[0] != start or mirror[1] > target or mirror[2].dtype != dtype:
            shape = (limit - start, self.spec.num_kv_heads, self.spec.head_size)
            mirror = (
                start,
                start,
                torch.empty(shape, dtype=dtype, pin_memory=True),
                torch.empty(shape, dtype=dtype, pin_memory=True),
            )
        _, filled, host_key, host_value = mirror
        if filled < target:
            key, value = self.read_compressed_range(block_table, filled, target, dtype)
            host_key[filled - start : target - start].copy_(key, non_blocking=True)
            host_value[filled - start : target - start].copy_(value, non_blocking=True)
        self.host_mirror[request_id] = (start, target, host_key, host_value)
        self.host_mirror_peak_bytes = max(self.host_mirror_peak_bytes, self.host_mirror_storage_bytes)
        # D2H and H2D use the current NPU stream; no CPU read or mutation of
        # in-flight pinned data occurs between these ordered transfers.
        if destination is None:
            key = host_key[: target - start].to(device=block_table.device, non_blocking=True)
            value = host_value[: target - start].to(device=block_table.device, non_blocking=True)
        else:
            key, value = destination
            key[: target - start].copy_(host_key[: target - start], non_blocking=True)
            value[: target - start].copy_(host_value[: target - start], non_blocking=True)
        if target < end:
            extra_key, extra_value = self.read_compressed_range(block_table, target, end, dtype)
            if destination is None:
                key = torch.cat((key, extra_key))
                value = torch.cat((value, extra_value))
            else:
                key[target - start :].copy_(extra_key)
                value[target - start :].copy_(extra_value)
        return key, value

    @property
    def host_mirror_storage_bytes(self):
        return sum(
            tensor.numel() * tensor.element_size()
            for _, _, key, value in self.host_mirror.values()
            for tensor in (key, value)
        )

    def _stream_decode_attention(self, query, block_table, seq_len, scale, request_id, current):
        """Single-query attention with bounded dense KV scratch and online LSE merge."""
        state = self.recent.get(request_id)
        if self.spec.uses_window_state and (state is None or state.total != seq_len):
            raise ValueError("D-Quant request length does not match its recent state")
        compressed = state.compressed if state is not None else seq_len
        num_blocks = (seq_len + self.spec.block_size - 1) // self.spec.block_size
        if block_table.ndim != 1 or block_table.numel() < num_blocks:
            raise ValueError("Block table does not cover the D-Quant sequence")
        merged_out = merged_lse = None

        def consume(key, value):
            nonlocal merged_out, merged_lse
            part, lse = torch_npu.npu_fused_infer_attention_score(
                query=query.contiguous(),
                key=key.contiguous(),
                value=value.contiguous(),
                input_layout="TND",
                num_heads=query.shape[1],
                num_key_value_heads=self.spec.num_kv_heads,
                scale=scale,
                actual_seq_lengths=[1],
                actual_seq_lengths_kv=[key.shape[0]],
                sparse_mode=0,
                softmax_lse_flag=True,
            )
            part = part.float().view(query.shape[1], query.shape[2])
            lse = lse.float().reshape(query.shape[1])
            if merged_out is None:
                merged_out, merged_lse = part, lse
            else:
                # Existing Ascend attention-update operator performs the stable
                # logsumexp weighting; retain its accumulator in FP32.
                merged_out, merged_lse = torch_npu.npu_attention_update(
                    [merged_lse, lse],
                    [merged_out, part],
                    1,
                )

        for start in range(0, compressed, self.decode_chunk_size):
            end = min(compressed, start + self.decode_chunk_size)
            key, value = self.read_compressed_range(block_table, start, end, query.dtype)
            consume(key, value)
            del key, value
        # Tail and current token stay raw and never enter the compressed reader.
        if state is not None:
            for start in range(0, state.length, self.decode_chunk_size):
                end = min(state.length, start + self.decode_chunk_size)
                consume(state.key[start:end].to(query.dtype), state.value[start:end].to(query.dtype))
        if current is not None:
            consume(*current)
        if merged_out is None:
            raise ValueError("Streaming attention requires at least one KV token")
        return merged_out.reshape_as(query).to(query.dtype)

    def _fused_decode_attention(self, query, block_table, seq_len, scale, request_id, current, *, output=None):
        """Decode compressed KV directly into attention, without dense K/V."""
        state = self.recent.get(request_id)
        if (
            not self.fused_decode
            or state is None
            or not self.spec.uses_window_state
            or self.spec.num_kv_heads != 2
            or state.total != seq_len
            or not state.compressed
            or self.key.scales.dtype != torch.float16
            or self.spec.dtype != query.dtype
        ):
            return None
        tile_tokens = int(envs.VLLM_ASCEND_DQUANT_FUSED_TILE_TOKENS)
        chunk_tokens = int(envs.VLLM_ASCEND_DQUANT_FUSED_CHUNK_TOKENS)
        if tile_tokens < 16 or tile_tokens & (tile_tokens - 1):
            raise ValueError("Fused token tile size must be a power of two and at least 16")
        if chunk_tokens < tile_tokens or chunk_tokens % tile_tokens:
            raise ValueError("Fused chunk size must be a positive multiple of the token tile")
        if (
            self.short_prefix_program_budget
            and query.shape[1] == 12
            and chunk_tokens == tile_tokens == 64
            and not self.spec.dquant_symbol_cache
        ):
            for candidate in (16, 32):
                if state.compressed <= self.short_prefix_program_budget * candidate:
                    chunk_tokens = tile_tokens = candidate
                    break
        if self.spec.dquant_symbol_cache:
            parts, lses = symbol_dquant_attention_npu(
                query,
                hadamard_rotate_npu(query),
                self.key.symbols,
                self.value.symbols,
                self.key.scales,
                self.key.offsets,
                self.value.scales,
                self.value.offsets,
                block_table,
                self.key.storage,
                scale=scale,
                compressed=state.compressed,
                block_size=self.spec.block_size,
                key_window=self.spec.dquant_key_window,
                mean_offset=self.key.record_bytes,
                stripe_bytes=self.spec.mean_stripe_bytes,
                chunk_tokens=chunk_tokens,
            )
        else:
            parts, lses = self._rans_attention_partials(
                query, block_table, state.compressed, scale, chunk_tokens, tile_tokens
            )
        return finalize_dquant_attention_npu(
            parts,
            lses,
            query,
            state.key[: state.length],
            state.value[: state.length],
            scale,
            current,
            output=output,
        )

    def _batched_fused_decode_attention(
        self,
        query,
        block_tables,
        seq_lens,
        scale,
        output,
        request_ids,
        updates,
    ):
        """Decode independent one-token requests with two batched launches."""
        count = len(request_ids)
        if count < 2 or updates is None or self.spec.dquant_symbol_cache:
            return None
        states = [self.recent.get(request_id) for request_id in request_ids]
        if any(
            state is None or state.total != seq_len - 1 or not state.compressed
            for state, seq_len in zip(states, seq_lens)
        ):
            return None
        compressed_lengths = [state.compressed for state in states]
        if any(length <= 0 or length % (self.spec.dquant_key_window or 1) for length in compressed_lengths):
            return None
        block_tables = block_tables[:count]
        if block_tables.shape[1] * self.spec.block_size < max(compressed_lengths):
            return None
        tile_tokens = int(envs.VLLM_ASCEND_DQUANT_FUSED_TILE_TOKENS)
        chunk_tokens = int(envs.VLLM_ASCEND_DQUANT_FUSED_CHUNK_TOKENS)
        if tile_tokens < 16 or tile_tokens & (tile_tokens - 1):
            raise ValueError("Fused token tile size must be a power of two and at least 16")
        if chunk_tokens < tile_tokens or chunk_tokens % tile_tokens:
            raise ValueError("Fused chunk size must be a positive multiple of the token tile")
        maximum = max(compressed_lengths)
        if self.short_prefix_program_budget and query.shape[1] == 12 and chunk_tokens == tile_tokens == 64:
            for candidate in (16, 32):
                if maximum <= self.short_prefix_program_budget * candidate:
                    chunk_tokens = tile_tokens = candidate
                    break
        self._compressed_cpu[:count] = torch.tensor(compressed_lengths, dtype=torch.int32)
        self._compressed_lengths[:count].copy_(self._compressed_cpu[:count])
        compressed = self._compressed_lengths[:count]
        tail_lengths = [state.length for state in states]
        pool = self.recent_pool
        if query.dtype != pool.key.dtype or any(state.pool_slot < 0 for state in states):
            raise ValueError("Batched D-Quant decode requires recent-pool views with the query dtype")
        self._tail_metadata_cpu[0, :count] = torch.tensor([state.pool_slot * pool.stride for state in states])
        self._tail_metadata_cpu[1, :count] = torch.tensor(tail_lengths)
        self._tail_metadata.copy_(self._tail_metadata_cpu)
        tail_key, tail_value = pool.key.flatten(0, 1), pool.value.flatten(0, 1)
        tail_metadata = self._tail_metadata[:, :count]
        current = (updates[1][:count], updates[2][:count])
        parts, lses, chunk_tokens = self._rans_batch_attention_partials(
            query,
            block_tables,
            compressed,
            maximum,
            scale,
            chunk_tokens,
            tile_tokens,
        )
        result = finalize_dquant_batch_attention_npu(
            parts,
            lses,
            query,
            tail_key,
            tail_value,
            compressed,
            scale,
            current,
            tail_metadata,
            chunk_tokens,
            output=output,
        )
        if not self._fused_decode_logged:
            logging.getLogger(__name__).info(
                "D-Quant batched fused rANS attention active: requests=%d, max_compressed=%d",
                count,
                maximum,
            )
            self._fused_decode_logged = True
        return result

    def _rans_batch_attention_partials(
        self,
        query,
        block_tables,
        compressed,
        maximum,
        scale,
        chunk_tokens,
        tile_tokens,
    ):
        if self.native_attention is not None and self.native_attention.supports(query, block_tables, 64, 64):
            chunk_tokens = tile_tokens = 64
        return paged_dquant_batch_attention_npu(
            query,
            block_tables,
            compressed,
            scale,
            self.key.storage,
            self.key.scales,
            self.key.offsets,
            self.fused_key_lut,
            self.value.storage,
            self.value.scales,
            self.value.offsets,
            self.fused_value_lut,
            block_size=self.spec.block_size,
            head_size=self.spec.head_size,
            num_kv_heads=self.spec.num_kv_heads,
            key_container_bytes=self.key.container_bytes,
            value_container_bytes=self.value.container_bytes,
            key_window=self.spec.dquant_key_window,
            mean_offset=self.key.record_bytes,
            stripe_bytes=self.spec.mean_stripe_bytes,
            chunk_tokens=chunk_tokens,
            tile_tokens=min(chunk_tokens, tile_tokens),
            compressed_max=maximum,
            native_backend=self.native_attention,
        )

    def _rans_attention_partials(self, query, block_table, compressed, scale, chunk_tokens, tile_tokens):
        return paged_dquant_attention_npu(
            query,
            block_table,
            compressed,
            scale,
            self.key.storage,
            self.key.scales,
            self.key.offsets,
            self.fused_key_lut,
            self.value.storage,
            self.value.scales,
            self.value.offsets,
            self.fused_value_lut,
            block_size=self.spec.block_size,
            head_size=self.spec.head_size,
            num_kv_heads=self.spec.num_kv_heads,
            key_container_bytes=self.key.container_bytes,
            value_container_bytes=self.value.container_bytes,
            key_window=self.spec.dquant_key_window,
            mean_offset=self.key.record_bytes,
            stripe_bytes=self.spec.mean_stripe_bytes,
            chunk_tokens=chunk_tokens,
            tile_tokens=min(chunk_tokens, tile_tokens),
        )

    @property
    def recent_storage_bytes(self) -> int:
        return self.recent_pool.capacity * self.spec.recent_state_bytes if self.recent_pool is not None else 0

    def _dbg(self, *parts):
        if os.getenv("VLLM_ASCEND_DQUANT_DEBUG", "0") == "1":
            print("[DQ]", *parts, file=sys.stderr, flush=True)

    def write_sequence(self, request_id, slots, key, value, seq_len) -> None:
        expected = (slots.numel(), self.spec.num_kv_heads, self.spec.head_size)
        if slots.ndim != 1 or key.shape != expected or value.shape != expected or seq_len < slots.numel():
            raise ValueError("D-Quant sequence update shape or length is invalid")
        if not self.spec.uses_window_state:
            self.write(slots, key, value)
            return
        prefix = seq_len - slots.numel()
        if prefix == 0:
            self.release([request_id])
        state = self.recent.get(request_id)
        if state is None:
            if prefix:
                raise ValueError("D-Quant recent state missing: cached prefixes require their raw tail")
            state = self._new_recent_state(request_id)
            self.recent[request_id] = state
        self._dbg(
            "write_seq", request_id, "total", state.total, "prefix", prefix, "n", slots.numel(), "seq_len", seq_len
        )
        if state.total != prefix:
            raise ValueError("D-Quant request prefix changed without recomputation")
        total_length = state.length + slots.numel()
        window = self.spec.dquant_key_window or 1
        flush = max(0, total_length - self.spec.dquant_recent_window) // window * window
        self._dbg("flush", request_id, "flush", flush, "state_len", state.length)
        if not flush:
            # Most centered decode steps only append one token. Avoid copying
            # the entire raw tail through three temporary concatenations.
            state.key[state.length : total_length].copy_(key)
            state.value[state.length : total_length].copy_(value)
            state.slots[state.length : total_length].copy_(slots)
            state.length = total_length
            state.total = seq_len
            self.recent_pool.totals[state.pool_slot].fill_(seq_len)
            return
        single_token = slots.numel() == 1
        if single_token:
            state.key[state.length : total_length].copy_(key)
            state.value[state.length : total_length].copy_(value)
            state.slots[state.length : total_length].copy_(slots)
            key, value, slots = state.key[:total_length], state.value[:total_length], state.slots[:total_length]
        else:
            key = torch.cat((state.key[: state.length], key))
            value = torch.cat((state.value[: state.length], value))
            slots = torch.cat((state.slots[: state.length], slots))
        if flush:
            compressed_key = key[:flush]
            if self.spec.dquant_key_window:
                shape = (-1, window, self.spec.num_kv_heads, self.spec.head_size)
                means = compressed_key.view(shape).float().mean(dim=1).to(self.spec.dtype)
                compressed_key = (compressed_key.view(shape).float() - means[:, None].float()).to(key.dtype)
                compressed_key = compressed_key.reshape_as(key[:flush])
                stripes = torch.zeros(
                    (flush // window, window * self.spec.mean_stripe_bytes),
                    dtype=torch.uint8,
                    device=key.device,
                )
                mean_bytes = means.contiguous().view(torch.uint8).flatten(1)
                stripes[:, : mean_bytes.shape[1]] = mean_bytes
                self.means[slots[:flush]] = stripes.view(flush, self.spec.mean_stripe_bytes)
            self.write(slots[:flush], compressed_key, value[:flush], centered=True)
        length = key.shape[0] - flush
        if single_token and length:
            if self._flush_scratch is None:
                self._flush_scratch = (
                    torch.empty_like(state.key),
                    torch.empty_like(state.value),
                    torch.empty_like(state.slots),
                )
            for scratch, source in zip(self._flush_scratch, (key, value, slots)):
                scratch[:length].copy_(source[flush:])
            key, value, slots = (scratch[:length] for scratch in self._flush_scratch)
            flush = 0
        state.key[:length].copy_(key[flush:])
        state.value[:length].copy_(value[flush:])
        state.slots[:length].copy_(slots[flush:])
        state.length = length
        state.total = seq_len
        state.compressed = seq_len - length
        self.recent_pool.totals[state.pool_slot].fill_(seq_len)

    def write(self, slots: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *, centered=False) -> None:
        if self.spec.uses_window_state and not centered:
            raise ValueError("Windowed D-Quant writes require write_sequence and a request ID")
        expected = (slots.numel(), self.spec.num_kv_heads, self.spec.head_size)
        if key.shape != expected or value.shape != expected:
            raise ValueError("Current K/V shape does not match the D-Quant slots and heads")
        # Negative slots are vLLM padding and must never index the last record.
        valid = slots >= 0
        slots = slots[valid]
        key, value = key[valid], value[valid]
        if slots.numel() == 0:
            return
        self.key.write(slots, hadamard_rotate_npu(key).reshape(-1, self.spec.stream_length))
        self.value.write(slots, hadamard_rotate_npu(value).reshape(-1, self.spec.stream_length))

    def read(
        self,
        block_table: torch.Tensor,
        seq_len: int,
        dtype: torch.dtype,
        request_id: str | None = None,
        *,
        include_recent: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len <= 0:
            raise ValueError("D-Quant sequence length must be positive")
        num_blocks = (seq_len + self.spec.block_size - 1) // self.spec.block_size
        if block_table.ndim != 1 or block_table.numel() < num_blocks:
            raise ValueError("Block table does not cover the D-Quant sequence")
        # Page copies operate on raw records, so per-wrapper initialization bits
        # cannot describe ownership. The scheduler determines readable slots.
        state = self.recent.get(request_id)
        if self.spec.uses_window_state and request_id is not None and (state is None or state.total != seq_len):
            raise ValueError(
                f"D-Quant request length does not match its recent state: request={request_id!r} "
                f"read_len={seq_len} state_total={None if state is None else state.total} "
                f"state_len={None if state is None else state.length} "
                f"compressed={None if state is None else state.compressed}"
            )
        compressed = state.compressed if state is not None else seq_len
        window = self.spec.dquant_key_window
        if window and compressed % window:
            raise ValueError("Only complete centered windows can be read without recent request state")
        if state is not None and compressed == 0:
            length = state.length if include_recent else 0
            return state.key[:length].to(dtype), state.value[:length].to(dtype)
        slots = None

        def compressed_slots():
            nonlocal slots
            if slots is None:
                positions = torch.arange(compressed, device=block_table.device)
                slots = block_table[positions // self.spec.block_size].long() * self.spec.block_size
                slots = slots + positions % self.spec.block_size
            return slots

        shape = (compressed, self.spec.num_kv_heads, self.spec.head_size)
        mirror = self.mirror.get(request_id) if (self.paged_decode and request_id is not None) else None
        if mirror is not None and mirror[0] > compressed:
            # A recomputed request cannot reuse its old decoded prefix.
            self.mirror.pop(request_id, None)
            self.mirror_tail_end.pop(request_id, None)
            self.host_mirror.pop(request_id, None)
            mirror = None
        if self.paged_decode:
            if mirror is not None:
                # Reuse the decoded dense prefix; decode only new records.
                # Persistent buffers grow in bounded increments rather than
                # retaining an additional full dense prefix as spare capacity.
                filled, rows, key_buf, val_buf = mirror
                key, value = key_buf[:filled], val_buf[:filled]
                window_size = window or 1
                target = min(compressed, self.mirror_cap // window_size * window_size)
                if target > filled:
                    if target > rows:
                        self.mirror_tail_end.pop(request_id, None)
                        new_rows = self._mirror_rows(target)
                        new_key = torch.empty(
                            (self._mirror_buffer_rows(new_rows),) + key_buf.shape[1:],
                            dtype=key_buf.dtype,
                            device=key_buf.device,
                        )
                        new_val = torch.empty(
                            (self._mirror_buffer_rows(new_rows),) + val_buf.shape[1:],
                            dtype=val_buf.dtype,
                            device=val_buf.device,
                        )
                        new_key[:filled].copy_(key_buf[:filled])
                        new_val[:filled].copy_(val_buf[:filled])
                        key_buf, val_buf, rows = new_key, new_val, new_rows
                    new_key, new_value = self.read_compressed_range(block_table, filled, target, dtype)
                    key_buf[filled:target].copy_(new_key)
                    val_buf[filled:target].copy_(new_value)
                    filled = target
                    key, value = key_buf[:filled], val_buf[:filled]
                self.mirror[request_id] = (filled, rows, key_buf, val_buf)
                if compressed > filled:
                    # Keep the bounded prefix after reaching the cap. Only the
                    # uncached suffix needs entropy decoding on later steps.
                    suffix_k, suffix_v = self._read_overflow(block_table, filled, compressed, dtype, request_id)
                    key = torch.cat((key, suffix_k))
                    value = torch.cat((value, suffix_v))
            else:

                def decode(cache):
                    return paged_rans_dequantize_npu(
                        cache.storage,
                        block_table,
                        compressed,
                        self.spec.block_size,
                        cache.container_bytes,
                        cache.stream_length,
                        cache.scales,
                        cache.offsets,
                        cache.decode_luts,
                        dtype,
                        output_chunk=self.output_chunk,
                    ).reshape(shape)

                key, value = decode(self.key), decode(self.value)
                if compressed:
                    key, value = hadamard_rotate_npu(key), hadamard_rotate_npu(value)
                    if window:
                        key = self._restore_key_means(key, block_table, 0)
                filled = min(compressed, self.mirror_cap // (window or 1) * (window or 1))
                if request_id is not None and filled > 0:
                    self.mirror_tail_end.pop(request_id, None)
                    self.host_mirror.pop(request_id, None)
                    rows = self._mirror_rows(filled)
                    key_buf = torch.empty(
                        (self._mirror_buffer_rows(rows),) + key.shape[1:], dtype=key.dtype, device=key.device
                    )
                    val_buf = torch.empty(
                        (self._mirror_buffer_rows(rows),) + value.shape[1:], dtype=value.dtype, device=value.device
                    )
                    key_buf[:filled].copy_(key[:filled])
                    val_buf[:filled].copy_(value[:filled])
                    self.mirror[request_id] = (filled, rows, key_buf, val_buf)
        else:
            key = self.key.read(compressed_slots(), dtype, require_initialized=False).reshape(shape)
            value = self.value.read(compressed_slots(), dtype, require_initialized=False).reshape(shape)
            if compressed:
                key, value = hadamard_rotate_npu(key), hadamard_rotate_npu(value)
                if window:
                    mean_bytes = self.means[compressed_slots()].reshape(-1, window * self.spec.mean_stripe_bytes)
                    width = self.spec.stream_length * torch.empty((), dtype=self.spec.dtype).element_size()
                    means = mean_bytes[:, :width].contiguous().view(self.spec.dtype)
                    means = means.view(-1, self.spec.num_kv_heads, self.spec.head_size).repeat_interleave(window, 0)
                    key = (key.float() + means.float()).to(dtype)
        if state is not None and include_recent:
            key = torch.cat((key, state.key[: state.length].to(dtype)))
            value = torch.cat((value, state.value[: state.length].to(dtype)))
        return key, value

    def _decode_batch(self, query, block_tables, seq_lens, scale, output, request_ids, updates):
        slots, new_key, new_value = updates
        keys, values, kv_ends = [], [], []
        total = 0
        for index, (request_id, seq_len) in enumerate(zip(request_ids, seq_lens)):
            if seq_len > 1:
                key, value = self.read(block_tables[index], seq_len - 1, query.dtype, request_id, include_recent=False)
                if key.shape[0]:
                    keys.append(key)
                    values.append(value)
                state = self.recent[request_id]
                if state.length:
                    keys.append(state.key[: state.length].to(query.dtype))
                    values.append(state.value[: state.length].to(query.dtype))
            keys.append(new_key[index : index + 1])
            values.append(new_value[index : index + 1])
            total += seq_len
            kv_ends.append(total)
        count = len(seq_lens)
        result, _ = torch_npu.npu_fused_infer_attention_score(
            query=query[:count].contiguous(),
            key=torch.cat(keys),
            value=torch.cat(values),
            input_layout="TND",
            num_heads=query.shape[1],
            num_key_value_heads=self.spec.num_kv_heads,
            scale=scale,
            actual_seq_lengths=list(range(1, count + 1)),
            actual_seq_lengths_kv=kv_ends,
            sparse_mode=0,
        )
        output[:count].copy_(result.view(count, query.shape[1], self.spec.head_size))
        # Commit only after every sequence has consumed its previous cache.
        for index, (request_id, seq_len) in enumerate(zip(request_ids, seq_lens)):
            self.write_sequence(
                request_id, slots[index : index + 1], new_key[index : index + 1], new_value[index : index + 1], seq_len
            )
        return output

    def attention(
        self,
        query: torch.Tensor,
        block_tables: torch.Tensor,
        query_ends: list[int],
        seq_lens: list[int],
        scale: float,
        output: torch.Tensor,
        attn_mask: torch.Tensor,
        request_ids: Sequence[str] | None = None,
        updates: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if len(query_ends) != len(seq_lens) or block_tables.shape[0] < len(seq_lens):
            raise ValueError("D-Quant batch metadata lengths do not agree")
        if self.spec.uses_window_state and (request_ids is None or len(request_ids) != len(seq_lens)):
            raise ValueError("D-Quant window attention requires one request ID per sequence")
        if query.ndim != 3 or query.shape[-1] != self.spec.head_size:
            raise ValueError("D-Quant query must have shape [tokens, query_heads, head_size]")
        if query.shape[1] % self.spec.num_kv_heads:
            raise ValueError("Query heads must be divisible by KV heads")
        if output.shape != query.shape or not query_ends or query_ends[-1] > query.shape[0]:
            raise ValueError("D-Quant output or query lengths do not match the query")
        if (
            len(seq_lens) > 1
            and updates is not None
            and self.spec.uses_window_state
            and not self.decode_chunk_size
            and not self.fused_decode
            and query_ends == list(range(1, len(seq_lens) + 1))
            and all(length > 0 for length in seq_lens)
        ):
            return self._decode_batch(query, block_tables, seq_lens, scale, output, request_ids, updates)
        if (
            len(seq_lens) > 1
            and updates is not None
            and self.fused_decode
            and request_ids is not None
            and query_ends == list(range(1, len(seq_lens) + 1))
            and all(length > 1 for length in seq_lens)
        ):
            batched = self._batched_fused_decode_attention(
                query,
                block_tables,
                seq_lens,
                scale,
                output,
                request_ids,
                updates,
            )
            if batched is not None:
                slots, new_key, new_value = updates
                for index, (request_id, seq_len) in enumerate(zip(request_ids, seq_lens)):
                    self.write_sequence(
                        request_id,
                        slots[index : index + 1],
                        new_key[index : index + 1],
                        new_value[index : index + 1],
                        seq_len,
                    )
                return batched
        start = 0
        for index, (end, seq_len) in enumerate(zip(query_ends, seq_lens)):
            q_len = end - start
            if q_len <= 0 or q_len > seq_len:
                raise ValueError("Each D-Quant query length must be positive and no larger than its KV length")
            request_id = request_ids[index] if request_ids is not None else None
            prefix = seq_len - q_len
            self._dbg(
                "attn",
                index,
                request_id,
                "q_len",
                q_len,
                "seq_len",
                seq_len,
                "prefix",
                prefix,
                "has_state",
                request_id in self.recent,
            )
            # Reject stale state rather than returning invalid attention.
            degraded = False
            if updates is not None and self.spec.uses_window_state and request_id is not None and prefix:
                stale = self.recent.get(request_id)
                if stale is None or stale.total != prefix:
                    degraded = True
            if degraded:
                self.mirror.pop(request_id, None)
                self.mirror_tail_end.pop(request_id, None)
                self.host_mirror.pop(request_id, None)
                raise ValueError("D-Quant request state mismatch; refusing to substitute values for attention")
            if q_len == 1 and prefix and updates is not None and self.fused_decode:
                current = (updates[1][start:end], updates[2][start:end])
                destination = output[start:end]
                fused = self._fused_decode_attention(
                    query[start:end],
                    block_tables[index],
                    prefix,
                    scale,
                    request_id,
                    current,
                    output=destination if destination.is_contiguous() else None,
                )
                if fused is not None:
                    if fused is not destination:
                        destination.copy_(fused)
                    if not self._fused_decode_logged:
                        logging.getLogger(__name__).info(
                            "D-Quant fused rANS attention active: compressed_tokens=%d, "
                            "local_kv_heads=%d, device_mirror_tokens=%d, host_mirror_tokens=%d",
                            self.recent[request_id].compressed,
                            self.spec.num_kv_heads,
                            self.mirror_cap,
                            self.host_mirror_cap,
                        )
                        self._fused_decode_logged = True
                    self.write_sequence(request_id, updates[0][start:end], *current, seq_len)
                    start = end
                    continue
            if q_len == 1 and self.decode_chunk_size and prefix:
                current = None if updates is None else (updates[1][start:end], updates[2][start:end])
                result = self._stream_decode_attention(
                    query[start:end],
                    block_tables[index],
                    prefix if updates is not None else seq_len,
                    scale,
                    request_id,
                    current,
                )
                output[start:end].copy_(result)
                if updates is not None:
                    self.write_sequence(request_id, updates[0][start:end], *current, seq_len)
                start = end
                continue
            if updates is None:
                key, value = self.read(block_tables[index], seq_len, query.dtype, request_id)
            else:
                slots, new_key, new_value = updates
                prefix = seq_len - q_len
                key, value = new_key[start:end], new_value[start:end]
                if prefix:
                    reused = None
                    if q_len == 1 and self.spec.uses_window_state:
                        reused = self._single_decode_kv(
                            block_tables[index], prefix, query.dtype, request_id, (key, value)
                        )
                    if reused is not None:
                        key, value = reused
                    else:
                        old_key, old_value = self.read(block_tables[index], prefix, query.dtype, request_id)
                        key = torch.cat((old_key, key))
                        value = torch.cat((old_value, value))
            # sparse_mode=3 applies lower-right causal alignment. For a chunk
            # extending a cached prefix, query position zero is seq_len-q_len.
            result, _ = torch_npu.npu_fused_infer_attention_score(
                query=query[start:end].contiguous(),
                key=key,
                value=value,
                input_layout="TND",
                num_heads=query.shape[1],
                num_key_value_heads=self.spec.num_kv_heads,
                scale=scale,
                actual_seq_lengths=[q_len],
                actual_seq_lengths_kv=[key.shape[0]],
                atten_mask=attn_mask,
                sparse_mode=3,
            )
            output[start:end].copy_(result.view(q_len, query.shape[1], self.spec.head_size))
            if updates is not None and not degraded:
                # Commit after attention: a prefill chunk must not see means
                # computed from its future keys through quantization error.
                self.write_sequence(request_id, slots[start:end], new_key[start:end], new_value[start:end], seq_len)
            start = end
        return output
