# SPDX-License-Identifier: Apache-2.0
"""Isolated service backend for the fixed-split compressed Cube experiment."""

from functools import lru_cache

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.nova_kv_v2 import NovaV2AttentionBackend, NovaV2AttentionImpl
from vllm_ascend.compilation.updatable_graph import get_capture_resource
from vllm_ascend.ops.nova_kv import nova_exact_window_update
from vllm_ascend.ops.triton.nova_compressed_state import ExactWindowPool, planar_cache_views
from vllm_ascend.ops.triton.nova_cube.nova_decode_encode_a5 import encode_decode_cache, prepare_encoder
from vllm_ascend.ops.triton.nova_cube.nova_decode_query_a5 import prepare_query_projection, project_query
from vllm_ascend.ops.triton.nova_cube.nova_exact_a5 import exact_partial, exact_physical, merge_partial_into
from vllm_ascend.ops.triton.nova_cube.nova_prefill_a5 import bounded_fia_prefill
from vllm_ascend.ops.triton.nova_cube.nova_query_lut import TritonQueryLut
from vllm_ascend.ops.triton.nova_cube.nova_triton_fixed_split import TritonFixedSplitDecode

_SCORE_HEADS = 0
_SCORE_PARTS = 4
_SCORE_STAGES = 1
_SCORE_TILE = 128
_VALUE_DIMS = 128
_PV_TILE = 128
_PV_FEED = 128
_FUSE_SCORE_WEIGHTS = False
_PV_STAGES = 2
_EXACT_TILE = 64
_ENCODE_GROUP = 1
_ENCODE_OVERLAP = True
_EXACT_OVERLAP = True
_MERGE_DIMS = 64
_QUERY_PROJECT = 2
_LONG_CONTEXT_SPLIT_TOKENS = 4096
_LONG_CONTEXT_MIN_TOKENS = 65536


@lru_cache(maxsize=8)
def _encode_stream(device):
    return torch.npu.Stream(device=device)


@triton.jit
def _batch_bounds(
    SEQ, SLOTS, LENGTHS, STARTS, ENDS, POSITIONS, LIMIT: tl.constexpr, SINK: tl.constexpr, RING: tl.constexpr
):
    batch = tl.program_id(0)
    slot = tl.load(SLOTS + batch)
    length = tl.where((slot >= 0) & (slot < LIMIT), tl.load(SEQ + batch), 0)
    start = tl.minimum(length, SINK)
    tl.store(LENGTHS + batch, length)
    tl.store(STARTS + batch, start)
    tl.store(ENDS + batch, tl.maximum(start, length - RING))
    tl.store(POSITIONS + batch, length - 1)


class NovaCubeAttentionBackend(NovaV2AttentionBackend):
    @staticmethod
    def get_impl_cls():
        return NovaCubeAttentionImpl


class NovaCubeAttentionImpl(NovaV2AttentionImpl):
    _prefill_impl = staticmethod(bounded_fia_prefill)

    def _initialize_nova_state(self, config):
        super()._initialize_nova_state(config)
        self._prefill_query_tile = 8192
        self._prefill_kv_tile = 32768

    # Keep the V2 forward writer: generated KV must remain available after it
    # leaves the exact recent window and enters the compressed cold range.
    def _codec(self, layer, device):
        codec = super()._codec(layer, device)
        if getattr(self, "_batch_lut", None) is None:
            self._batch_lut = TritonQueryLut(codec.codebooks, self.num_heads // self.num_kv_heads, max_batch=4)
        if _ENCODE_GROUP:
            prepare_encoder(codec)
        if _ENCODE_OVERLAP or _EXACT_OVERLAP:
            _encode_stream(device)
        if _QUERY_PROJECT:
            prepare_query_projection(codec)
        return codec

    def _ensure_runtime(self, device, dtype):
        if self._window is None:
            self._window = ExactWindowPool(
                self._limit,
                self.num_kv_heads,
                self.head_size,
                self._sink,
                self._recent,
                device,
                dtype,
            )
            self._causal_mask = torch.ones((2048, 2048), device=device, dtype=torch.bool).triu_(1)

    def forward(
        self, layer, query, key, value, kv_cache, attn_metadata, output=None, output_scale=None, output_block_scale=None
    ):
        if attn_metadata is not None:
            if (
                _ENCODE_GROUP
                and attn_metadata.attn_state == AscendAttentionState.DecodeOnly
                and len(attn_metadata.seq_lens_list) in (1, 4)
                and output is not None
                and output_scale is None
                and output_block_scale is None
                and key is not None
                and value is not None
                and getattr(layer, "head_size_v", self.head_size) == self.head_size
            ):
                codec = self._codec(layer, query.device)
                self._ensure_runtime(query.device, query.dtype)
                views = planar_cache_views(kv_cache)
                batch = len(attn_metadata.seq_lens_list)
                requests = attn_metadata.nova_request_slots[:batch].contiguous()
                if output.shape[0] != batch:
                    output.zero_()
                if _ENCODE_OVERLAP and self._window.ring >= 256:
                    stream = _encode_stream(query.device)
                    if batch == 4:
                        return self._decode(
                            query,
                            key,
                            value,
                            views,
                            attn_metadata,
                            output,
                            codec,
                            requests,
                            pending_encoder=stream,
                            encode_after_exact=True,
                        )
                    current = torch.npu.current_stream(query.device)
                    # The current token lies beyond even the final padded cold tile.
                    # Finish the writer before returning so allocator reuse is ordered.
                    stream.wait_stream(current)
                    with torch.npu.stream(stream):
                        encode_decode_cache(
                            key[:batch],
                            value[:batch],
                            codec,
                            views,
                            attn_metadata.slot_mapping[:batch],
                            requests,
                            self._limit,
                            _ENCODE_GROUP,
                        )
                    return self._decode(
                        query, key, value, views, attn_metadata, output, codec, requests, pending_encoder=stream
                    )
                encode_decode_cache(
                    key[:batch],
                    value[:batch],
                    codec,
                    views,
                    attn_metadata.slot_mapping[:batch],
                    requests,
                    self._limit,
                    _ENCODE_GROUP,
                )
                return self._decode(query, key, value, views, attn_metadata, output, codec, requests)
            return super().forward(
                layer, query, key, value, kv_cache, attn_metadata, output, output_scale, output_block_scale
            )
        if output is None or output_scale is not None or output_block_scale is not None:
            raise ValueError("Cube warmup requires an unquantized output buffer")
        codec = self._codec(layer, query.device)
        self._ensure_runtime(query.device, query.dtype)
        if key is not None and value is not None:
            codec.encode(key, value, chunk_size=self._options.codec_batch_size)
        return output.zero_()

    def _decode(
        self,
        query,
        key,
        value,
        views,
        metadata,
        output,
        codec,
        request_slots=None,
        pending_encoder=None,
        encode_after_exact=False,
    ):
        batch = len(metadata.seq_lens_list)
        if not 1 <= batch <= 4:
            raise ValueError("fixed-split Cube requires 1..4 decode requests")
        max_tokens = self.vllm_config.model_config.max_model_len
        if max(metadata.seq_lens_list) > max_tokens:
            raise ValueError("decode sequence exceeds experimental capacity")
        if request_slots is None:
            request_slots = self._request_slots(metadata, batch)
        request_slots = request_slots.contiguous()
        lengths, starts, ends, positions = get_capture_resource(
            ("nova_batch_decode_bounds", batch, query.device),
            lambda: (
                torch.empty(batch, device=query.device, dtype=torch.int32),
                torch.empty(batch, device=query.device, dtype=torch.int32),
                torch.empty(batch, device=query.device, dtype=torch.int32),
                torch.empty(batch, device=query.device, dtype=torch.int64),
            ),
        )
        _batch_bounds[(batch,)](
            metadata.seq_lens_gpu,
            request_slots,
            lengths,
            starts,
            ends,
            positions,
            self._limit,
            self._window.sink,
            self._window.ring,
        )
        nova_exact_window_update(
            self._window,
            key[:batch].contiguous(),
            value[:batch].contiguous(),
            request_slots=request_slots,
            positions=positions,
        )
        grouped = query[:batch].reshape(batch, self.num_kv_heads, -1, self.head_size).float().contiguous()
        side = None
        # Exact attention shares the encoder stream; cold PV only reads older KV.
        if _EXACT_OVERLAP and batch in (1, 4) and _EXACT_TILE and output.is_contiguous():
            grouped = grouped.contiguous()
            side = _encode_stream(query.device)
            current = torch.npu.current_stream(query.device)
            parts = triton.cdiv(self._window.capacity, _EXACT_TILE)
            exact = get_capture_resource(
                ("nova_parallel_exact", batch, self.num_kv_heads, self.num_heads, query.device),
                lambda: (
                    torch.empty((*grouped.shape[:-1], parts, self.head_size), device=query.device, dtype=torch.float32),
                    torch.empty((*grouped.shape[:-1], parts), device=query.device, dtype=torch.float32),
                ),
            )
            side.wait_stream(current)
            with torch.npu.stream(side):
                exact_partial(self._window, grouped, request_slots, lengths, self.scale, _EXACT_TILE, exact)
        if encode_after_exact:
            # Launch the short exact reduction before encoding occupies the AIVs.
            # Current-token compressed KV is outside the padded cold read range.
            pending_encoder.wait_stream(torch.npu.current_stream(query.device))
            with torch.npu.stream(pending_encoder):
                encode_decode_cache(
                    key[:batch],
                    value[:batch],
                    codec,
                    views,
                    metadata.slot_mapping[:batch],
                    request_slots,
                    self._limit,
                    _ENCODE_GROUP,
                )
        if _QUERY_PROJECT and batch in (1, 4):
            transformed, mean = project_query(grouped, codec, fuse_outputs=_QUERY_PROJECT >= 2)
        else:
            transformed = torch.einsum("bhgd,hde->bhge", grouped, codec.key_inverse.transpose(-1, -2)).contiguous()
            mean = torch.einsum("bhgd,hd->bhg", grouped, codec.key_mean).contiguous()
        work = get_capture_resource(
            ("nova_triton_fixed_split_batch", batch, max_tokens, self.num_kv_heads, self.num_heads, query.device),
            lambda: TritonFixedSplitDecode(
                max_tokens,
                self.num_kv_heads,
                self.num_heads // self.num_kv_heads,
                query.device,
                split_tokens=_LONG_CONTEXT_SPLIT_TOKENS
                if batch == 4 and max_tokens >= _LONG_CONTEXT_MIN_TOKENS
                else 1024,
                tile_tokens=_PV_TILE,
                value_dims=_VALUE_DIMS,
                feed_tokens=_PV_FEED,
                prepared_weights=True,
                batch=batch,
                score_heads=_SCORE_HEADS,
                score_parts=_SCORE_PARTS if batch == 4 else 10,
                score_tile=_SCORE_TILE,
                fuse_score_weights=_FUSE_SCORE_WEIGHTS and batch == 4,
                launch_options={"num_stages": _PV_STAGES if batch == 4 else 1},
                merge_dims=_MERGE_DIMS if batch in (1, 4) else 0,
                score_stages=_SCORE_STAGES if batch == 4 else 1,
            ),
        )
        lut = self._batch_lut(transformed)
        cold, cold_lse = work(
            transformed,
            mean,
            views,
            codec.codebooks,
            metadata.block_tables[:batch],
            starts,
            ends,
            self.scale,
            lut=lut,
        )
        cold = torch.einsum("bhgd,hde->bhge", cold, codec.value_transform.transpose(-1, -2))
        if side is not None:
            current.wait_stream(side)
            merge_partial_into(cold, cold_lse, exact, output[:batch])
            return output
        if _EXACT_TILE and batch == 4:
            result, _, _ = exact_physical(
                self._window,
                grouped,
                request_slots,
                lengths,
                self.scale,
                cold,
                cold_lse,
                _EXACT_TILE,
            )
        else:
            result, _, _ = self._window.attend_and_merge(
                grouped,
                request_slots,
                lengths,
                self.scale,
                cold,
                cold_lse,
            )
        output[:batch].copy_(result.reshape_as(output[:batch]).to(output.dtype))
        if pending_encoder is not None:
            torch.npu.current_stream(query.device).wait_stream(pending_encoder)
        return output
