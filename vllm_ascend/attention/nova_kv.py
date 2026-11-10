# SPDX-License-Identifier: Apache-2.0
"""NOVA attention with compressed history and bounded exact sink/recent windows."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import regex as re
import torch
import torch_npu
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from vllm.logger import logger
from vllm.v1.attention.backend import AttentionCGSupport, AttentionType, MultipleOf

from vllm_ascend.ascend_config import NovaKVConfig
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackend,
    AscendAttentionBackendImpl,
    AscendAttentionMetadataBuilder,
    AscendAttentionState,
)
from vllm_ascend.ops.triton.nova_bounded_fia_prefill import bounded_fia_prefill
from vllm_ascend.ops.triton.nova_bounded_prefill import exact_window_reader, planar_cold_reader
from vllm_ascend.ops.triton.nova_compressed_state import (
    ExactWindowPool,
    planar_cache_views,
    write_planar_packets,
)
from vllm_ascend.ops.triton.nova_direct_decode import DirectDecode
from vllm_ascend.quantization.methods.kv_cache.nova_kv.codec import NovaCodec, cache_views, packet_bytes

HEAD_DIM = 256
MAX_QUERY_GROUPS = 6
FIA_MASK_SIZE = 2048


class NovaMetadataBuilder(AscendAttentionMetadataBuilder):
    _cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    @classmethod
    def get_cudagraph_support(cls, vllm_config, kv_cache_spec):
        return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        # Decode graphs include padded rows; their slots stay in the persistent
        # buffer and are masked by the runner's request-slot sentinel.
        metadata.slot_mapping = common_attn_metadata.slot_mapping
        return metadata


class NovaAttentionBackend(AscendAttentionBackend):
    uses_nova_kv_cache = True

    @staticmethod
    def reshape_cache(raw, spec, kernel_block_size):
        return cache_views(raw, spec, kernel_block_size)

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None):
        return [MultipleOf(32)]

    @classmethod
    def customize_spec(cls, spec):
        return replace(spec, dtype=torch.uint8, state_content_bytes=packet_bytes(spec.head_size))

    @staticmethod
    def get_impl_cls():
        return NovaAttentionImpl

    @staticmethod
    def get_builder_cls():
        return NovaMetadataBuilder


class NovaCapture:
    """Preserve the existing calibration entry point."""

    def __init__(self, layer_name, options):
        self.layer_id = int(re.search(r"layers\.(\d+)", layer_name).group(1))
        self.root = Path(options.capture_path)
        self.root.mkdir(parents=True, exist_ok=True)
        self.rank = get_tensor_model_parallel_rank()
        self.limit = options.sample_tokens
        self.seen = 0
        self.generator = torch.Generator().manual_seed(1024 + self.layer_id)
        self.key = self.value = self.query_moment = None

    def add(self, query, key, value):
        q, k, v = [x.detach().float().cpu() for x in (query, key, value)]
        heads, dim = k.shape[1:]
        q = q.reshape(q.shape[0], heads, -1, dim).double()
        moment = torch.einsum("thgd,thgf->hdf", q, q)
        self.query_moment = moment if self.query_moment is None else self.query_moment + moment
        if self.key is None:
            self.key = torch.empty((self.limit, heads, dim), dtype=torch.bfloat16)
            self.value = torch.empty_like(self.key)
        for row in range(k.shape[0]):
            position = self.seen
            if position >= self.limit:
                position = int(torch.randint(self.seen + 1, (1,), generator=self.generator))
            if position < self.limit:
                self.key[position], self.value[position] = k[row], v[row]
            self.seen += 1
        count = min(self.limit, self.seen)
        path = self.root / f"layer_{self.layer_id:02d}_rank_{self.rank}.pt"
        temporary = path.with_suffix(".tmp")
        torch.save(
            {
                "key": self.key[:count],
                "value": self.value[:count],
                "query_moment": self.query_moment,
                "seen": self.seen,
            },
            temporary,
        )
        temporary.replace(path)


class NovaAttentionImpl(AscendAttentionBackendImpl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initialize_nova_state(self.vllm_config)

    def _initialize_nova_state(self, config):
        options = NovaKVConfig(**config.additional_config["nova_kv"])
        self._options = options
        block_size = config.cache_config.block_size
        if block_size < 128 or block_size % 32:
            raise ValueError("NOVA-KV requires cache block_size >= 128 and divisible by 32")
        if self.head_size != HEAD_DIM or not 1 <= self.num_queries_per_kv <= MAX_QUERY_GROUPS:
            raise ValueError("compressed NOVA currently requires head_size=256 and <=6 query groups")
        if self.sliding_window is not None or self.alibi_slopes is not None or self.sinks is not None:
            raise ValueError("NOVA-KV does not support sliding windows, ALiBi or attention sinks")
        if self.attn_type != AttentionType.DECODER or self.kv_sharing_target_layer_name is not None:
            raise ValueError("NOVA-KV requires full decoder attention without cross-layer KV sharing")
        if config.cache_config.enable_prefix_caching:
            raise ValueError("NOVA exact-window ownership requires prefix caching disabled")
        self.nova_codec = None
        self._window = self._cold = None
        self._causal_mask = None
        self._limit = config.scheduler_config.max_num_seqs
        self._sink = options.sink_tokens
        self._recent = options.recent_tokens
        self._prefill_query_tile = options.prefill_query_tile
        self._prefill_kv_tile = options.prefill_kv_tile

    def _codec(self, layer, device):
        if self.nova_codec is None:
            options = self._options
            layer_id = int(re.search(r"layers\.(\d+)", layer.layer_name).group(1))
            self.nova_codec = NovaCodec.load(
                str(Path(options.artifact_path) / f"layer_{layer_id:02d}.pt"),
                device,
                get_tensor_model_parallel_rank(),
                get_tensor_model_parallel_world_size(),
                self.num_kv_heads,
            )
        return self.nova_codec

    def _ensure_runtime(self, device, dtype):
        if self._window is None:
            self._window = ExactWindowPool(
                self._limit, self.num_kv_heads, self.head_size, self._sink, self._recent, device, dtype
            )
            self._cold = DirectDecode(
                self._limit,
                self.num_kv_heads,
                self.num_heads // self.num_kv_heads,
                device,
                self._options.direct_library_root,
                self.vllm_config.model_config.max_model_len,
                split_tokens=self._options.direct_split_tokens,
                half_lut=True,
                tile_tokens=128,
            )
            self._causal_mask = torch.ones((FIA_MASK_SIZE, FIA_MASK_SIZE), device=device, dtype=torch.bool).triu(1)
            logger.info_once("NOVA bounded cache: no full-history KV mirror; fixed decode workspace is profiled")

    def _request_slots(self, metadata, batch):
        if metadata.seq_lens_gpu is None or metadata.nova_request_slots is None:
            raise RuntimeError("NOVA needs persistent device sequence lengths and request slots")
        return metadata.nova_request_slots[:batch]

    def _decode(self, query, key, value, views, metadata, output, codec, request_slots=None):
        batch = len(metadata.seq_lens_list)
        if request_slots is None:
            request_slots = self._request_slots(metadata, batch)
        valid = (request_slots >= 0) & (request_slots < self._limit)
        lengths = torch.where(valid, metadata.seq_lens_gpu[:batch], 0).to(torch.int32).contiguous()
        self._window.write_rows(
            key[:batch].contiguous(),
            value[:batch].contiguous(),
            request_slots.contiguous(),
            (lengths.long() - 1).contiguous(),
        )
        query = query[:batch].reshape(batch, self.num_kv_heads, -1, self.head_size).float()
        transformed = torch.einsum("bhgd,hde->bhge", query, codec.key_inverse.transpose(-1, -2)).contiguous()
        mean = torch.einsum("bhgd,hd->bhg", query, codec.key_mean).contiguous()
        starts = lengths.clamp(0, self._sink)
        ends = torch.maximum(starts, lengths - self._recent - 1).contiguous()
        # The launch geometry is captured by ACLGraph. It must cover all later
        # sequence lengths, not just the dummy lengths used during capture.
        max_tokens = min(
            self.vllm_config.model_config.max_model_len, metadata.block_tables.shape[1] * views[0].shape[-1]
        )
        cold, cold_lse = self._cold(
            transformed,
            mean,
            *views,
            codec.codebooks,
            metadata.block_tables[:batch],
            starts,
            ends,
            max_tokens,
            self.scale,
        )
        cold = torch.einsum("bhgd,hde->bhge", cold, codec.value_transform.transpose(-1, -2))
        result, _, _ = self._window.attend_and_merge(query, request_slots, lengths, self.scale, cold, cold_lse)
        output[:batch].copy_(result.reshape_as(output[:batch]).to(output.dtype))
        return output

    def _prefill(self, query, key, value, views, metadata, output):
        batch = len(metadata.seq_lens_list)
        slots = metadata.nova_request_slots_cpu
        if slots is None:
            raise RuntimeError("NOVA needs host request slots for prefill")
        slots = slots + [-1] * (batch - len(slots))
        # A mixed chunked-prefill batch can contain many one-token decode rows.
        # Compact them once, retaining the ownership claimed for the full batch.
        decode_requests, decode_rows = [], []
        first = 0
        for request, end in enumerate(metadata.actual_seq_lengths_q):
            length = metadata.seq_lens_list[request]
            if end - first == 1 and length > 1 and 0 <= slots[request] < self._limit:
                decode_requests.append(request)
                decode_rows.append(first)
            first = end
        if decode_requests:
            requests = torch.tensor(decode_requests, device=query.device, dtype=torch.long)
            rows = torch.tensor(decode_rows, device=query.device, dtype=torch.long)
            decode_metadata = SimpleNamespace(
                seq_lens_list=[metadata.seq_lens_list[i] for i in decode_requests],
                seq_lens_gpu=metadata.seq_lens_gpu.index_select(0, requests),
                block_tables=metadata.block_tables.index_select(0, requests),
            )
            decoded = torch.empty_like(query.index_select(0, rows))
            self._decode(
                query.index_select(0, rows),
                key.index_select(0, rows),
                value.index_select(0, rows),
                views,
                decode_metadata,
                decoded,
                self.nova_codec,
                request_slots=metadata.nova_request_slots.index_select(0, requests),
            )
            output.index_copy_(0, rows, decoded)
        start = 0
        for request, end in enumerate(metadata.actual_seq_lengths_q):
            count = end - start
            length = metadata.seq_lens_list[request]
            old_length = length - count
            slot = slots[request]
            if not count or not 0 <= slot < self._limit:
                start = end
                continue
            if request in decode_requests:
                start = end
                continue
            q = query[start:end].reshape(count, self.num_kv_heads, -1, self.head_size)
            k, v = key[start:end], value[start:end]
            if old_length == 0:
                # Initial prefill consumes the model's current K/V directly.
                # Only old compressed history ever needs bounded reconstruction.
                result, _ = torch_npu.npu_fused_infer_attention_score(
                    q.permute(1, 2, 0, 3).reshape(1, self.num_heads, count, self.head_size).contiguous(),
                    k.transpose(0, 1).unsqueeze(0).contiguous(),
                    v.transpose(0, 1).unsqueeze(0).contiguous(),
                    atten_mask=self._causal_mask if count > 1 else None,
                    input_layout="BNSD",
                    actual_seq_lengths=[count],
                    actual_seq_lengths_kv=[count],
                    num_heads=self.num_heads,
                    num_key_value_heads=self.num_kv_heads,
                    scale=self.scale,
                    sparse_mode=3 if count > 1 else 0,
                    pre_tokens=2147483647,
                    next_tokens=0,
                    inner_precise=1,
                )
                result = result.reshape(self.num_kv_heads, -1, count, self.head_size).permute(2, 0, 1, 3)
            else:
                cold = planar_cold_reader(*views, metadata.block_tables[request], self.nova_codec)
                exact = exact_window_reader(
                    self._window.key, self._window.value, slot, self._sink, self._window.ring, old_length
                )
                result, _ = bounded_fia_prefill(
                    q,
                    k,
                    v,
                    old_length,
                    cold,
                    exact,
                    self._sink,
                    self._recent,
                    query_tile=self._prefill_query_tile,
                    kv_tile=self._prefill_kv_tile,
                    scale=self.scale,
                )
            output[start:end].copy_(result.reshape_as(output[start:end]).to(output.dtype))
            self._window.seed_chunk(k.contiguous(), v.contiguous(), slot, old_length)
            start = end
        return output

    def forward(
        self, layer, query, key, value, kv_cache, attn_metadata, output=None, output_scale=None, output_block_scale=None
    ):
        if output is None:
            raise ValueError("NOVA requires an output buffer")
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("NOVA does not support output quantization")
        if getattr(layer, "head_size_v", self.head_size) != self.head_size:
            raise ValueError("NOVA-KV requires equal key and value head dimensions")
        codec = self._codec(layer, query.device)
        # Include all fixed per-layer state in vLLM's memory profiling pass.
        self._ensure_runtime(query.device, query.dtype)
        if attn_metadata is None:
            # Profiling has no cache pages, but must still observe the bounded
            # codec and lookup-table temporaries before cache sizing.
            if key is not None and value is not None:
                codec.encode(key, value, chunk_size=self._options.codec_batch_size)
            batch = min(self._limit, query.shape[0])
            grouped = query[:batch].reshape(batch, self.num_kv_heads, -1, self.head_size).float()
            transformed = torch.einsum("bhgd,hde->bhge", grouped, codec.key_inverse.transpose(-1, -2)).contiguous()
            self._cold.prepare_lut(transformed, codec.codebooks)
            return output.fill_(0)
        if key is None or value is None:
            raise NotImplementedError("NOVA requires current-token K/V")
        views = planar_cache_views(kv_cache)
        decoding = attn_metadata.attn_state == AscendAttentionState.DecodeOnly
        count = len(attn_metadata.seq_lens_list) if decoding else attn_metadata.num_actual_tokens
        output.zero_()
        encoded_key, encoded_value, scales = codec.encode(
            key[:count], value[:count], chunk_size=self._options.codec_batch_size
        )
        token_slots = attn_metadata.slot_mapping[:count]
        if decoding:
            token_slots = torch.where(attn_metadata.nova_request_slots[:count] >= 0, token_slots, -1)
        write_planar_packets(
            views,
            token_slots.long().contiguous(),
            encoded_key.contiguous(),
            encoded_value.contiguous(),
            scales.contiguous(),
        )
        if decoding:
            return self._decode(query, key, value, views, attn_metadata, output, codec)
        return self._prefill(query, key, value, views, attn_metadata, output)
