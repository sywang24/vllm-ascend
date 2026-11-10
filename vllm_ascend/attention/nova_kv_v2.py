# SPDX-License-Identifier: Apache-2.0
"""NOVA operator interfaces for the V2 model runner."""

from types import SimpleNamespace

import torch

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.nova_kv import NovaAttentionBackend, NovaAttentionImpl
from vllm_ascend.ops.nova_kv import (
    nova_exact_window_update,
    nova_quant_cache,
    nova_quant_flash_attention,
    nova_quant_flash_attention_metadata,
    nova_quant_query_lut,
)
from vllm_ascend.ops.triton.nova_compressed_state import planar_cache_views


class NovaV2AttentionBackend(NovaAttentionBackend):
    @staticmethod
    def get_impl_cls():
        return NovaV2AttentionImpl


class NovaV2AttentionImpl(NovaAttentionImpl):
    def _decode(self, query, key, value, views, metadata, output, codec, request_slots=None):
        batch = len(metadata.seq_lens_list)
        if request_slots is None:
            request_slots = self._request_slots(metadata, batch)
        valid = (request_slots >= 0) & (request_slots < self._limit)
        lengths = torch.where(valid, metadata.seq_lens_gpu[:batch], 0).to(torch.int32).contiguous()
        nova_exact_window_update(
            self._window,
            key[:batch].contiguous(),
            value[:batch].contiguous(),
            request_slots=request_slots.contiguous(),
            positions=(lengths.long() - 1).contiguous(),
        )
        query = query[:batch].reshape(batch, self.num_kv_heads, -1, self.head_size).float()
        transformed, mean, lut = nova_quant_query_lut(query, codec, self._cold)
        table = metadata.block_tables[:batch]
        plan = nova_quant_flash_attention_metadata(
            self._cold, table, views, self.vllm_config.model_config.max_model_len
        )
        result, _ = nova_quant_flash_attention(
            query,
            codec,
            views,
            table,
            self._window,
            self.scale,
            transformed=transformed,
            mean=mean,
            lut=lut,
            request_slots=request_slots,
            lengths=lengths,
            direct_decode=self._cold,
            plan=plan,
        )
        output[:batch].copy_(result.reshape_as(output[:batch]).to(output.dtype))
        return output

    def _prefill(self, query, key, value, views, metadata, output):
        batch = len(metadata.seq_lens_list)
        slots = metadata.nova_request_slots_cpu
        if slots is None:
            raise RuntimeError("NOVA needs host request slots for prefill")
        slots = slots + [-1] * (batch - len(slots))
        decode_requests, decode_rows = [], []
        first = 0
        for request, end in enumerate(metadata.actual_seq_lengths_q):
            if end - first == 1 and metadata.seq_lens_list[request] > 1 and 0 <= slots[request] < self._limit:
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
            slot = slots[request]
            if not count or not 0 <= slot < self._limit or request in decode_requests:
                start = end
                continue
            old_length = metadata.seq_lens_list[request] - count
            k, v = key[start:end], value[start:end]
            result, _ = nova_quant_flash_attention(
                query[start:end].reshape(count, self.num_kv_heads, -1, self.head_size),
                self.nova_codec,
                views,
                metadata.block_tables[request],
                self._window,
                self.scale,
                key=k,
                value=v,
                old_length=old_length,
                request_slot=slot,
                causal_mask=self._causal_mask,
                query_tile=self._prefill_query_tile,
                kv_tile=self._prefill_kv_tile,
            )
            output[start:end].copy_(result.reshape_as(output[start:end]).to(output.dtype))
            nova_exact_window_update(
                self._window, k.contiguous(), v.contiguous(), request_slot=slot, old_length=old_length
            )
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
        self._ensure_runtime(query.device, query.dtype)
        if attn_metadata is None:
            if key is not None and value is not None:
                codec.encode(key, value, chunk_size=self._options.codec_batch_size)
            batch = min(self._limit, query.shape[0])
            grouped = query[:batch].reshape(batch, self.num_kv_heads, -1, self.head_size)
            nova_quant_query_lut(grouped, codec, self._cold)
            return output.fill_(0)
        if key is None or value is None:
            raise NotImplementedError("NOVA requires current-token K/V")
        views = planar_cache_views(kv_cache)
        decoding = attn_metadata.attn_state == AscendAttentionState.DecodeOnly
        count = len(attn_metadata.seq_lens_list) if decoding else attn_metadata.num_actual_tokens
        output.zero_()
        token_slots = attn_metadata.slot_mapping[:count]
        if decoding:
            token_slots = torch.where(attn_metadata.nova_request_slots[:count] >= 0, token_slots, -1)
        nova_quant_cache(
            key[:count], value[:count], codec, views, token_slots, chunk_size=self._options.codec_batch_size
        )
        if decoding:
            return self._decode(query, key, value, views, attn_metadata, output, codec)
        return self._prefill(query, key, value, views, attn_metadata, output)
