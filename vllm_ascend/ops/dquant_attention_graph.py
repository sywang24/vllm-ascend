# SPDX-License-Identifier: Apache-2.0
"""Opt-in single-token decoder graphs around eager D-Quant attention."""

from types import MethodType

import torch
from vllm.forward_context import get_forward_context

from vllm_ascend.ops.dquant_segment_graph import SUPPORTED_DECODE_BATCHES
from vllm_ascend.patch.worker.patch_qwen3_5 import _uses_multimodal_rope


class DQuantStaticCall:
    """Capture stateless computation with stable, privately owned inputs."""

    def __init__(self, function, inputs, pool):
        self.inputs = tuple(t.clone() if t is not None else None for t in inputs)
        self.replays = 0
        self.graph = torch.npu.NPUGraph()
        for _ in range(3):
            self.output = function(*self.inputs)
        torch.npu.synchronize()
        with torch.npu.graph(self.graph, pool=pool):
            self.output = function(*self.inputs)
        torch.npu.synchronize()

    def matches(self, inputs):
        return len(inputs) == len(self.inputs) and all(
            (actual is None and saved is None)
            or (
                actual is not None
                and saved is not None
                and actual.shape == saved.shape
                and actual.dtype == saved.dtype
                and actual.device == saved.device
            )
            for actual, saved in zip(inputs, self.inputs)
        )

    def replay(self, inputs):
        for saved, actual in zip(self.inputs, inputs):
            if actual is not None:
                saved.copy_(actual)
        self.graph.replay()
        self.replays += 1
        return self.output


class DQuantAttentionDecoderGraphs:
    def __init__(self, layer, pool):
        self.layer = layer
        self.original = layer.forward
        self.pool = pool
        self.pre = self.post = None
        self.pre_by_batch = {}
        self.post_by_batch = {}
        self.enabled = True
        layer.forward = MethodType(self.forward, layer)

    def before_attention(self, hidden, residual, positions):
        layer = self.layer
        attention = layer.self_attn
        if residual is None:
            residual = hidden
            hidden = layer.input_layernorm(hidden)
        else:
            hidden, residual = layer.input_layernorm(hidden, residual)
        qkv, _ = attention.qkv_proj(hidden)
        cos_sin = attention.rotary_emb.cos_sin_cache[positions].to(device=qkv.device, dtype=qkv.dtype)
        q, k, v, gate = torch.ops.vllm.triton_split_qkv_rmsnorm_mrope(
            qkv=qkv,
            q_weight=1.0 + attention.q_norm.weight,
            k_weight=1.0 + attention.k_norm.weight,
            cos_sin=cos_sin,
            num_q_heads=attention.num_heads,
            num_kv_heads=attention.num_kv_heads,
            head_size=attention.head_dim,
            eps=attention.config.rms_norm_eps,
            mrope_section=attention.rotary_emb.mrope_section,
            is_interleaved=attention.rotary_emb.mrope_interleaved,
            rope_dim=attention.rotary_emb.rotary_dim,
            has_gate=attention.attn_output_gate,
        )
        return q, k, v, gate, residual

    def after_attention(self, attended, gate, residual):
        layer = self.layer
        if layer.self_attn.attn_output_gate:
            attended = attended * torch.sigmoid(gate)
        hidden, _ = layer.self_attn.o_proj(attended)
        hidden, residual = layer.post_attention_layernorm(hidden, residual)
        return layer.mlp(hidden), residual

    def forward(self, layer, hidden_states, residual, positions=None, **kwargs):
        context = get_forward_context()
        metadata = context.attn_metadata
        attention = layer.self_attn
        meta = metadata.get(attention.attn.layer_name) if isinstance(metadata, dict) else None
        if (
            not self.enabled
            or kwargs
            or positions is None
            or hidden_states.ndim != 2
            or hidden_states.shape[0] not in SUPPORTED_DECODE_BATCHES
            or hidden_states.device.type != "npu"
            or hidden_states.dtype not in (torch.bfloat16, torch.float16)
            or meta is None
            or meta.num_actual_tokens != hidden_states.shape[0]
            or meta.num_decodes != hidden_states.shape[0]
            or meta.num_prefills != 0
            or layer.layer_scale
            or not _uses_multimodal_rope(attention)
        ):
            return self.original(hidden_states=hidden_states, residual=residual, positions=positions, **kwargs)
        inputs = (hidden_states, residual, positions)
        count = hidden_states.shape[0]
        self.pre = self.pre_by_batch.get(count)
        if self.pre is None:
            self.pre = DQuantStaticCall(self.before_attention, inputs, self.pool)
            self.pre_by_batch[count] = self.pre
        if not self.pre.matches(inputs):
            return self.original(hidden_states=hidden_states, residual=residual, positions=positions, **kwargs)
        q, k, v, gate, residual = self.pre.replay(inputs)
        # Stateful KV reads/writes must execute exactly once outside capture.
        attended = attention.attn(q, k, v)
        inputs = (attended, gate, residual)
        self.post = self.post_by_batch.get(count)
        if self.post is None:
            self.post = DQuantStaticCall(self.after_attention, inputs, self.pool)
            self.post_by_batch[count] = self.post
        if not self.post.matches(inputs):
            return self.after_attention(*inputs)
        return self.post.replay(inputs)


def install_dquant_attention_graphs(model):
    """Install after weight loading, under the worker's eager/D-Quant guards."""
    from vllm.model_executor.models.qwen3_5 import Qwen3_5DecoderLayer

    if hasattr(model, "_dquant_attention_graphs"):
        return model._dquant_attention_graphs
    pool = torch.npu.graph_pool_handle()
    graphs = [
        DQuantAttentionDecoderGraphs(layer, pool)
        for layer in model.modules()
        if isinstance(layer, Qwen3_5DecoderLayer)
        and layer.layer_type == "full_attention"
        and not layer.layer_scale
        and _uses_multimodal_rope(layer.self_attn)
    ]
    model._dquant_attention_graphs = graphs
    return graphs
