# SPDX-License-Identifier: Apache-2.0
"""Opt-in GDN core graph replay within the eager D-Quant model path."""

import torch

from vllm_ascend.ops.gdn import DeviceOperator, _causal_conv1d, _recurrent_gated_delta_rule
from vllm_ascend.ops.triton.gdn_qk_norm import l2norm_qk_fwd


class SingleTokenGDNGraph:
    @staticmethod
    def supports(layer, metadata, mixed_qkv, b, a):
        if (
            metadata.num_actual_tokens != 1
            or metadata.num_decodes != 1
            or metadata.num_prefills != 0
            or metadata.spec_sequence_masks is not None
            or layer.head_k_dim != 128
            or layer.head_v_dim != 128
        ):
            return False
        if mixed_qkv.device.type != "npu" or mixed_qkv.dtype not in (torch.float16, torch.bfloat16):
            return False
        decode = metadata.non_spec_decode_metadata
        return (
            decode is not None
            and mixed_qkv.ndim == 2
            and mixed_qkv.shape[0] >= 1
            and b.shape[0] >= 1
            and a.shape[0] >= 1
            and decode.causal_conv1d.cache_indices.numel() == 1
            and metadata.non_spec_state_indices_tensor.numel() == 1
        )

    def matches(self, mixed_qkv, b, a, metadata):
        decode = metadata.non_spec_decode_metadata
        pairs = (
            (mixed_qkv, self.mixed_qkv),
            (b, self.b),
            (a, self.a),
            (decode.causal_conv1d.cache_indices, self.conv_indices),
            (decode.causal_conv1d.query_start_loc, self.query_start),
            (metadata.non_spec_state_indices_tensor, self.state_indices),
            (decode.actual_seq_lengths, self.actual_lengths),
        )
        return (
            all(
                actual.shape == saved.shape and actual.dtype == saved.dtype and actual.device == saved.device
                for actual, saved in pairs
            )
            and self.layer.kv_cache[0] is self.conv_state
            and self.layer.kv_cache[1] is self.ssm_state
        )

    def __init__(self, layer, mixed_qkv, b, a, metadata, pool=None):
        self.layer = layer
        self.mixed_qkv = mixed_qkv.clone()
        self.b = b.clone()
        self.a = a.clone()
        decode = metadata.non_spec_decode_metadata
        self.conv_indices = decode.causal_conv1d.cache_indices.clone()
        self.query_start = decode.causal_conv1d.query_start_loc.clone()
        self.state_indices = metadata.non_spec_state_indices_tensor.clone()
        self.actual_lengths = decode.actual_seq_lengths.clone()
        self.conv_state, self.ssm_state = layer.kv_cache
        self.conv_weight = (
            layer.conv1d.weight.view(layer.conv1d.weight.size(0), layer.conv1d.weight.size(2))
            .transpose(0, 1)
            .to(dtype=mixed_qkv.dtype)
            .contiguous()
        )
        self.conv_output = torch.empty_like(mixed_qkv)
        self.output = torch.empty(
            (1, layer.num_v_heads // layer.tp_size, layer.head_v_dim),
            device=mixed_qkv.device,
            dtype=mixed_qkv.dtype,
        )
        conv_slots = self.conv_indices.flatten().long()
        state_slots = self.state_indices.flatten().long()
        saved_conv = self.conv_state.index_select(0, conv_slots).clone()
        saved_ssm = self.ssm_state.index_select(0, state_slots).clone()
        self.graph = torch.npu.NPUGraph()
        self.replays = 0
        try:
            for _ in range(3):
                self._core()
            torch.npu.synchronize()
            with torch.npu.graph(self.graph, pool=pool):
                self._core()
            torch.npu.synchronize()
        finally:
            # Warmup/capture must not advance the real request's recurrent state.
            self.conv_state.index_copy_(0, conv_slots, saved_conv)
            self.ssm_state.index_copy_(0, state_slots, saved_ssm)
            torch.npu.synchronize()

    def _core(self):
        _causal_conv1d(
            self.conv_output,
            self.mixed_qkv,
            self.conv_weight,
            conv_state=self.conv_state,
            bias_opt=self.layer.conv1d.bias,
            query_start_loc_opt=self.query_start,
            cache_indices_opt=self.conv_indices,
            activation_mode=1 if self.layer.activation else 0,
            run_mode=1,
        )
        query, key, value = self.layer.rearrange_mixed_qkv(self.conv_output)
        query, key = l2norm_qk_fwd(query, key)
        g, beta = DeviceOperator.fused_gdn_gating(self.layer.A_log, self.a, self.b, self.layer.dt_bias)
        result = _recurrent_gated_delta_rule(
            query=query.squeeze(0),
            key=key.squeeze(0),
            value=value.squeeze(0),
            g=g.squeeze(0),
            beta=beta.squeeze(0),
            state=self.ssm_state,
            scale=key.shape[-1] ** -0.5,
            actual_seq_lengths=self.actual_lengths,
            ssm_state_indices=self.state_indices,
        )
        self.output.copy_(result)

    def replay(self, mixed_qkv, b, a, metadata, output):
        decode = metadata.non_spec_decode_metadata
        self.mixed_qkv.copy_(mixed_qkv)
        self.b.copy_(b)
        self.a.copy_(a)
        self.conv_indices.copy_(decode.causal_conv1d.cache_indices)
        self.query_start.copy_(decode.causal_conv1d.query_start_loc)
        self.state_indices.copy_(metadata.non_spec_state_indices_tensor)
        self.actual_lengths.copy_(decode.actual_seq_lengths)
        self.graph.replay()
        output[:1].copy_(self.output)
        self.replays += 1
