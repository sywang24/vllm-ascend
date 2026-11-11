# SPDX-License-Identifier: Apache-2.0
"""Opt-in graph replay of contiguous Qwen GDN decoder layers in eager mode."""

import copy
from types import MethodType

import torch
from vllm.forward_context import get_forward_context

MAX_SEGMENT_LAYERS = 3
SUPPORTED_DECODE_BATCHES = (1, 2)


def _metadata_inputs(metadata):
    decode = metadata.non_spec_decode_metadata
    return (
        metadata.non_spec_state_indices_tensor,
        decode.actual_seq_lengths,
        decode.causal_conv1d.cache_indices,
        decode.causal_conv1d.query_start_loc,
    )


def _tensor_matches(actual, saved):
    return actual.shape == saved.shape and actual.dtype == saved.dtype and actual.device == saved.device


class DQuantSegmentGraph:
    def __init__(self, layers, originals, hidden, residual, positions, pool, input_cache=None):
        self.layers = layers
        self.originals = originals
        self.hidden = hidden.clone()
        self.residual = None if residual is None else residual.clone()
        self.positions = positions.clone()
        self.inputs = []
        self.states = []
        self.replays = 0
        context = get_forward_context()
        previous = {}
        static_metadata = {}
        saved_states = []
        enabled = []
        for layer in layers:
            attention = layer.linear_attn
            metadata = context.attn_metadata[attention.prefix]
            previous[attention.prefix] = metadata
            static = copy.copy(metadata)
            static.non_spec_decode_metadata = copy.copy(metadata.non_spec_decode_metadata)
            static.non_spec_decode_metadata.causal_conv1d = copy.copy(metadata.non_spec_decode_metadata.causal_conv1d)
            tensors = []
            for owner, name in (
                (static, "non_spec_state_indices_tensor"),
                (static.non_spec_decode_metadata, "actual_seq_lengths"),
                (static.non_spec_decode_metadata.causal_conv1d, "cache_indices"),
                (static.non_spec_decode_metadata.causal_conv1d, "query_start_loc"),
            ):
                source = getattr(owner, name)
                if input_cache is None:
                    tensor = source.clone()
                else:
                    key = (source.data_ptr(), source.shape, source.stride(), source.dtype, source.device)
                    tensor = input_cache.get(key)
                    if tensor is None:
                        tensor = source.clone()
                        input_cache[key] = tensor
                setattr(owner, name, tensor)
                tensors.append(tensor)
            self.inputs.append((attention.prefix, tensors))
            conv_slots = tensors[2].flatten().long()
            state_slots = tensors[0].flatten().long()
            conv, state = attention.kv_cache
            self.states.append((conv, state))
            saved_states.append((conv, conv_slots, conv.index_select(0, conv_slots).clone()))
            saved_states.append((state, state_slots, state.index_select(0, state_slots).clone()))
            enabled.append((attention, getattr(attention, "_dquant_gdn_graph_enabled", None)))
            static_metadata[attention.prefix] = static
        self.graph = torch.npu.NPUGraph()
        try:
            context.attn_metadata.update(static_metadata)
            for attention, _ in enabled:
                attention._dquant_gdn_graph_enabled = False
            for _ in range(3):
                self._core()
            torch.npu.synchronize()
            with torch.npu.graph(self.graph, pool=pool):
                self._core()
            torch.npu.synchronize()
        finally:
            context.attn_metadata.update(previous)
            for attention, value in enabled:
                attention._dquant_gdn_graph_enabled = value
            # Capturing a stateful decoder must not advance the real request.
            for state, slots, saved in saved_states:
                state.index_copy_(0, slots, saved)
            torch.npu.synchronize()

    def _core(self):
        hidden, residual = self.hidden, self.residual
        for original in self.originals:
            hidden, residual = original(hidden_states=hidden, residual=residual, positions=self.positions)
        self.output = hidden, residual

    def matches(self, hidden, residual, positions, metadata):
        if not _tensor_matches(hidden, self.hidden) or not _tensor_matches(positions, self.positions):
            return False
        if (residual is None) != (self.residual is None):
            return False
        if residual is not None and not _tensor_matches(residual, self.residual):
            return False
        for layer, (prefix, saved), state in zip(self.layers, self.inputs, self.states):
            if any(actual is not previous for actual, previous in zip(layer.linear_attn.kv_cache, state)):
                return False
            if not all(
                _tensor_matches(actual, previous) for actual, previous in zip(_metadata_inputs(metadata[prefix]), saved)
            ):
                return False
        return True

    def replay(self, hidden, residual, positions):
        self.hidden.copy_(hidden)
        if self.residual is not None:
            self.residual.copy_(residual)
        self.positions.copy_(positions)
        metadata = get_forward_context().attn_metadata
        for prefix, destinations in self.inputs:
            for destination, source in zip(destinations, _metadata_inputs(metadata[prefix])):
                destination.copy_(source)
        self.graph.replay()
        self.replays += 1
        return self.output


class DQuantDecodeSegment:
    def __init__(self, layers, pool):
        self.layers = tuple(layers)
        self.originals = tuple(layer.forward for layer in layers)
        self.pool = pool
        self.graph = None
        self.graphs_by_batch = {}
        self.enabled = True
        self._skip = False
        for index, layer in enumerate(layers):
            layer.forward = MethodType(self._make_forward(index), layer)

    def _supports(self, hidden, positions, metadata):
        if (
            not self.enabled
            or positions is None
            or hidden.ndim != 2
            or hidden.shape[0] not in SUPPORTED_DECODE_BATCHES
            or hidden.device.type != "npu"
            or hidden.dtype not in (torch.bfloat16, torch.float16)
            or not isinstance(metadata, dict)
        ):
            return False
        for layer in self.layers:
            attention = layer.linear_attn
            meta = metadata.get(attention.prefix)
            if (
                meta is None
                or meta.num_actual_tokens != hidden.shape[0]
                or meta.num_decodes != hidden.shape[0]
                or meta.num_prefills != 0
                or meta.spec_sequence_masks is not None
                or attention.head_k_dim != 128
                or attention.head_v_dim != 128
                or meta.non_spec_decode_metadata is None
                or meta.non_spec_decode_metadata.causal_conv1d.cache_indices.numel() != hidden.shape[0]
                or meta.non_spec_state_indices_tensor.numel() != hidden.shape[0]
            ):
                return False
        return True

    def _make_forward(self, index):
        def forward(layer, hidden_states, residual, positions=None, **kwargs):
            if index == 0:
                self._skip = False
                context = get_forward_context()
                if not kwargs and self._supports(hidden_states, positions, context.attn_metadata):
                    count = hidden_states.shape[0]
                    self.graph = self.graphs_by_batch.get(count)
                    if self.graph is None:
                        self.graph = DQuantSegmentGraph(
                            self.layers, self.originals, hidden_states, residual, positions, self.pool
                        )
                        self.graphs_by_batch[count] = self.graph
                    if self.graph.matches(hidden_states, residual, positions, context.attn_metadata):
                        result = self.graph.replay(hidden_states, residual, positions)
                        self._skip = True
                        return result
            elif self._skip:
                if index == len(self.layers) - 1:
                    self._skip = False
                return hidden_states, residual
            return self.originals[index](hidden_states=hidden_states, residual=residual, positions=positions, **kwargs)

        return forward


def install_dquant_segment_graphs(model):
    """Install once after weight loading, before cache allocation and warmup."""
    from vllm.model_executor.models.qwen3_5 import Qwen3_5DecoderLayer

    if hasattr(model, "_dquant_decode_segments"):
        return model._dquant_decode_segments
    pool = torch.npu.graph_pool_handle()
    segments = []
    for module in model.modules():
        layers = getattr(module, "layers", None)
        if not isinstance(layers, torch.nn.ModuleList):
            continue
        group = []
        for layer in layers:
            if isinstance(layer, Qwen3_5DecoderLayer) and layer.layer_type == "linear_attention":
                group.append(layer)
                if len(group) < MAX_SEGMENT_LAYERS:
                    continue
            if group:
                segments.append(DQuantDecodeSegment(group, pool))
                group = []
        if group:
            segments.append(DQuantDecodeSegment(group, pool))
    model._dquant_decode_segments = segments
    return segments
