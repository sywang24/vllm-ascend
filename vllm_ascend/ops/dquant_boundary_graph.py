# SPDX-License-Identifier: Apache-2.0
"""Single-token graph regions separated only by eager D-Quant attention."""

from types import MethodType

import torch
from vllm.forward_context import get_forward_context

from vllm_ascend.ops.dquant_segment_graph import DQuantSegmentGraph, _metadata_inputs, _tensor_matches


class DQuantBoundaryGraph(DQuantSegmentGraph):
    def __init__(self, layers, originals, before, after, hidden, residual, gate, positions, pool, input_cache):
        self.before = before
        self.after = after
        self.gate = None if gate is None else gate.clone()
        super().__init__(layers, originals, hidden, residual, positions, pool, input_cache=input_cache)

    def _core(self):
        hidden, residual = self.hidden, self.residual
        if self.before is not None:
            hidden, residual = self.before.after_attention(hidden, self.gate, residual)
        for original in self.originals:
            hidden, residual = original(hidden_states=hidden, residual=residual, positions=self.positions)
        self.output = (
            (hidden, residual) if self.after is None else self.after.before_attention(hidden, residual, self.positions)
        )

    def replay_boundary(self, hidden, residual, gate, positions, copied_inputs):
        self.hidden.copy_(hidden)
        if residual is not None:
            self.residual.copy_(residual)
        if self.after is not None:
            self.positions.copy_(positions)
        if gate is not None:
            self.gate.copy_(gate)
        metadata = get_forward_context().attn_metadata
        for prefix, destinations in self.inputs:
            actual = metadata[prefix]
            for index, source in (
                (0, actual.non_spec_state_indices_tensor),
                (2, actual.non_spec_decode_metadata.causal_conv1d.cache_indices),
            ):
                destination = destinations[index]
                if id(destination) not in copied_inputs:
                    destination.copy_(source)
                    copied_inputs.add(id(destination))
        # Pure decode emits [0, ..., batch] for the cumulative token offsets.
        self.graph.replay()
        self.replays += 1
        return self.output

    def metadata_matches(self, metadata):
        for layer, (prefix, saved), states in zip(self.layers, self.inputs, self.states):
            if any(actual is not previous for actual, previous in zip(layer.linear_attn.kv_cache, states)):
                return False
            if not all(_tensor_matches(actual, old) for actual, old in zip(_metadata_inputs(metadata[prefix]), saved)):
                return False
        return True


class DQuantBoundaryChain:
    def __init__(self, layers, segments, attentions, pool, owner=None):
        self.layers = tuple(layers)
        self.originals = tuple(layer.forward for layer in layers)
        self.segments = tuple(segments)
        self.attentions = tuple(attentions)
        self.pool = pool
        self.owner = owner
        self.graphs = []
        self.graphs_by_batch = {}
        self.metadata_inputs_by_batch = {}
        self.enabled = True
        self._skip = False
        self.gdn_originals = {
            id(layer): original for segment in segments for layer, original in zip(segment.layers, segment.originals)
        }
        for index, layer in enumerate(layers):
            layer.forward = MethodType(self._make_forward(index), layer)

    def _supports(self, hidden, residual, positions, metadata):
        if not self.enabled or not isinstance(metadata, dict):
            return False
        if self.owner is not None and getattr(self.owner, "aux_hidden_state_layers", ()):
            return False
        if positions is None or not all(segment._supports(hidden, positions, metadata) for segment in self.segments):
            return False
        for wrapper in self.attentions:
            layer = wrapper.layer
            meta = metadata.get(layer.self_attn.attn.layer_name)
            if (
                meta is None
                or meta.num_actual_tokens != hidden.shape[0]
                or meta.num_decodes != hidden.shape[0]
                or meta.num_prefills != 0
                or layer.layer_scale
            ):
                return False
        for segment in self.segments:
            for layer in segment.layers:
                decode = metadata[layer.linear_attn.prefix].non_spec_decode_metadata
                offsets_shape = (hidden.shape[0] + 1,)
                if (
                    decode.actual_seq_lengths.shape != offsets_shape
                    or decode.causal_conv1d.query_start_loc.shape != offsets_shape
                ):
                    return False
        graphs = self.graphs_by_batch.get(hidden.shape[0], ())
        if graphs:
            first = graphs[0]
            if not first.matches(hidden, residual, positions, metadata):
                return False
            if not all(graph.metadata_matches(metadata) for graph in graphs):
                return False
            # Shared captured inputs may only be reused while source aliasing
            # still agrees; otherwise fall back to independent decoder graphs.
            aliases = {}
            for graph in graphs:
                for prefix, saved in graph.inputs:
                    actual = _metadata_inputs(metadata[prefix])
                    for index in (0, 2):
                        source = actual[index]
                        signature = (source.data_ptr(), source.stride())
                        if aliases.setdefault(id(saved[index]), signature) != signature:
                            return False
        return True

    def _run(self, hidden, residual, positions):
        self.graphs = self.graphs_by_batch.setdefault(hidden.shape[0], [])
        input_cache = self.metadata_inputs_by_batch.setdefault(hidden.shape[0], {})
        copied_inputs = set()
        gate = None
        previous = None
        for index in range(len(self.attentions) + 1):
            following = self.attentions[index] if index < len(self.attentions) else None
            gdn_layers = self.layers[index * 4 : index * 4 + 3] if following is not None else ()
            if index == len(self.graphs):
                originals = [self.gdn_originals[id(layer)] for layer in gdn_layers]
                self.graphs.append(
                    DQuantBoundaryGraph(
                        gdn_layers,
                        originals,
                        previous,
                        following,
                        hidden,
                        residual,
                        gate,
                        positions,
                        self.pool,
                        input_cache,
                    )
                )
            output = self.graphs[index].replay_boundary(hidden, residual, gate, positions, copied_inputs)
            if following is None:
                return output
            query, key, value, gate, residual = output
            # KV state is committed exactly once by the unchanged eager backend.
            hidden = following.layer.self_attn.attn(query, key, value)
            previous = following
        raise AssertionError("Missing final D-Quant decoder region")

    def _make_forward(self, index):
        def forward(layer, hidden_states, residual, positions=None, **kwargs):
            if index == 0:
                self._skip = False
                if not kwargs and self._supports(
                    hidden_states, residual, positions, get_forward_context().attn_metadata
                ):
                    output = self._run(hidden_states, residual, positions)
                    self._skip = True
                    return output
            elif self._skip:
                if index == len(self.layers) - 1:
                    self._skip = False
                return hidden_states, residual
            return self.originals[index](hidden_states=hidden_states, residual=residual, positions=positions, **kwargs)

        return forward


def install_dquant_boundary_graphs(model):
    """Install after the existing segment and attention graph wrappers."""
    if hasattr(model, "_dquant_boundary_chains"):
        return model._dquant_boundary_chains
    segments = getattr(model, "_dquant_decode_segments", ())
    attentions = getattr(model, "_dquant_attention_graphs", ())
    attention_map = {id(wrapper.layer): wrapper for wrapper in attentions}
    chains = []
    for module in model.modules():
        layers = getattr(module, "layers", None)
        if not isinstance(layers, torch.nn.ModuleList) or not layers or len(layers) % 4:
            continue
        if getattr(module, "start_layer", 0) != 0 or getattr(module, "end_layer", len(layers)) != len(layers):
            continue
        if getattr(module, "aux_hidden_state_layers", ()):
            continue
        if not all(
            getattr(layer, "layer_type", None) == ("full_attention" if index % 4 == 3 else "linear_attention")
            and not getattr(layer, "layer_scale", False)
            for index, layer in enumerate(layers)
        ):
            continue
        selected = [segment for segment in segments if any(segment.layers[0] is layer for layer in layers)]
        if len(selected) != len(layers) // 4 or any(len(segment.layers) != 3 for segment in selected):
            continue
        if not all(id(layer) in attention_map for layer in layers[3::4]):
            continue
        chains.append(
            DQuantBoundaryChain(
                layers,
                selected,
                [attention_map[id(layer)] for layer in layers[3::4]],
                torch.npu.graph_pool_handle(),
                owner=module,
            )
        )
    model._dquant_boundary_chains = chains
    return chains
