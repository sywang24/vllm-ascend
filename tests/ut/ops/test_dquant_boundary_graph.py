# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import vllm_ascend.ops.dquant_boundary_graph as boundary
import vllm_ascend.ops.dquant_segment_graph as segment


def make_metadata(slot):
    slots = [slot] if isinstance(slot, int) else slot
    count = len(slots)
    index = torch.tensor(slots, device="npu", dtype=torch.int32)
    return SimpleNamespace(
        num_actual_tokens=count,
        num_decodes=count,
        num_prefills=0,
        spec_sequence_masks=None,
        non_spec_state_indices_tensor=index,
        non_spec_decode_metadata=SimpleNamespace(
            actual_seq_lengths=torch.arange(count + 1, device="npu", dtype=torch.int32),
            causal_conv1d=SimpleNamespace(
                cache_indices=index.reshape(count, 1),
                query_start_loc=torch.arange(count + 1, device="npu", dtype=torch.int32),
            ),
        ),
    )


@pytest.mark.parametrize("has_residual", [False, True])
@pytest.mark.parametrize("shared_metadata", [False, True])
@pytest.mark.parametrize("batch_sequence", [(1, 1, 1, 1), (2, 2, 2, 2), (1, 2, 1, 2)])
def test_boundary_chain_preserves_state_slots_positions_and_eager_attention(
    has_residual, shared_metadata, batch_sequence
):
    torch.npu.set_device(0)
    torch.manual_seed(103)
    context = SimpleNamespace(attn_metadata={})
    calls = []

    class GDN(torch.nn.Module):
        layer_type = "linear_attention"
        layer_scale = False

        def __init__(self, index):
            super().__init__()
            self.linear_attn = SimpleNamespace(
                prefix=str(index),
                head_k_dim=128,
                head_v_dim=128,
                kv_cache=tuple(torch.randn(4, 128, device="npu", dtype=torch.bfloat16) for _ in range(2)),
            )
            context.attn_metadata[str(index)] = make_metadata(1)

        def forward(self, hidden_states, residual, positions=None):
            meta = context.attn_metadata[self.linear_attn.prefix]
            slot = meta.non_spec_state_indices_tensor.long()
            conv, state = self.linear_attn.kv_cache
            value = conv.index_select(0, slot) * 0.1 + hidden_states
            conv.index_copy_(0, slot, value)
            value = value + state.index_select(0, slot) * 0.2
            state.index_copy_(0, slot, value)
            residual = hidden_states if residual is None else residual + hidden_states
            return value * 0.3, residual

    class Full(torch.nn.Module):
        layer_type = "full_attention"
        layer_scale = False

        def __init__(self, index):
            super().__init__()

            def attention(query, key, value):
                calls.append(index)
                return query + key * value

            self.self_attn = SimpleNamespace(attn=attention)
            attention.layer_name = str(index)
            context.attn_metadata[str(index)] = make_metadata(1)

        def before_attention(self, hidden, residual, positions):
            residual = hidden if residual is None else residual + hidden
            query = hidden * 0.2 + positions.reshape(-1, 1).to(hidden.dtype) * 0.01
            return query, hidden * 0.1, hidden * 0.3, hidden * 0.4, residual

        def after_attention(self, attended, gate, residual):
            return attended * torch.sigmoid(gate) + residual * 0.2, residual

        def forward(self, hidden_states, residual, positions=None):
            q, k, v, gate, residual = self.before_attention(hidden_states, residual, positions)
            return self.after_attention(self.self_attn.attn(q, k, v), gate, residual)

    layers = torch.nn.ModuleList([Full(i) if i % 4 == 3 else GDN(i) for i in range(8)])
    eager = [layer.forward for layer in layers]
    segments = [segment.DQuantDecodeSegment(layers[i : i + 3], None) for i in (0, 4)]
    attentions = [
        SimpleNamespace(
            layer=layers[i], before_attention=layers[i].before_attention, after_attention=layers[i].after_attention
        )
        for i in (3, 7)
    ]
    chain = boundary.DQuantBoundaryChain(layers, segments, attentions, None)
    states = [
        state for layer in layers if layer.layer_type == "linear_attention" for state in layer.linear_attn.kv_cache
    ]
    positions = torch.tensor([7], device="npu")
    with (
        patch.object(boundary, "get_forward_context", return_value=context),
        patch.object(segment, "get_forward_context", return_value=context),
    ):
        for slot, batch in zip((1, 2, 3, 1), batch_sequence):
            shared = make_metadata([slot, (slot + 1) % 4][:batch])
            for index in range(8):
                context.attn_metadata[str(index)] = (
                    shared if shared_metadata else make_metadata([slot, (slot + 1) % 4][:batch])
                )
            hidden = torch.randn(batch, 128, device="npu", dtype=torch.bfloat16)
            residual = torch.randn_like(hidden) if has_residual else None
            positions = torch.arange(batch, device="npu") + slot + 7
            saved = [state.clone() for state in states]
            expected = (hidden, residual)
            for forward in eager:
                expected = forward(*expected, positions=positions)
            expected = tuple(t.clone() for t in expected)
            expected_states = [state.clone() for state in states]
            for state, old in zip(states, saved):
                state.copy_(old)
            calls.clear()
            actual = (hidden, residual)
            for layer in layers:
                actual = layer(*actual, positions=positions)
            assert calls == [3, 7]
            for result, reference in zip(actual, expected):
                torch.testing.assert_close(result, reference, atol=0, rtol=0)
            for result, reference in zip(states, expected_states):
                torch.testing.assert_close(result, reference, atol=0, rtol=0)
        assert len(chain.graphs) == 3
        assert set(chain.graphs_by_batch) == set(batch_sequence)
        for batch, graphs in chain.graphs_by_batch.items():
            assert all(graph.replays == batch_sequence.count(batch) for graph in graphs)
            if shared_metadata:
                assert len({id(saved[index]) for graph in graphs for _, saved in graph.inputs for index in (0, 2)}) == 2
        if shared_metadata:
            context.attn_metadata["4"] = make_metadata([1, 2][: batch_sequence[-1]])
            assert not chain._supports(hidden, residual, positions, context.attn_metadata)
        replays = [graph.replays for graph in chain.graphs]
        for meta in context.attn_metadata.values():
            meta.num_prefills = 1
        calls.clear()
        for layer in layers:
            hidden, residual = layer(hidden, residual, positions=positions)
        assert calls == [3, 7]
        assert [graph.replays for graph in chain.graphs] == replays


def test_installer_rejects_partial_or_unsupported_layer_patterns():
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(4)])
    assert boundary.install_dquant_boundary_graphs(model) == []
    assert boundary.install_dquant_boundary_graphs(model) is model._dquant_boundary_chains
