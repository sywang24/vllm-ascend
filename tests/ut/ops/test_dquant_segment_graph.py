from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

import vllm_ascend.ops.dquant_segment_graph as module


def metadata(slot):
    index = torch.tensor([slot], device="npu", dtype=torch.int32)
    return SimpleNamespace(
        num_actual_tokens=1,
        num_decodes=1,
        num_prefills=0,
        spec_sequence_masks=None,
        non_spec_state_indices_tensor=index,
        non_spec_decode_metadata=SimpleNamespace(
            actual_seq_lengths=torch.tensor([0, 1], device="npu", dtype=torch.int32),
            causal_conv1d=SimpleNamespace(
                cache_indices=index.reshape(1, 1),
                query_start_loc=torch.tensor([0, 1], device="npu", dtype=torch.int32),
            ),
        ),
    )


@pytest.mark.parametrize("has_residual", [False, True])
def test_segment_preserves_capture_state_and_dynamic_slots(has_residual):
    torch.npu.set_device(0)
    torch.manual_seed(59)
    context = SimpleNamespace(attn_metadata={})
    layers, originals = [], []
    for index in range(3):
        attention = SimpleNamespace(
            prefix=str(index),
            _dquant_gdn_graph_enabled=True,
            kv_cache=(torch.randn(4, 128, device="npu"), torch.randn(4, 128, device="npu")),
        )
        layer = SimpleNamespace(linear_attn=attention)
        layers.append(layer)
        context.attn_metadata[attention.prefix] = metadata(1)

        def make_original(attention):
            def forward(hidden_states, residual, positions):
                meta = context.attn_metadata[attention.prefix]
                slot = meta.non_spec_state_indices_tensor.long()
                conv, state = attention.kv_cache
                value = conv.index_select(0, slot) * 0.1 + hidden_states
                conv.index_copy_(0, slot, value)
                value = value + state.index_select(0, slot) * 0.2
                state.index_copy_(0, slot, value)
                residual = hidden_states if residual is None else residual + hidden_states
                return value * 0.3 + positions.reshape(-1, 1), residual

            return forward

        originals.append(make_original(attention))
    hidden = torch.randn(1, 128, device="npu")
    residual = torch.randn_like(hidden) if has_residual else None
    positions = torch.tensor([7], device="npu")
    states = [state for layer in layers for state in layer.linear_attn.kv_cache]
    initial = [state.clone() for state in states]
    previous = context.attn_metadata.copy()
    with patch.object(module, "get_forward_context", return_value=context):
        graph = module.DQuantSegmentGraph(layers, originals, hidden, residual, positions, None)
        for actual, expected in zip(states, initial):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        for prefix, meta in previous.items():
            assert context.attn_metadata[prefix] is meta
        for slot in (1, 2, 3, 1):
            for prefix in context.attn_metadata:
                context.attn_metadata[prefix] = metadata(slot)
            hidden = torch.randn_like(hidden)
            residual = torch.randn_like(hidden) if has_residual else None
            positions = positions + 1
            saved = [state.clone() for state in states]
            expected = (hidden, residual)
            for original in originals:
                expected = original(*expected, positions)
            expected = tuple(t.clone() for t in expected)
            expected_states = [state.clone() for state in states]
            for state, before in zip(states, saved):
                state.copy_(before)
            actual = graph.replay(hidden, residual, positions)
            for result, reference in zip(actual, expected):
                torch.testing.assert_close(result, reference, atol=0, rtol=0)
            for state, reference in zip(states, expected_states):
                torch.testing.assert_close(state, reference, atol=0, rtol=0)
        assert graph.matches(hidden, residual, positions, context.attn_metadata)
        layers[0].linear_attn.kv_cache = tuple(t.clone() for t in layers[0].linear_attn.kv_cache)
        assert not graph.matches(hidden, residual, positions, context.attn_metadata)


def test_segment_wrapper_skips_only_after_successful_replay():
    torch.npu.set_device(0)
    hidden = torch.ones((1, 128), device="npu", dtype=torch.bfloat16)
    positions = torch.tensor([1], device="npu")
    meta = metadata(0)
    layers = []
    for index in range(3):
        layers.append(
            SimpleNamespace(
                linear_attn=SimpleNamespace(prefix=str(index), head_k_dim=128, head_v_dim=128),
                forward=Mock(return_value=(hidden, hidden)),
            )
        )
    segment = module.DQuantDecodeSegment(layers, None)
    graph = Mock()
    graph.matches.return_value = True
    graph.replay.return_value = (hidden + 1, hidden + 2)
    segment.graph = graph
    segment.graphs_by_batch[1] = graph
    context = SimpleNamespace(attn_metadata={str(index): meta for index in range(3)})

    def run():
        result = hidden, hidden
        for layer in layers:
            result = layer.forward(hidden_states=result[0], residual=result[1], positions=positions)
        return result

    with patch.object(module, "get_forward_context", return_value=context):
        result = run()
        assert all(actual is expected for actual, expected in zip(result, graph.replay.return_value))
        for original in segment.originals:
            original.assert_not_called()
        for name, value in (
            ("num_actual_tokens", 2),
            ("num_prefills", 1),
            ("num_decodes", 2),
            ("spec_sequence_masks", torch.ones(1)),
        ):
            previous = getattr(meta, name)
            setattr(meta, name, value)
            run()
            setattr(meta, name, previous)
        segment.enabled = False
        run()
        segment.enabled = True
        graph.matches.return_value = False
        run()
    assert graph.replay.call_count == 1
    for original in segment.originals:
        assert original.call_count == 6


def test_segment_capture_failure_restores_states_and_metadata():
    torch.npu.set_device(0)
    attention = SimpleNamespace(
        prefix="0",
        _dquant_gdn_graph_enabled=True,
        kv_cache=(torch.randn(4, 128, device="npu"), torch.randn(4, 128, device="npu")),
    )
    layer = SimpleNamespace(linear_attn=attention)
    meta = metadata(1)
    context = SimpleNamespace(attn_metadata={"0": meta})
    initial = tuple(t.clone() for t in attention.kv_cache)
    hidden = torch.ones(1, 128, device="npu")

    def fail(graph):
        graph.states[0][0][1].add_(1)
        graph.states[0][1][1].add_(2)
        raise RuntimeError("capture failed")

    with (
        patch.object(module, "get_forward_context", return_value=context),
        patch.object(module.DQuantSegmentGraph, "_core", fail),
        pytest.raises(RuntimeError, match="capture failed"),
    ):
        module.DQuantSegmentGraph([layer], [Mock()], hidden, None, hidden, None)
    assert context.attn_metadata["0"] is meta
    assert attention._dquant_gdn_graph_enabled is True
    for actual, expected in zip(attention.kv_cache, initial):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_install_groups_contiguous_layers_and_is_idempotent():
    import vllm.model_executor.models.qwen3_5 as qwen

    class Decoder(torch.nn.Module):
        def __init__(self, kind):
            super().__init__()
            self.layer_type = kind

        def forward(self, hidden_states, residual, positions=None):
            return hidden_states, residual

    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList(
        Decoder(kind)
        for kind in (
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "linear_attention",
        )
    )
    full_forward = model.layers[3].forward
    with (
        patch.object(qwen, "Qwen3_5DecoderLayer", Decoder),
        patch.object(torch.npu, "graph_pool_handle", return_value=None),
    ):
        segments = module.install_dquant_segment_graphs(model)
        assert [len(segment.layers) for segment in segments] == [3, 1, 2]
        assert module.install_dquant_segment_graphs(model) is segments
    assert model.layers[3].forward == full_forward
