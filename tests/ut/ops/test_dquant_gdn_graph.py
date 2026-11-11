from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.ops.gdn import AscendGatedDeltaNetAttention
from vllm_ascend.ops.gdn_decode_graph import SingleTokenGDNGraph
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


def metadata(slot):
    return SimpleNamespace(
        num_actual_tokens=1,
        num_decodes=1,
        num_prefills=0,
        spec_sequence_masks=None,
        non_spec_state_indices_tensor=torch.tensor([slot], device="npu", dtype=torch.int32),
        non_spec_decode_metadata=SimpleNamespace(
            actual_seq_lengths=torch.tensor([0, 1], device="npu", dtype=torch.int32),
            causal_conv1d=SimpleNamespace(
                query_start_loc=torch.tensor([0, 1], device="npu", dtype=torch.int32),
                cache_indices=torch.tensor([[slot]], device="npu", dtype=torch.int32),
            ),
        ),
    )


@pytest.fixture
def inputs():
    torch.npu.set_device(0)
    init_device_properties_triton()
    torch.manual_seed(31)
    nk, nv, dim, slots = 8, 24, 128, 4
    channels = (2 * nk + nv) * dim

    def split(x):
        q, k, v = x.split([nk * dim, nk * dim, nv * dim], dim=-1)
        return q.view(1, 1, nk, dim), k.view(1, 1, nk, dim), v.view(1, 1, nv, dim)

    layer = SimpleNamespace(
        num_v_heads=nv,
        tp_size=1,
        head_k_dim=dim,
        head_v_dim=dim,
        activation="silu",
        conv1d=SimpleNamespace(
            weight=torch.randn((channels, 1, 4), device="npu", dtype=torch.bfloat16) * 0.05,
            bias=None,
        ),
        A_log=torch.zeros(nv, device="npu"),
        dt_bias=torch.zeros(nv, device="npu"),
        kv_cache=(
            torch.randn((slots, 3, channels), device="npu", dtype=torch.bfloat16) * 0.01,
            torch.randn((slots, nv, dim, dim), device="npu") * 0.01,
        ),
        rearrange_mixed_qkv=split,
        _dquant_gdn_graph_enabled=True,
    )
    x = torch.randn((1, channels), device="npu", dtype=torch.bfloat16)
    b = torch.randn((1, nv), device="npu", dtype=torch.bfloat16)
    return layer, x, b, torch.randn_like(b)


def test_integrated_graph_restores_state_and_matches_dynamic_eager_replays(inputs):
    layer, x, b, a = inputs
    initial = tuple(t.clone() for t in layer.kv_cache)
    output = torch.empty((1, 24, 128), device="npu", dtype=x.dtype)
    context = SimpleNamespace(additional_kwargs={})
    assert AscendGatedDeltaNetAttention._try_dquant_decode_graph(layer, context, metadata(1), x, b, a, output)
    graph = layer._dquant_decode_graph
    # Compare the first integrated replay against one eager state transition.
    first_state = tuple(t.clone() for t in layer.kv_cache)
    first_output = output.clone()
    for actual, saved in zip(layer.kv_cache, initial):
        actual.copy_(saved)
    graph._core()
    torch.testing.assert_close(graph.output, first_output, atol=0, rtol=0)
    for actual, expected in zip(layer.kv_cache, first_state):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for slot in (1, 2, 1, 3, 2):
        new_x, new_b, new_a = torch.randn_like(x), torch.randn_like(b), torch.randn_like(a)
        meta = metadata(slot)
        previous = tuple(t.clone() for t in layer.kv_cache)
        graph.mixed_qkv.copy_(new_x)
        graph.a.copy_(new_a)
        graph.b.copy_(new_b)
        graph.conv_indices.copy_(meta.non_spec_decode_metadata.causal_conv1d.cache_indices)
        graph.state_indices.copy_(meta.non_spec_state_indices_tensor)
        graph._core()
        expected_output = graph.output.clone()
        expected_state = tuple(t.clone() for t in layer.kv_cache)
        for actual, saved in zip(layer.kv_cache, previous):
            actual.copy_(saved)
        assert AscendGatedDeltaNetAttention._try_dquant_decode_graph(
            layer, SimpleNamespace(), meta, new_x, new_b, new_a, output
        )
        torch.npu.synchronize()
        torch.testing.assert_close(output, expected_output, atol=0, rtol=0)
        for actual, expected in zip(layer.kv_cache, expected_state):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert graph.replays == 6
    layer.kv_cache = tuple(t.clone() for t in layer.kv_cache)
    assert not AscendGatedDeltaNetAttention._try_dquant_decode_graph(layer, context, metadata(1), x, b, a, output)


def test_graph_warmup_failure_restores_recurrent_states(inputs):
    layer, x, b, a = inputs
    initial = tuple(t.clone() for t in layer.kv_cache)
    core = SingleTokenGDNGraph._core

    def failed(graph):
        core(graph)
        raise RuntimeError("warmup failure")

    with patch.object(SingleTokenGDNGraph, "_core", failed), pytest.raises(RuntimeError, match="warmup failure"):
        SingleTokenGDNGraph(layer, x, b, a, metadata(1))
    for actual, expected in zip(layer.kv_cache, initial):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_forward_core_works_when_patched_onto_an_unrelated_layer(inputs):
    import vllm_ascend.ops.gdn as gdn

    layer, x, b, a = inputs
    layer.prefix = "test_layer"
    assert not hasattr(layer, "_try_dquant_decode_graph")
    output = torch.empty((1, 24, 128), device="npu", dtype=x.dtype)
    context = SimpleNamespace(attn_metadata={layer.prefix: metadata(1)}, additional_kwargs={})
    with (
        patch.object(gdn, "get_forward_context", return_value=context),
        patch.object(gdn, "GDNAttentionMetadata", SimpleNamespace),
    ):
        AscendGatedDeltaNetAttention._forward_core(layer, x, b, a, output)
    assert layer._dquant_decode_graph.replays == 1
    assert torch.isfinite(output).all().item()


def test_graph_falls_back_for_prefill_multi_token_and_speculation(inputs):
    layer, x, b, a = inputs
    output = torch.empty((1, 24, 128), device="npu", dtype=x.dtype)
    for name, value in (
        ("num_prefills", 1),
        ("num_decodes", 2),
        ("num_actual_tokens", 2),
        ("spec_sequence_masks", torch.ones(1, device="npu")),
    ):
        meta = metadata(1)
        setattr(meta, name, value)
        assert not AscendGatedDeltaNetAttention._try_dquant_decode_graph(
            layer, SimpleNamespace(), meta, x, b, a, output
        )
    layer._dquant_gdn_graph_enabled = False
    assert not AscendGatedDeltaNetAttention._try_dquant_decode_graph(
        layer, SimpleNamespace(), metadata(1), x, b, a, output
    )
