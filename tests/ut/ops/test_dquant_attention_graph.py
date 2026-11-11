from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

import vllm_ascend.ops.dquant_attention_graph as module


def test_static_call_tracks_dynamic_inputs_and_optional_residual():
    torch.npu.set_device(0)
    weight = torch.randn(32, 32, device="npu", dtype=torch.bfloat16)

    def forward(hidden, residual):
        result = hidden @ weight
        return result if residual is None else result + residual

    for residual in (None, torch.randn(1, 32, device="npu", dtype=torch.bfloat16)):
        hidden = torch.randn(1, 32, device="npu", dtype=torch.bfloat16)
        graph = module.DQuantStaticCall(forward, (hidden, residual), None)
        for _ in range(4):
            hidden = torch.randn_like(hidden)
            if residual is not None:
                residual = torch.randn_like(residual)
            expected = forward(hidden, residual)
            assert graph.matches((hidden, residual))
            torch.testing.assert_close(graph.replay((hidden, residual)), expected, atol=0, rtol=0)
        assert not graph.matches((hidden,))
        assert not graph.matches((hidden.repeat(2, 1), residual))
        assert not graph.matches((hidden.float(), residual))
        assert not graph.matches((hidden, hidden if residual is None else None))


def test_attention_graph_falls_back_without_duplicate_cache_updates():
    torch.npu.set_device(0)
    hidden = torch.ones(1, 32, device="npu", dtype=torch.bfloat16)
    attention = SimpleNamespace(attn=Mock(layer_name="attention"))
    layer = SimpleNamespace(self_attn=attention, layer_scale=False, forward=Mock(return_value=(hidden, hidden)))
    wrapper = module.DQuantAttentionDecoderGraphs(layer, None)
    wrapper.enabled = True
    meta = SimpleNamespace(num_actual_tokens=1, num_decodes=1, num_prefills=0)
    context = SimpleNamespace(attn_metadata={"attention": meta})
    wrapper.pre = Mock()
    wrapper.pre.matches.return_value = True
    wrapper.pre.replay.return_value = (hidden, hidden, hidden, hidden, hidden)
    wrapper.post = Mock()
    wrapper.post.matches.return_value = False
    wrapper.pre_by_batch[1] = wrapper.pre
    wrapper.post_by_batch[1] = wrapper.post
    wrapper.after_attention = Mock(return_value=(hidden + 1, hidden + 2))
    with (
        patch.object(module, "get_forward_context", return_value=context),
        patch.object(module, "_uses_multimodal_rope", return_value=True),
    ):
        result = layer.forward(hidden, hidden, positions=hidden)
        assert result is wrapper.after_attention.return_value
        attention.attn.assert_called_once()
        wrapper.original.assert_not_called()
        meta.num_prefills = 1
        layer.forward(hidden, hidden, positions=hidden)
        wrapper.original.assert_called_once()
        assert attention.attn.call_count == 1


@pytest.mark.parametrize("field,value", [("num_actual_tokens", 2), ("num_decodes", 2), ("num_prefills", 1)])
def test_attention_graph_rejects_non_single_decode(field, value):
    torch.npu.set_device(0)
    hidden = torch.ones(1, 32, device="npu", dtype=torch.bfloat16)
    layer = SimpleNamespace(
        self_attn=SimpleNamespace(attn=Mock(layer_name="attention")), layer_scale=False, forward=Mock()
    )
    wrapper = module.DQuantAttentionDecoderGraphs(layer, None)
    meta = SimpleNamespace(num_actual_tokens=1, num_decodes=1, num_prefills=0)
    setattr(meta, field, value)
    context = SimpleNamespace(attn_metadata={"attention": meta})
    with patch.object(module, "get_forward_context", return_value=context):
        layer.forward(hidden, hidden, positions=hidden)
    wrapper.original.assert_called_once()
    assert wrapper.pre is None
    layer.self_attn.attn.assert_not_called()


def test_attention_graph_installer_targets_supported_layers_once():
    import vllm.model_executor.models.qwen3_5 as qwen

    class Decoder(torch.nn.Module):
        def __init__(self, kind, scale=False):
            super().__init__()
            self.layer_type = kind
            self.layer_scale = scale
            self.self_attn = SimpleNamespace()

        def forward(self, hidden_states, residual, positions=None):
            return hidden_states, residual

    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList(
        [Decoder("linear_attention"), Decoder("full_attention"), Decoder("full_attention", scale=True)]
    )
    with (
        patch.object(qwen, "Qwen3_5DecoderLayer", Decoder),
        patch.object(module, "_uses_multimodal_rope", return_value=True),
        patch.object(torch.npu, "graph_pool_handle", return_value=None),
    ):
        graphs = module.install_dquant_attention_graphs(model)
        assert len(graphs) == 1
        assert graphs[0].layer is model.layers[1]
        assert module.install_dquant_attention_graphs(model) is graphs
