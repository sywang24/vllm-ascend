# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_ascend.attention.nova_kv import NovaAttentionBackend, NovaAttentionImpl, NovaMetadataBuilder
from vllm_ascend.attention.nova_kv_v2 import NovaV2AttentionBackend, NovaV2AttentionImpl
from vllm_ascend.patch.worker import patch_nova_kv as patch


def make_config(options=None):
    return SimpleNamespace(
        additional_config={} if options is None else {"nova_kv": options},
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type="qwen3_5_text")),
        use_v2_model_runner=False,
        scheduler_config=SimpleNamespace(async_scheduling=False),
        compilation_config=SimpleNamespace(cudagraph_mode=SimpleNamespace(name="NONE")),
        cache_config=SimpleNamespace(enable_prefix_caching=False, cache_dtype="auto"),
        speculative_config=None,
        kv_transfer_config=None,
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1, prefill_context_parallel_size=1, pipeline_parallel_size=1
        ),
    )


def test_disabled_constructor_keeps_original_arguments(monkeypatch):
    calls = []
    monkeypatch.setattr(patch, "get_current_vllm_config", lambda: make_config())
    monkeypatch.setattr(patch, "_ORIGINAL_INIT", lambda *args, **kwargs: calls.append((args, kwargs)))
    layer = SimpleNamespace()
    patch.nova_init(layer, 4, 256, 0.0625, prefix="model.layers.0.self_attn.attn")
    assert calls == [((layer, 4, 256, 0.0625), {"prefix": "model.layers.0.self_attn.attn"})]


@pytest.mark.parametrize("use_v2", [False, True])
def test_backend_selected_before_original_initialization(monkeypatch, use_v2):
    config = make_config({"artifact_path": "/artifacts", "direct_library_root": "/kernel"})
    config.use_v2_model_runner = use_v2
    observed = []
    monkeypatch.setattr(patch, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(patch, "enable_sp", lambda config: False)
    monkeypatch.setattr(patch, "_ORIGINAL_INIT", lambda *args, **kwargs: observed.append(kwargs["attn_backend"]))
    layer = SimpleNamespace()
    patch.nova_init(layer, 4, 256, 0.0625)
    assert observed == [NovaV2AttentionBackend if use_v2 else NovaAttentionBackend]
    assert layer.align_kv_cache_with_mamba is False


@pytest.mark.parametrize("backend", [NovaAttentionBackend, NovaV2AttentionBackend])
def test_packed_spec_preserves_allocator_padding(monkeypatch, backend):
    dense = FullAttentionSpec(
        block_size=128, num_kv_heads=2, head_size=256, dtype=torch.bfloat16, page_size_padded=262144
    )
    monkeypatch.setattr(patch, "_ORIGINAL_SPEC", lambda self, config: dense)
    layer = SimpleNamespace(attn_backend=backend, align_kv_cache_with_mamba=False)
    result = patch.nova_spec(layer, make_config())
    assert result.dtype == torch.uint8
    assert result.state_content_bytes == 134
    assert result.page_size_padded == 262144
    assert result.block_size == dense.block_size
    assert NovaAttentionBackend.customize_spec(result) == result


@pytest.mark.parametrize(
    "field,value",
    [
        ("speculative_config", object()),
        ("kv_transfer_config", object()),
    ],
)
def test_unsupported_runtime_rejected(monkeypatch, field, value):
    config = make_config()
    setattr(config, field, value)
    monkeypatch.setattr(patch, "enable_sp", lambda config: False)
    with pytest.raises(ValueError):
        patch._validate(config)


@pytest.mark.parametrize("mode", ["NONE", "FULL_DECODE_ONLY"])
@pytest.mark.parametrize("use_v2", [False, True])
def test_graph_modes_allowed_for_compressed_backend(monkeypatch, mode, use_v2):
    config = make_config({"artifact_path": "/artifacts", "direct_library_root": "/kernel"})
    config.use_v2_model_runner = use_v2
    config.compilation_config.cudagraph_mode.name = mode
    monkeypatch.setattr(patch, "enable_sp", lambda config: False)
    monkeypatch.setattr(patch, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(patch, "_ORIGINAL_INIT", lambda *args, **kwargs: None)
    patch.nova_init(SimpleNamespace(), 4, 256, 0.0625)
    assert NovaMetadataBuilder.get_cudagraph_support(config, None) == AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE


@pytest.mark.parametrize("mode", ["PIECEWISE", "FULL", "FULL_AND_PIECEWISE"])
def test_unsupported_graph_modes_rejected(monkeypatch, mode):
    config = make_config({"artifact_path": "/artifacts", "direct_library_root": "/kernel"})
    config.compilation_config.cudagraph_mode.name = mode
    monkeypatch.setattr(patch, "enable_sp", lambda config: False)
    monkeypatch.setattr(patch, "get_current_vllm_config", lambda: config)
    with pytest.raises(ValueError, match="NONE or FULL_DECODE_ONLY"):
        patch.nova_init(SimpleNamespace(), 4, 256, 0.0625)


@pytest.mark.parametrize("field,value", [("async_scheduling", True), ("pipeline_parallel_size", 2)])
def test_recovery_rejects_overlapping_batches(monkeypatch, field, value):
    config = make_config({"artifact_path": "/artifacts", "direct_library_root": "/kernel"})
    setattr(config.scheduler_config if field == "async_scheduling" else config.parallel_config, field, value)
    monkeypatch.setattr(patch, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(patch, "enable_sp", lambda config: False)
    with pytest.raises(ValueError, match="synchronous scheduling"):
        patch.nova_init(SimpleNamespace(), 4, 256, 0.0625)


def test_calibration_capture_still_requires_eager(monkeypatch):
    config = make_config({"capture_path": "/capture"})
    config.compilation_config.cudagraph_mode.name = "FULL_DECODE_ONLY"
    monkeypatch.setattr(patch, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(patch, "enable_sp", lambda config: False)
    with pytest.raises(ValueError, match="calibration capture requires --enforce-eager"):
        patch.nova_init(SimpleNamespace(), 4, 256, 0.0625)


@pytest.mark.parametrize("impl_cls", [NovaAttentionImpl, NovaV2AttentionImpl])
def test_memory_profile_exercises_codec_and_lookup_temporaries(impl_cls):
    calls = []
    codec = SimpleNamespace(
        key_inverse=torch.eye(256)[None],
        key_mean=torch.zeros(1, 256),
        codebooks=torch.zeros(1, 64, 256, 4),
        _books_float=torch.zeros(1, 64, 256, 4),
        encode=lambda key, value, **kwargs: calls.append(("encode", len(key), kwargs)),
    )
    impl = object.__new__(impl_cls)
    impl.head_size = 256
    impl.num_kv_heads = 1
    impl._limit = 2
    impl._options = SimpleNamespace(codec_batch_size=3)
    impl._codec = lambda layer, device: codec
    impl._ensure_runtime = lambda device, dtype: None
    impl._cold = SimpleNamespace(prepare_lut=lambda query, books: calls.append(("lut", tuple(query.shape))))
    query = torch.ones(5, 2, 256)
    key = torch.ones(5, 1, 256)
    output = torch.empty_like(query)
    impl.forward(SimpleNamespace(head_size_v=256), query, key, key, None, None, output=output)
    assert calls == [("encode", 5, {"chunk_size": 3}), ("lut", (2, 1, 2, 256))]
    assert not output.any()
