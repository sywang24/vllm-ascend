from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor, MambaSpec

from vllm_ascend.utils import adapt_patch

# Worker imports require the platform compatibility patches first.
adapt_patch()

from vllm_ascend.attention.utils import AscendCommonAttentionMetadata  # noqa: E402
from vllm_ascend.core.kv_cache_interface import DQuantAttentionSpec  # noqa: E402
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner  # noqa: E402
from vllm_ascend.worker.worker import NPUWorker  # noqa: E402


def test_metadata_host_mirrors_survive_unpadding():
    lengths = torch.tensor([3, 5], dtype=torch.int32)
    computed = torch.tensor([1, 2], dtype=torch.int32)
    metadata = AscendCommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 2, 5], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2, 5], dtype=torch.int32),
        seq_lens=lengths,
        num_reqs=2,
        num_actual_tokens=5,
        max_query_len=3,
        max_seq_len=5,
        block_table_tensor=torch.zeros((2, 1), dtype=torch.int32),
        slot_mapping=torch.arange(5),
        seq_lens_cpu=lengths,
        _seq_lens_cpu=lengths,
        _num_computed_tokens_cpu=computed,
        dcp_local_seq_lens_cpu=lengths,
        actual_seq_lengths_q=[2, 5],
    )
    unpadded = metadata.unpadded(2, 1)
    assert torch.equal(unpadded._seq_lens_cpu, lengths[:1])
    assert torch.equal(unpadded._num_computed_tokens_cpu, computed[:1])
    assert torch.equal(unpadded.dcp_local_seq_lens_cpu, lengths[:1])


def test_standardized_descriptor_keeps_layers_independent():
    spec = DQuantAttentionSpec(block_size=256, num_kv_heads=4, head_size=256, dtype=torch.bfloat16)
    state = MambaSpec(block_size=256, shapes=((4, 2), (2, 2)), dtypes=(torch.bfloat16, torch.float32))
    config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[
            KVCacheTensor(
                size=4 * spec.page_size_bytes,
                layers=["attn0", "attn1"],
                layer_stride=2 * spec.page_size_bytes,
                block_stride=spec.page_size_bytes,
            ),
            KVCacheTensor(
                size=4 * spec.page_size_bytes,
                layers=["linear_attn"],
                layer_stride=2 * state.page_size_bytes,
                block_stride=state.page_size_bytes,
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["attn0", "attn1"], kv_cache_spec=spec),
            KVCacheGroupSpec(layer_names=["linear_attn"], kv_cache_spec=state),
        ],
    )
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.device = torch.device("cpu")
    runner.runner_only_attn_layers = set()
    runner.vllm_config = SimpleNamespace(kv_transfer_config=None, compilation_config=None)
    raw = runner._allocate_dquant_layer_tensors(config)
    key0, value0 = raw["attn0"]
    key1, value1 = raw["attn1"]
    assert key0.numel() == 2 * spec.key_page_size_bytes
    assert value0.numel() == 2 * spec.value_page_size_bytes
    assert raw["linear_attn"].numel() == 2 * state.page_size_bytes
    key0.fill_(7)
    assert torch.count_nonzero(key1) == 0
    assert torch.count_nonzero(value1) == 0
    assert torch.count_nonzero(raw["linear_attn"]) == 0


def test_private_layer_budget_accounts_for_every_group():
    worker = NPUWorker.__new__(NPUWorker)
    worker.vllm_config = SimpleNamespace()
    worker.get_kv_cache_spec = lambda: {}
    groups = [
        SimpleNamespace(
            layer_names=["a", "b"], kv_cache_spec=SimpleNamespace(page_size_bytes=100, real_page_size_bytes=100)
        ),
        SimpleNamespace(layer_names=["c"], kv_cache_spec=SimpleNamespace(page_size_bytes=50, real_page_size_bytes=50)),
    ]
    with (
        patch("vllm_ascend.envs.VLLM_ASCEND_ENABLE_DQUANT", True),
        patch("vllm.v1.core.kv_cache_utils.get_kv_cache_groups", return_value=groups),
    ):
        assert worker._scale_dquant_kv_budget(1000) == 800


def test_dense_reference_ignores_planner_padding_and_keeps_layers_independent(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_DQUANT_DENSE_REFERENCE", "1")
    spec = FullAttentionSpec(
        block_size=256,
        num_kv_heads=4,
        head_size=256,
        dtype=torch.bfloat16,
        page_size_padded=2 * 256 * 4 * 256 * 2 + 4096,
    )
    names = ["attn0", "attn1"]
    config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[
            KVCacheTensor(
                size=4 * spec.page_size_bytes,
                layers=names,
                layer_stride=2 * spec.page_size_bytes,
                block_stride=spec.page_size_bytes,
            )
        ],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=names, kv_cache_spec=spec)],
    )
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.device = torch.device("cpu")
    runner.runner_only_attn_layers = set()
    runner.vllm_config = SimpleNamespace(
        additional_config={"dquant_dense_reference": True},
        kv_transfer_config=None,
    )
    runner.use_compress = False
    runner.use_hybrid_blocks = True
    backend = SimpleNamespace(get_supported_kernel_block_sizes=lambda: [128])
    group = SimpleNamespace(backend=backend, kv_cache_spec=spec, layer_names=names)
    runner._kv_cache_spec_attn_group_iterator = lambda: iter([group])
    raw = runner._allocate_kv_cache_tensors(config)
    caches = runner._reshape_kv_cache_tensors(config, raw)
    for key, value in caches.values():
        assert key.shape == value.shape == (4, 128, 4, 256)
        assert key.dtype == value.dtype == torch.bfloat16
    caches["attn0"][0].fill_(7)
    assert torch.count_nonzero(caches["attn1"][0]) == 0
    assert torch.count_nonzero(caches["attn0"][1]) == 0


def test_dense_reference_scales_budget_even_when_quantization_is_disabled(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_DQUANT_DENSE_REFERENCE", "1")
    worker = NPUWorker.__new__(NPUWorker)
    worker.vllm_config = SimpleNamespace(additional_config={"dquant_dense_reference": True})
    worker.get_kv_cache_spec = lambda: {}
    groups = [
        SimpleNamespace(
            layer_names=["a", "b"], kv_cache_spec=SimpleNamespace(page_size_bytes=100, real_page_size_bytes=100)
        ),
        SimpleNamespace(layer_names=["c"], kv_cache_spec=SimpleNamespace(page_size_bytes=50, real_page_size_bytes=50)),
    ]
    with (
        patch("vllm_ascend.envs.VLLM_ASCEND_ENABLE_DQUANT", False),
        patch("vllm.v1.core.kv_cache_utils.get_kv_cache_groups", return_value=groups),
    ):
        assert worker._scale_dquant_kv_budget(1000) == 800


def test_window_reservation_is_charged_before_planner_scaling():
    worker = NPUWorker.__new__(NPUWorker)
    worker.vllm_config = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=4))
    worker.get_kv_cache_spec = lambda: {}
    spec = DQuantAttentionSpec(
        block_size=128,
        num_kv_heads=4,
        head_size=256,
        dtype=torch.bfloat16,
        dquant_recent_window=64,
        dquant_key_window=64,
    )
    groups = [
        SimpleNamespace(layer_names=["a", "b"], kv_cache_spec=spec),
        SimpleNamespace(
            layer_names=["state"], kv_cache_spec=SimpleNamespace(page_size_bytes=100, real_page_size_bytes=100)
        ),
    ]
    reserved = 2 * 4 * spec.recent_state_bytes
    available = 10_000_000
    with (
        patch("vllm_ascend.envs.VLLM_ASCEND_ENABLE_DQUANT", True),
        patch("vllm.v1.core.kv_cache_utils.get_kv_cache_groups", return_value=groups),
    ):
        assert worker._scale_dquant_kv_budget(available) == (
            (available - reserved) * (2 * spec.page_size_bytes) // (2 * spec.page_size_bytes + 100)
        )
        with pytest.raises(ValueError, match="reservation exceeds"):
            worker._scale_dquant_kv_budget(reserved)
