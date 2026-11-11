from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, KVCacheTensor, MambaSpec

from vllm_ascend.core.kv_cache_interface import DQuantAttentionSpec
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
from vllm_ascend.worker.worker import NPUWorker


def specs():
    attention = DQuantAttentionSpec(
        block_size=128,
        num_kv_heads=2,
        head_size=256,
        dtype=torch.bfloat16,
        page_size_padded=262144,
        dquant_key_container_bytes=155,
        dquant_value_container_bytes=109,
        dquant_recent_window=128,
        dquant_key_window=64,
    )
    mamba = MambaSpec(
        block_size=4096, shapes=((3, 16), (2, 8, 8)), dtypes=(torch.bfloat16, torch.float32), page_size_padded=262144
    )
    return {"attention": attention, "mamba": mamba}


def test_planner_budget_preserves_physical_block_limit(monkeypatch):
    import vllm.v1.core.kv_cache_utils as utils

    import vllm_ascend.envs as envs

    monkeypatch.setattr(envs, "VLLM_ASCEND_ENABLE_DQUANT", True)
    monkeypatch.setattr(envs, "VLLM_ASCEND_DQUANT_DENSE_REFERENCE", False)
    mapping = specs()
    groups = [KVCacheGroupSpec([name], spec) for name, spec in mapping.items()]
    monkeypatch.setattr(utils, "get_kv_cache_groups", lambda config, cache: groups)
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=2)), get_kv_cache_spec=lambda: mapping
    )
    physical = sum(spec.real_page_size_bytes for spec in mapping.values())
    planner = max(spec.page_size_bytes for spec in mapping.values())
    raw_reserve = 2 * mapping["attention"].recent_state_bytes
    budget = raw_reserve + 7 * physical + physical // 2
    converted = NPUWorker._scale_dquant_kv_budget(worker, budget)
    assert converted // planner == 7
    assert (converted // planner) * physical + raw_reserve <= budget
    with pytest.raises(ValueError, match="recent window reservation"):
        NPUWorker._scale_dquant_kv_budget(worker, raw_reserve)


@pytest.mark.parametrize("blocks", [1, 3, 7])
def test_compact_mamba_allocation_and_state_views(monkeypatch, blocks):
    import vllm_ascend.envs as envs

    monkeypatch.setattr(envs, "VLLM_ASCEND_ENABLE_DQUANT", True)
    monkeypatch.setattr(envs, "VLLM_ASCEND_DQUANT_DENSE_REFERENCE", False)
    mapping = specs()
    groups = [KVCacheGroupSpec([name], spec) for name, spec in mapping.items()]
    tensors = [
        KVCacheTensor(
            size=blocks * spec.page_size_bytes,
            layers=[name],
            layer_stride=blocks * spec.page_size_bytes,
            block_stride=spec.page_size_bytes,
            offset=0,
        )
        for name, spec in mapping.items()
    ]
    config = KVCacheConfig(num_blocks=blocks, kv_cache_tensors=tensors, kv_cache_groups=groups)
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.__dict__.update(
        device=torch.device("cpu"),
        runner_only_attn_layers=set(),
        use_compress=False,
        use_hybrid_blocks=False,
        vllm_config=SimpleNamespace(),
        _get_layer_kv_cache_specs=lambda cfg: mapping,
        _allocate_int8_cache_tensor=lambda count, alignment: torch.zeros(count, dtype=torch.int8),
        _kv_cache_spec_attn_group_iterator=lambda: [
            SimpleNamespace(backend=None, kv_cache_spec=group.kv_cache_spec, layer_names=group.layer_names)
            for group in groups
        ],
    )
    raw = NPUModelRunner._allocate_dquant_layer_tensors(runner, config)
    mamba = mapping["mamba"]
    assert raw["mamba"].numel() == blocks * mamba.real_page_size_bytes
    assert raw["mamba"].numel() < blocks * mamba.page_size_bytes
    views = NPUModelRunner._reshape_kv_cache_tensors(runner, config, raw)
    for index, (view, shape, dtype) in enumerate(zip(views["mamba"], mamba.shapes, mamba.dtypes)):
        assert view.shape == (blocks, *shape)
        assert view.dtype == dtype
        view.fill_(index + 1)
    for index, view in enumerate(views["mamba"]):
        assert torch.all(view == index + 1)
    assert views["attention"][0].shape == (blocks, 128, mapping["attention"].key_record_bytes)
    assert views["attention"][1].shape == (blocks, 128, mapping["attention"].value_record_bytes)
