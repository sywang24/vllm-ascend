import math
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_ascend.core.kv_cache_interface import DQuantAttentionSpec


def base_spec():
    return FullAttentionSpec(
        block_size=128,
        num_kv_heads=2,
        head_size=256,
        dtype=torch.bfloat16,
    )


def test_explicit_precision_reaches_scheduler_layout():
    overrides = {
        "VLLM_ASCEND_DQUANT_KEY_BYTES": 288,
        "VLLM_ASCEND_DQUANT_VALUE_BYTES": 288,
        "VLLM_ASCEND_DQUANT_KEY_LEVELS": 16,
        "VLLM_ASCEND_DQUANT_VALUE_LEVELS": 16,
        "VLLM_ASCEND_DQUANT_ALPHA": 1.0,
        "VLLM_ASCEND_DQUANT_RECENT_WINDOW": 128,
        "VLLM_ASCEND_DQUANT_KEY_WINDOW": 64,
    }
    with patch.multiple("vllm_ascend.envs", **overrides):
        spec = DQuantAttentionSpec.from_spec(base_spec())
    assert spec.stream_length == 512
    assert spec.dquant_key_levels == spec.dquant_value_levels == 16
    assert spec.dquant_alpha == 1.0
    assert spec.dquant_key_container_bytes == spec.dquant_value_container_bytes == 289
    assert spec.real_page_size_bytes == 128 * (294 + 16 + 294)
    assert DQuantAttentionSpec.merge([spec, spec]) == spec


def test_defaults_retain_original_tp2_format():
    overrides = {
        "VLLM_ASCEND_DQUANT_KEY_BYTES": 0,
        "VLLM_ASCEND_DQUANT_VALUE_BYTES": 0,
        "VLLM_ASCEND_DQUANT_KEY_LEVELS": 8,
        "VLLM_ASCEND_DQUANT_VALUE_LEVELS": 6,
        "VLLM_ASCEND_DQUANT_ALPHA": 1.4,
        "VLLM_ASCEND_DQUANT_RECENT_WINDOW": 0,
        "VLLM_ASCEND_DQUANT_KEY_WINDOW": 0,
    }
    with patch.multiple("vllm_ascend.envs", **overrides):
        spec = DQuantAttentionSpec.from_spec(base_spec())
    assert spec.dquant_key_container_bytes == 167
    assert spec.dquant_value_container_bytes == 117
    assert spec.real_page_size_bytes == 128 * (172 + 122)


@pytest.mark.parametrize(
    "field,value",
    [
        ("dquant_key_levels", 1),
        ("dquant_value_levels", 257),
        ("dquant_alpha", 0),
        ("dquant_alpha", -1),
        ("dquant_alpha", math.nan),
        ("dquant_alpha", math.inf),
        ("dquant_key_container_bytes", 3),
        ("dquant_value_container_bytes", -1),
    ],
)
def test_invalid_precision_fails_before_allocation(field, value):
    spec = DQuantAttentionSpec(
        block_size=128,
        num_kv_heads=2,
        head_size=256,
        dtype=torch.bfloat16,
    )
    with pytest.raises(ValueError):
        replace(spec, **{field: value})


def test_merge_rejects_different_probability_models():
    spec = DQuantAttentionSpec(
        block_size=128,
        num_kv_heads=2,
        head_size=256,
        dtype=torch.bfloat16,
    )
    with pytest.raises(AssertionError, match="share one record layout"):
        DQuantAttentionSpec.merge([spec, replace(spec, dquant_alpha=1.0)])


def test_negative_override_is_rejected():
    with (
        patch("vllm_ascend.envs.VLLM_ASCEND_DQUANT_KEY_BYTES", -1),
        pytest.raises(ValueError, match="nonnegative"),
    ):
        DQuantAttentionSpec.from_spec(base_spec())
