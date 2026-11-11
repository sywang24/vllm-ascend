# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.attention.dquant_attention import DQuantAttentionCache, DQuantDecodeWorkspace
from vllm_ascend.core.kv_cache_interface import DQuantAttentionSpec


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("host_limit", [0, 24, 128])
@pytest.mark.parametrize("head_size", [128, 256])
def test_workspace_matches_legacy_kv_across_flush_and_recompute(dtype, host_limit, head_size):
    torch.npu.set_device(0)
    torch.manual_seed(71)
    spec = DQuantAttentionSpec(
        block_size=64, num_kv_heads=2, head_size=head_size, dtype=dtype, dquant_key_window=4, dquant_recent_window=8
    )
    storage = [
        torch.zeros(4, 64, width, device="npu", dtype=torch.uint8)
        for width in (spec.key_record_bytes, spec.value_record_bytes)
    ]
    cache = DQuantAttentionCache(storage, spec, output_chunk=64, decode_workspace=DQuantDecodeWorkspace())
    cache.mirror_cap = 16
    cache.host_mirror_cap = host_limit
    table = torch.tensor([3, 0, 2, 1], device="npu", dtype=torch.int32)
    keys = torch.randn(96, 2, head_size, device="npu", dtype=dtype)
    values = torch.randn_like(keys)
    positions = torch.arange(96, device="npu")
    slots = table[positions // 64].long() * 64 + positions % 64
    cache.write_sequence("a", slots[:28], keys[:28], values[:28], 28)
    cache.read(table, 28, dtype, "a")
    workspace_pointer = None
    for end in (29, 30, 33, 37, 44, 65):
        begin = cache.recent["a"].total
        if begin < end - 1:
            cache.write_sequence("a", slots[begin : end - 1], keys[begin : end - 1], values[begin : end - 1], end - 1)
        old = cache.read(table, end - 1, dtype, "a")
        expected = tuple(
            torch.cat((prefix, current)) for prefix, current in zip(old, (keys[end - 1 : end], values[end - 1 : end]))
        )
        with patch("torch.cat", side_effect=AssertionError("Overflow decode must not concatenate")):
            actual = cache._single_decode_kv(table, end - 1, dtype, "a", (keys[end - 1 : end], values[end - 1 : end]))
        assert actual is not None
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference, atol=0, rtol=0)
        if workspace_pointer is None:
            workspace_pointer = actual[0].data_ptr()
        assert actual[0].data_ptr() == workspace_pointer
        cache.write_sequence("a", slots[end - 1 : end], keys[end - 1 : end], values[end - 1 : end], end)
    cache.release(["a"])
    assert not cache.host_mirror and not cache.mirror
    cache.write_sequence("a", slots[:20], keys[:20], values[:20], 20)
    cache.read(table, 20, dtype, "a")
    # Below the mirror cap, the established incremental dense path still applies.
    actual = cache._single_decode_kv(table, 20, dtype, "a", (keys[20:21], values[20:21]))
    old = cache.read(table, 20, dtype, "a")
    for result, prefix, current in zip(actual, old, (keys[20:21], values[20:21])):
        torch.testing.assert_close(result, torch.cat((prefix, current)), atol=0, rtol=0)


def test_workspace_is_shared_across_layers_and_isolated_between_streams():
    torch.npu.set_device(0)
    workspace = DQuantDecodeWorkspace()
    device = torch.device("npu:0")
    first = workspace.get(64, 2, 128, torch.bfloat16, device)
    again = workspace.get(128, 2, 128, torch.bfloat16, device)
    assert first[0].data_ptr() == again[0].data_ptr()
    other_stream = torch.npu.Stream()
    with torch.npu.stream(other_stream):
        isolated = workspace.get(64, 2, 128, torch.bfloat16, device)
    assert isolated[0].data_ptr() != first[0].data_ptr()
    grown = workspace.get(513, 2, 128, torch.bfloat16, device)
    assert grown[0].shape[0] == 513
    assert grown[0].data_ptr() != first[0].data_ptr()
    assert len(workspace.buffers) == 2


def test_cold_or_disabled_workspace_falls_back_without_partial_updates():
    torch.npu.set_device(0)
    spec = DQuantAttentionSpec(
        block_size=64, num_kv_heads=2, head_size=128, dtype=torch.bfloat16, dquant_key_window=4, dquant_recent_window=8
    )
    storage = [
        torch.zeros(1, 64, width, device="npu", dtype=torch.uint8)
        for width in (spec.key_record_bytes, spec.value_record_bytes)
    ]
    cache = DQuantAttentionCache(storage, spec, decode_workspace=DQuantDecodeWorkspace())
    cache.mirror_cap = 16
    table = torch.tensor([0], device="npu", dtype=torch.int32)
    key = torch.randn(29, 2, 128, device="npu", dtype=spec.dtype)
    cache.write_sequence("a", torch.arange(28, device="npu"), key[:28], key[:28], 28)
    assert cache._single_decode_kv(table, 28, spec.dtype, "a", (key[28:], key[28:])) is None
    cache.read(table, 28, spec.dtype, "a")
    with pytest.raises(ValueError, match="Block table"):
        cache._single_decode_kv(table[:0], 28, spec.dtype, "a", (key[28:], key[28:]))
    cache.decode_workspace = None
    assert cache._single_decode_kv(table, 28, spec.dtype, "a", (key[28:], key[28:])) is None
    assert cache.recent["a"].total == 28
