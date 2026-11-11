from unittest.mock import patch

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.dquant_restore import restore_key_means_npu


@pytest.mark.parametrize("mean_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("output_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("heads,head_size,window", [(2, 128, 4), (3, 64, 8)])
def test_striped_mean_restore_matches_reference(mean_dtype, output_dtype, heads, head_size, window):
    torch.npu.set_device(0)
    torch.manual_seed(482)
    block_size = 6
    table = torch.tensor([7, 2, 5, 0, 6, 1, 4, 3], device="npu", dtype=torch.int32)
    capacity = table.numel() * block_size
    width = heads * head_size
    mean_bytes = torch.empty((), dtype=mean_dtype).element_size()
    stripe_bytes = 2 * ((width * mean_bytes + 2 * window - 1) // (2 * window))
    mean_offset = 18
    storage = torch.zeros(capacity, mean_offset + stripe_bytes, device="npu", dtype=torch.uint8)
    means = torch.randn(capacity // window, heads, head_size, device="npu", dtype=mean_dtype)
    stripes = torch.zeros(capacity // window, window * stripe_bytes, device="npu", dtype=torch.uint8)
    stripes[:, : width * mean_bytes] = means.view(torch.uint8).flatten(1)
    positions = torch.arange(capacity, device="npu")
    slots = table[positions // block_size].long() * block_size + positions % block_size
    storage[slots, mean_offset:] = stripes.view(capacity, stripe_bytes)
    before = storage.clone()
    start, length = 3, 31
    keys = torch.randn(length, heads, head_size, device="npu", dtype=output_dtype)
    original_keys = keys.clone()
    expected_means = means.repeat_interleave(window, 0)[start : start + length]
    expected = (keys.float() + expected_means.float()).to(output_dtype)
    with patch.object(torch, "arange", side_effect=AssertionError("dense slot indexing")):
        actual = restore_key_means_npu(
            keys,
            storage,
            table,
            start_token=start,
            block_size=block_size,
            window=window,
            mean_offset=mean_offset,
            stripe_bytes=stripe_bytes,
            mean_dtype=mean_dtype,
        )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(storage, before, atol=0, rtol=0)
    torch.testing.assert_close(keys, original_keys, atol=0, rtol=0)


def test_mean_restore_rejects_incomplete_layout():
    torch.npu.set_device(0)
    keys = torch.ones(4, 2, 128, device="npu", dtype=torch.bfloat16)
    storage = torch.zeros(8, 32, device="npu", dtype=torch.uint8)
    table = torch.tensor([0], device="npu", dtype=torch.int32)
    with pytest.raises(ValueError, match="cannot hold"):
        restore_key_means_npu(
            keys,
            storage,
            table,
            start_token=0,
            block_size=8,
            window=4,
            mean_offset=16,
            stripe_bytes=16,
            mean_dtype=torch.bfloat16,
        )


def test_mean_restore_handles_empty_and_requires_complete_mean_window():
    torch.npu.set_device(0)
    storage = torch.zeros(8, 144, device="npu", dtype=torch.uint8)
    table = torch.tensor([0], device="npu", dtype=torch.int32)
    keys = torch.ones(0, 2, 128, device="npu", dtype=torch.bfloat16)
    kwargs = dict(start_token=0, block_size=8, window=4, mean_offset=16, stripe_bytes=128, mean_dtype=torch.bfloat16)
    actual = restore_key_means_npu(keys, storage, table, **kwargs)
    assert actual.shape == keys.shape and actual.dtype == keys.dtype
    keys = torch.ones(1, 2, 128, device="npu", dtype=torch.bfloat16)
    kwargs.update(start_token=0, block_size=1)
    with pytest.raises(ValueError, match="cover"):
        restore_key_means_npu(keys, storage, table, **kwargs)
