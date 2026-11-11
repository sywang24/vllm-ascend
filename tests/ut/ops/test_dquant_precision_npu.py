import torch
import torch_npu  # noqa: F401 - registers the NPU backend

from vllm_ascend.attention.dquant_attention import DQuantAttentionCache
from vllm_ascend.core.kv_cache_interface import DQuantAttentionSpec


def test_five_bit_paged_bridge_preserves_raw_tail_and_center_means():
    torch.npu.set_device(0)
    torch.manual_seed(20261023)
    spec = DQuantAttentionSpec(
        block_size=128,
        num_kv_heads=2,
        head_size=256,
        dtype=torch.bfloat16,
        dquant_key_container_bytes=353,
        dquant_value_container_bytes=353,
        dquant_key_levels=32,
        dquant_value_levels=32,
        dquant_alpha=1.0,
        dquant_recent_window=256,
        dquant_key_window=64,
    )
    key_storage = torch.zeros((4, 128, spec.key_record_bytes), dtype=torch.uint8, device="npu")
    value_storage = torch.zeros((4, 128, spec.value_record_bytes), dtype=torch.uint8, device="npu")
    cache = DQuantAttentionCache((key_storage, value_storage), spec, output_chunk=32)
    blocks = torch.tensor([2, 0, 3, 1], dtype=torch.int32, device="npu")
    positions = torch.arange(512, device="npu")
    slots = blocks[positions // 128].long() * 128 + positions % 128
    keys = torch.randn(512, 2, 256, device="npu", dtype=torch.bfloat16)
    keys += torch.randn(1, 2, 256, device="npu", dtype=torch.bfloat16) * 8
    values = torch.randn_like(keys)
    cache.write_sequence("diagnostic", slots[:384], keys[:384], values[:384], 384)
    decoded_keys, decoded_values = cache.read(blocks, 384, torch.bfloat16, "diagnostic")
    torch.testing.assert_close(decoded_keys[128:], keys[128:384], rtol=0, atol=0)
    torch.testing.assert_close(decoded_values[128:], values[128:384], rtol=0, atol=0)
    assert (decoded_keys[:128].float() - keys[:128].float()).square().mean().item() < 0.02
    assert (decoded_values[:128].float() - values[:128].float()).square().mean().item() < 0.02
    cache.write_sequence("diagnostic", slots[384:448], keys[384:448], values[384:448], 448)
    decoded_keys, decoded_values = cache.read(blocks, 448, torch.bfloat16, "diagnostic")
    torch.testing.assert_close(decoded_keys[192:], keys[192:448], rtol=0, atol=0)
    torch.testing.assert_close(decoded_values[192:], values[192:448], rtol=0, atol=0)
    assert cache.key.overflow_report() == 0
    assert cache.value.overflow_report() == 0
    cache.release(["diagnostic"])
    assert not cache.recent and not cache.mirror
