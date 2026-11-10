# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.ops import nova_kv as ops
from vllm_ascend.quantization.methods.kv_cache.nova_kv.codec import NovaCodec


def make_codec():
    torch.manual_seed(31)
    matrix = torch.randn(1, 256, 256) / 16
    return NovaCodec(
        torch.randn(1, 256),
        matrix,
        matrix.transpose(-1, -2).contiguous(),
        matrix,
        torch.randn(1, 64, 256, 4).half(),
    )


def test_query_lut_uses_inverse_transpose_and_original_mean():
    codec = make_codec()
    query = torch.randn(2, 1, 3, 256).bfloat16()
    direct = SimpleNamespace(
        prepare_lut=lambda q, books: torch.einsum("bhgrd,hrcd->bhgrc", q.reshape(2, 1, 3, 64, 4), books).half()
    )
    transformed, mean, lut = ops.nova_quant_query_lut(query, codec, direct)
    expected = query.float() @ codec.key_inverse[0].T
    torch.testing.assert_close(transformed, expected)
    torch.testing.assert_close(mean, (query.float() * codec.key_mean[:, None, :]).sum(-1))
    expected_lut = (expected.reshape(2, 1, 3, 64, 1, 4) * codec.codebooks.float()[None, :, None]).sum(-1).half()
    torch.testing.assert_close(lut, expected_lut)


def test_cache_interface_preserves_codec_packets_and_padding_slots(monkeypatch):
    codec = make_codec()
    key, value = torch.randn(3, 1, 256), torch.randn(3, 1, 256)
    expected = codec.encode(key, value, chunk_size=2)
    slots = torch.tensor([0, 129, -1], dtype=torch.int32)
    calls = []
    monkeypatch.setattr(ops, "write_planar_packets", lambda *args: calls.append(args))
    ops.nova_quant_cache(key, value, codec, (), slots, chunk_size=2)
    assert len(calls) == 1
    assert calls[0][1].dtype == torch.int64
    assert calls[0][1].tolist() == [0, 129, -1]
    for actual, reference in zip(calls[0][2:], expected):
        assert actual.is_contiguous()
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.parametrize("offset_bits", [32, 64])
def test_metadata_uses_context_bound_and_reports_workspace(offset_bits):
    views = (torch.empty(2, 1, 64, 128), None, torch.empty(2, 1, 3, 128))
    direct = SimpleNamespace(page_offset_bits=offset_bits, split_tokens=128, workspace_bytes=lambda: 8192)
    table = torch.empty(2, 4, dtype=torch.int32)
    plan = ops.nova_quant_flash_attention_metadata(direct, table, views, 400)
    assert plan == ops.NovaAttentionPlan(400, 128, 8192)
    assert ops.nova_quant_flash_attention_metadata(direct, table, views, 1000).max_tokens == 512


def test_large_cache_requires_widened_kernel_offsets():
    key = SimpleNamespace(shape=(65537, 1, 64, 128), stride=lambda dim: 65536)
    scale = SimpleNamespace(shape=(65537, 1, 3, 128), stride=lambda dim: 32768)
    direct = SimpleNamespace(page_offset_bits=32, split_tokens=128, workspace_bytes=lambda: 8192)
    table = torch.empty(1, 4, dtype=torch.int32)
    with pytest.raises(ValueError, match="64-bit page offsets"):
        ops.nova_quant_flash_attention_metadata(direct, table, (key, None, scale), 512)
    direct.page_offset_bits = 64
    assert ops.nova_quant_flash_attention_metadata(direct, table, (key, None, scale), 512).max_tokens == 512


def test_decode_passes_lut_and_disjoint_segment_boundaries():
    codec = make_codec()
    query = torch.randn(4, 1, 2, 256)
    lengths = torch.tensor([0, 2, 5, 20], dtype=torch.int32)
    slots = torch.tensor([-1, 0, 1, 2])
    lut = torch.zeros(4, 1, 2, 64, 256).half()
    cold = torch.randn_like(query)
    cold_lse = torch.randn(4, 1, 2)
    transformed_cold = torch.einsum("bhgd,hde->bhge", cold, codec.value_transform.transpose(-1, -2))
    calls = []

    def direct(*args, **kwargs):
        calls.append((args, kwargs))
        return cold, cold_lse

    def merge(q, request_slots, lens, scale, out, lse):
        assert q is query and request_slots is slots and lens is lengths
        torch.testing.assert_close(out, transformed_cold)
        return out, lse, None

    window = SimpleNamespace(sink=2, ring=3, attend_and_merge=merge)
    actual, lse = ops.nova_quant_flash_attention(
        query,
        codec,
        (None, None, None),
        None,
        window,
        1 / 16,
        transformed=query,
        mean=torch.zeros(4, 1, 2),
        lut=lut,
        request_slots=slots,
        lengths=lengths,
        direct_decode=direct,
        plan=ops.NovaAttentionPlan(512, 128, 8192),
    )
    torch.testing.assert_close(actual, transformed_cold)
    assert lse is cold_lse
    assert calls[0][1]["lut"] is lut
    assert calls[0][0][7].tolist() == [0, 2, 2, 2]
    assert calls[0][0][8].tolist() == [0, 2, 2, 17]


@pytest.mark.parametrize("old_length", [0, 15])
def test_prefill_dispatch_does_not_write_window(monkeypatch, old_length):
    query, key, value = torch.randn(3, 1, 2, 256), torch.randn(3, 1, 256), torch.randn(3, 1, 256)
    window = SimpleNamespace(key="key-window", value="value-window", sink=2, ring=3)
    expected = torch.randn_like(query)
    calls = []
    if old_length == 0:

        def fia(q, k, v, **kwargs):
            calls.append(kwargs)
            assert q.shape == (1, 2, 3, 256)
            assert k.shape == v.shape == (1, 1, 3, 256)
            return expected.permute(1, 2, 0, 3).reshape(1, 2, 3, 256), None

        monkeypatch.setattr(ops.torch_npu, "npu_fused_infer_attention_score", fia, raising=False)
    else:
        monkeypatch.setattr(ops, "planar_cold_reader", lambda *args: "cold")
        monkeypatch.setattr(ops, "exact_window_reader", lambda *args: "exact")

        def bounded(*args, **kwargs):
            calls.append((args, kwargs))
            assert args[3:] == (15, "cold", "exact", 2, 2)
            return expected, None

        monkeypatch.setattr(ops, "bounded_fia_prefill", bounded)
    output, _ = ops.nova_quant_flash_attention(
        query,
        None,
        (None, None, None),
        None,
        window,
        1 / 16,
        key=key,
        value=value,
        old_length=old_length,
        request_slot=0,
        causal_mask=torch.ones(3, 3, dtype=torch.bool).triu(1),
        query_tile=2,
        kv_tile=128,
    )
    torch.testing.assert_close(output, expected)
    assert len(calls) == 1
    if old_length == 0:
        assert calls[0]["sparse_mode"] == 3


def test_window_update_rejects_mixed_metadata():
    calls = []
    window = SimpleNamespace(write_rows=lambda *args: calls.append(args), seed_chunk=lambda *args: calls.append(args))
    key, value = torch.ones(1, 1, 256), torch.zeros(1, 1, 256)
    slots, positions = torch.tensor([0]), torch.tensor([7])
    ops.nova_exact_window_update(window, key, value, request_slots=slots, positions=positions)
    ops.nova_exact_window_update(window, key, value, request_slot=0, old_length=7)
    assert calls == [(key, value, slots, positions), (key, value, 0, 7)]
    with pytest.raises(ValueError, match="prefill window metadata"):
        ops.nova_exact_window_update(window, key, value, request_slot=0, old_length=7, positions=positions)
