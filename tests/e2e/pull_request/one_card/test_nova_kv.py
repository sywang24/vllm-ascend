# SPDX-License-Identifier: Apache-2.0
"""NPU integration regressions for NOVA page writes and request ownership."""

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.nova_kv import NovaAttentionImpl
from vllm_ascend.ops.triton.nova_compressed_state import (
    ExactWindowPool,
    planar_cache_views,
    write_planar_packets,
)
from vllm_ascend.ops.triton.nova_direct_decode import DirectDecode
from vllm_ascend.quantization.methods.kv_cache.nova_kv.codec import NovaCodec

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="NPU required")


@pytest.fixture(scope="module")
def nova_kernel_root(tmp_path_factory):
    if shutil.which("cmake") is None:
        pytest.skip("CMake required for the standalone NOVA kernel")
    soc = torch.npu.get_device_name(0)
    if not soc.startswith("Ascend910B"):
        pytest.skip("Standalone NOVA kernel currently targets Ascend 910B")
    source = Path(__file__).resolve().parents[4] / "csrc/attention/nova_kv"
    build = tmp_path_factory.mktemp("nova-kernel")
    subprocess.run(
        ["cmake", "-S", str(source), "-B", str(build), f"-DSOC_VERSION={soc}"],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    subprocess.run(["cmake", "--build", str(build), "-j2"], check=True, capture_output=True, text=True, timeout=300)
    return str(build)


@pytest.mark.parametrize("groups", [1, 6])
def test_direct_decode_matches_reconstructed_attention(nova_kernel_root, groups):
    torch.manual_seed(71)
    page, heads = 128, 1
    pitch = page * heads * 134 + 256
    raw = torch.zeros(4 * pitch, dtype=torch.uint8, device="npu")
    cache = torch.as_strided(raw, (4, page, heads, 134), (pitch, heads * 134, 134, 1))
    keys, values, scales = planar_cache_views(cache)
    keys.random_(0, 256)
    values.random_(0, 256)
    scales[:, :, 0].fill_(0.1)
    scales[:, :, 1].fill_(0.05)
    scales[:, :, 2].fill_(-0.075)
    books = torch.randn(heads, 64, 256, 4, device="npu").half()
    query = torch.randn(1, heads, groups, 256, device="npu")
    mean = torch.randn(1, heads, groups, device="npu")
    table = torch.tensor([[2, 0, 3, 1]], dtype=torch.int32, device="npu")
    kernel = DirectDecode(1, heads, groups, "npu", nova_kernel_root, 512, split_tokens=128)
    starts = torch.tensor([33], dtype=torch.int32, device="npu")
    ends = torch.tensor([297], dtype=torch.int32, device="npu")
    output, lse = kernel(query, mean, keys, values, scales, books, table, starts, ends, 512, 1 / 16)
    positions = torch.arange(33, 297, device="npu")
    physical_pages = table[0, positions // page].long()
    offsets = positions % page
    codes = keys[physical_pages, :, :, offsets].long()
    packed_values = values[physical_pages, :, :, offsets].long()
    parameters = scales[physical_pages, :, :, offsets].float()
    decoded_keys = (
        books[torch.arange(heads, device="npu")[None, :, None], torch.arange(64, device="npu")[None, None, :], codes]
        .reshape(-1, heads, 256)
        .float()
    )
    decoded_keys *= parameters[..., :1]
    symbols = torch.stack([(packed_values >> shift) & 3 for shift in (0, 2, 4, 6)], -1).flatten(-2)
    decoded_values = symbols.float() * parameters[..., 1:2] + parameters[..., 2:3]
    scores = (torch.einsum("bhgd,thd->bhgt", query, decoded_keys) + mean[..., None]) / 16
    expected = torch.einsum("bhgt,thd->bhgd", scores.softmax(-1), decoded_values)
    torch.testing.assert_close(output, expected, rtol=0.02, atol=0.003)
    torch.testing.assert_close(lse, scores.logsumexp(-1), rtol=0.003, atol=0.003)


def test_padded_page_scatter_ignores_graph_padding():
    page, heads, packet, pitch = 128, 2, 134, 65536
    backing = torch.full((2 * pitch + 64,), 123, dtype=torch.uint8, device="npu")
    cache = torch.as_strided(backing, (2, page, heads, packet), (pitch, heads * packet, packet, 1), storage_offset=64)
    views = planar_cache_views(cache)
    codes = torch.zeros((3, heads, 64), dtype=torch.uint8, device="npu")
    scales = torch.ones((3, heads, 3), dtype=torch.bfloat16, device="npu")
    slots = torch.tensor([0, page + 3, -1], dtype=torch.int64, device="npu")
    write_planar_packets(views, slots, codes, codes, scales)
    torch.npu.synchronize()
    actual = backing.cpu()
    assert (actual[:64] == 123).all()
    for physical_page in range(2):
        start = 64 + physical_page * pitch + page * heads * packet
        assert (actual[start : 64 + (physical_page + 1) * pitch] == 123).all()
    assert views[0][0, :, :, 0].cpu().eq(0).all()
    assert views[0][1, :, :, 3].cpu().eq(0).all()
    assert views[0][1, :, :, 0].cpu().eq(123).all()


@pytest.mark.parametrize("groups", [1, 6])
def test_decode_graph_replay_uses_updated_lengths_and_slots(nova_kernel_root, groups):
    torch.manual_seed(91)
    page, heads, max_tokens = 128, 1, 512
    cache = torch.zeros(8, page, heads, 134, dtype=torch.uint8, device="npu")
    views = planar_cache_views(cache)
    identity = torch.eye(256, device="npu")[None]
    codec = NovaCodec(
        torch.randn(heads, 256, device="npu") * 0.1,
        identity,
        identity,
        identity,
        torch.randn(heads, 64, 256, 4, device="npu").half(),
    )
    history_key = torch.randn(2, max_tokens, heads, 256, device="npu", dtype=torch.bfloat16)
    history_value = torch.randn_like(history_key)
    encoded = codec.encode(history_key.flatten(0, 1), history_value.flatten(0, 1))
    write_planar_packets(views, torch.arange(2 * max_tokens, device="npu"), *encoded)
    reconstructed = codec.decode(*encoded)
    query = torch.randn(2, heads * groups, 256, device="npu", dtype=torch.bfloat16)
    table = torch.zeros(2, 4, dtype=torch.int32, device="npu")
    lengths = torch.ones(2, dtype=torch.int32, device="npu")
    slots = torch.full((2,), -1, dtype=torch.int64, device="npu")
    token_slots = torch.full_like(slots, -1)
    key = torch.randn(2, heads, 256, device="npu", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    output = torch.empty_like(query)
    impl = object.__new__(NovaAttentionImpl)
    impl.head_size, impl.num_kv_heads, impl.num_heads = 256, heads, heads * groups
    impl.scale, impl._limit, impl._sink, impl._recent = 1 / 16, 2, 2, 2
    impl.vllm_config = SimpleNamespace(model_config=SimpleNamespace(max_model_len=max_tokens))
    impl._options = SimpleNamespace(codec_batch_size=256)
    impl._codec = lambda layer, device: codec
    impl._ensure_runtime = lambda device, dtype: None
    impl._window = ExactWindowPool(2, heads, 256, impl._sink, impl._recent, "npu", torch.bfloat16)
    impl._cold = DirectDecode(2, heads, groups, "npu", nova_kernel_root, max_tokens, split_tokens=128)
    metadata = SimpleNamespace(
        # Capture starts with a short dummy length and no owned windows.
        seq_lens_list=[1, 1],
        seq_lens_gpu=lengths,
        block_tables=table,
        nova_request_slots=slots,
        slot_mapping=token_slots,
        attn_state=AscendAttentionState.DecodeOnly,
    )

    def run():
        return impl.forward(SimpleNamespace(), query, key, value, cache, metadata, output=output)

    run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = run()
    for step, length in enumerate((33, 171, 389)):
        query.normal_()
        requests = [step % 2, -1] if step < 2 else [1, 0]
        seq_lens = [length, 1] if step < 2 else [length, 130]
        slots.copy_(torch.tensor(requests, dtype=torch.int64, device="npu"))
        lengths.copy_(torch.tensor(seq_lens, dtype=torch.int32, device="npu"))
        expected = torch.zeros_like(query)
        for row, request in enumerate(requests):
            if request < 0:
                # A stale positive page slot must still be masked for padding.
                token_slots[row].fill_(0)
                continue
            count = seq_lens[row]
            table[row].copy_(torch.arange(request * 4, request * 4 + 4, dtype=torch.int32, device="npu"))
            token_slots[row].fill_(request * max_tokens + count - 1)
            key[row].copy_(history_key[request, count - 1])
            value[row].copy_(history_value[request, count - 1])
            impl._window.seed_chunk(history_key[request, : count - 1], history_value[request, : count - 1], request, 0)
            reference_key, reference_value = [
                tensor[request * max_tokens : request * max_tokens + count].clone() for tensor in reconstructed
            ]
            for reference, exact in ((reference_key, history_key), (reference_value, history_value)):
                reference[: impl._sink] = exact[request, : impl._sink].float()
                reference[-(impl._recent + 1) :] = exact[request, count - impl._recent - 1 : count].float()
            scores = torch.einsum("hgd,thd->hgt", query[row].reshape(heads, groups, 256).float(), reference_key) / 16
            expected[row] = torch.einsum("hgt,thd->hgd", scores.softmax(-1), reference_value).reshape_as(expected[row])
        graph.replay()
        actual = captured.clone()
        eager = run().clone()
        torch.npu.synchronize()
        torch.testing.assert_close(actual, eager, rtol=0.01, atol=0.003)
        torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.003)
        if step < 2:
            assert torch.count_nonzero(actual[1]) == 0
