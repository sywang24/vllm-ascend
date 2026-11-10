# SPDX-License-Identifier: Apache-2.0
"""NPU integration regressions for NOVA page writes and request ownership."""

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.nova_kv import NovaAttentionImpl
from vllm_ascend.attention.nova_kv_v2 import NovaV2AttentionImpl
from vllm_ascend.ops.triton.nova_compressed_state import (
    ExactWindowPool,
    planar_cache_views,
    write_planar_packets,
)
from vllm_ascend.ops.triton.nova_direct_decode import DirectDecode
from vllm_ascend.quantization.methods.kv_cache.nova_kv.codec import NovaCodec
from vllm_ascend.worker.v2.nova_kv import NovaKVRuntime

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


def test_direct_decode_addresses_pages_above_four_gib(nova_kernel_root):
    torch.manual_seed(811)
    page, pitch = 128, 2**31 + 256
    # Only three page payloads are initialized; the unused gaps test GM offsets.
    backing = torch.empty(2 * pitch + page * 134, dtype=torch.uint8, device="npu")
    cache = torch.as_strided(backing, (3, page, 1, 134), (pitch, 134, 134, 1))
    keys, values, scales = planar_cache_views(cache)
    keys.fill_(0)
    values.fill_(0)
    scales[:, :, 0].fill_(0.1)
    scales[:, :, 1].fill_(0.05)
    scales[:, :, 2].fill_(0)
    keys[2].fill_(1)
    values[2].fill_(255)
    query = torch.randn(1, 1, 1, 256, device="npu")
    mean = torch.zeros(1, 1, 1, device="npu")
    books = torch.randn(1, 64, 256, 4, device="npu").half()
    table = torch.tensor([[2]], dtype=torch.int32, device="npu")
    starts = torch.tensor([0], dtype=torch.int32, device="npu")
    ends = torch.tensor([128], dtype=torch.int32, device="npu")
    kernel = DirectDecode(1, 1, 1, "npu", nova_kernel_root, 128, split_tokens=128)
    assert kernel.page_offset_bits == 64
    actual, _ = kernel(query, mean, keys, values, scales, books, table, starts, ends, 128, 1 / 16)
    expected = torch.ones_like(actual) * (3 * scales[2, 0, 1, 0].float())
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.003)


@pytest.mark.parametrize("groups", [1, 6])
@pytest.mark.parametrize("runner_version", ["v1", "v2"])
def test_decode_graph_replay_uses_updated_lengths_and_slots(nova_kernel_root, groups, runner_version):
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
    runtime = NovaKVRuntime(2, torch.device("npu")) if runner_version == "v2" else None
    slots = runtime.request_slots if runtime else torch.full((2,), -1, dtype=torch.int64, device="npu")
    token_slots = torch.full_like(slots, -1)
    key = torch.randn(2, heads, 256, device="npu", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    output = torch.empty_like(query)
    impl = object.__new__(NovaV2AttentionImpl if runner_version == "v2" else NovaAttentionImpl)
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
        if runtime:
            runtime.state.release(["request0", "request1"])
            history_lengths = [1, 1]
            for row, request in enumerate(requests):
                if request >= 0:
                    history_lengths[request] = seq_lens[row] - 1
            runtime.state.prepare(["request0", "request1"], [0, 0], history_lengths)
            live_rows = [row for row, request in enumerate(requests) if request >= 0]
            runtime.prepare(
                SimpleNamespace(
                    req_ids=[f"request{requests[row]}" for row in live_rows],
                    num_reqs=len(live_rows),
                    num_computed_tokens_np=np.array([seq_lens[row] - 1 for row in live_rows]),
                    num_scheduled_tokens=np.ones(len(live_rows), dtype=np.int32),
                    is_dummy=False,
                ),
                len(requests),
            )
        else:
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


@pytest.mark.parametrize("old_length,count", [(0, 17), (3, 1), (35, 19), (171, 139)])
def test_v2_prefill_matches_legacy_and_retains_exact_window(nova_kernel_root, old_length, count):
    torch.manual_seed(319)
    heads, groups, dim, max_tokens = 1, 2, 256, 512
    identity = torch.eye(dim, device="npu")[None]
    codec = NovaCodec(
        torch.randn(heads, dim, device="npu") * 0.1,
        identity,
        identity,
        identity,
        torch.randn(heads, 64, 256, 4, device="npu").half(),
    )
    total = old_length + count
    history_key = torch.randn(total, heads, dim, device="npu", dtype=torch.bfloat16)
    history_value = torch.randn_like(history_key)
    query = torch.randn(count, heads * groups, dim, device="npu", dtype=torch.bfloat16)
    table = torch.arange(4, dtype=torch.int32, device="npu")[None]
    lengths = torch.tensor([total], dtype=torch.int32, device="npu")
    slots = torch.tensor([1], dtype=torch.int64, device="npu")
    metadata = SimpleNamespace(
        seq_lens_list=[total],
        seq_lens_gpu=lengths,
        block_tables=table,
        nova_request_slots=slots,
        nova_request_slots_cpu=[1],
        slot_mapping=torch.arange(old_length, total, device="npu"),
        actual_seq_lengths_q=[count],
        num_actual_tokens=count,
        attn_state=None,
    )
    results, caches, windows = [], [], []
    for cls in (NovaAttentionImpl, NovaV2AttentionImpl):
        impl = object.__new__(cls)
        impl.head_size, impl.num_kv_heads, impl.num_heads = dim, heads, heads * groups
        impl.scale, impl._limit, impl._sink, impl._recent = 1 / 16, 2, 2, 2
        impl._prefill_query_tile, impl._prefill_kv_tile = 16, 128
        impl.vllm_config = SimpleNamespace(model_config=SimpleNamespace(max_model_len=max_tokens))
        impl._options = SimpleNamespace(codec_batch_size=16)
        impl.nova_codec = codec
        impl._codec = lambda layer, device: codec
        impl._ensure_runtime = lambda device, dtype: None
        impl._window = ExactWindowPool(2, heads, dim, 2, 2, "npu", torch.bfloat16)
        impl._cold = DirectDecode(2, heads, groups, "npu", nova_kernel_root, max_tokens, split_tokens=128)
        impl._causal_mask = torch.ones(2048, 2048, device="npu", dtype=torch.bool).triu(1)
        cache = torch.zeros(4, 128, heads, 134, dtype=torch.uint8, device="npu")
        views = planar_cache_views(cache)
        if old_length:
            write_planar_packets(
                views,
                torch.arange(old_length, device="npu"),
                *codec.encode(history_key[:old_length], history_value[:old_length], chunk_size=16),
            )
            impl._window.seed_chunk(history_key[:old_length], history_value[:old_length], 1, 0)
        output = torch.empty_like(query)
        impl.forward(
            SimpleNamespace(),
            query,
            history_key[old_length:],
            history_value[old_length:],
            cache,
            metadata,
            output=output,
        )
        results.append(output)
        caches.append(cache)
        windows.append((impl._window.key, impl._window.value))
    torch.npu.synchronize()
    torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)
    torch.testing.assert_close(caches[0], caches[1], rtol=0, atol=0)
    for before, after in zip(windows[0], windows[1]):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
