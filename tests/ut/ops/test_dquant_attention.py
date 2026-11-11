# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
import torch_npu
from vllm.triton_utils import tl, triton

import vllm_ascend.attention.dquant_attention as attention_module
import vllm_ascend.ops.triton.dquant_fused as dquant_fused
from vllm_ascend.attention.dquant_attention import (
    DQuantAttentionCache,
    merge_attention_partials,
)
from vllm_ascend.core.kv_cache_interface import DQuantAttentionSpec
from vllm_ascend.ops.triton.dquant_fused import (
    make_packed_decode_lut,
    paged_dquant_attention_npu,
)
from vllm_ascend.ops.triton.dquant_fused_window import _lookup_packed_lut

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="Ascend NPU is unavailable")


@triton.jit
def _check_packed_lut_gather(lut_ptr, output_ptr):
    offsets = tl.arange(0, 256)
    slots = (offsets * 73 + 41) % 256
    lut = tl.load(lut_ptr + offsets).to(tl.float32, bitcast=True)
    tl.store(output_ptr + offsets, _lookup_packed_lut(lut, slots))


@pytest.mark.parametrize("frequencies", [[256], [1, 2, 3, 4, 5, 6, 7, 228]])
def test_window_packed_lut_gather_preserves_all_bits(frequencies):
    lut = make_packed_decode_lut(frequencies, "npu")
    output = torch.empty_like(lut)
    _check_packed_lut_gather[(1,)](lut, output)
    slots = (torch.arange(256, device="npu") * 73 + 41) % 256
    assert torch.equal(output, lut[slots])


@pytest.mark.parametrize(
    "dtype,chunk,heads",
    [(torch.float16, 16, 4), (torch.bfloat16, 32, 4), (torch.bfloat16, 64, 4), (torch.float16, 64, 6)],
)
def test_window_specialization_matches_general_fused_kernel(dtype, chunk, heads, monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_DQUANT_SYMBOL_CACHE", "0")
    monkeypatch.setenv("VLLM_ASCEND_DQUANT_FUSED_ATTENTION", "1")
    spec = DQuantAttentionSpec(
        block_size=128,
        num_kv_heads=2,
        head_size=256,
        dtype=dtype,
        dquant_key_levels=8,
        dquant_value_levels=6,
        dquant_alpha=1.4,
        dquant_key_container_bytes=155,
        dquant_value_container_bytes=109,
        dquant_recent_window=128,
        dquant_key_window=64,
    )
    storage = [
        torch.zeros((4, 128, width), dtype=torch.uint8, device="npu")
        for width in (spec.key_record_bytes, spec.value_record_bytes)
    ]
    table = torch.tensor([3, 1, 0, 2], dtype=torch.int32, device="npu")
    positions = torch.arange(510, device="npu")
    slots = table[positions // 128].long() * 128 + positions % 128
    torch.manual_seed(2973)
    key = torch.randn((510, 2, 256), dtype=dtype, device="npu")
    value = torch.randn_like(key)
    cache = DQuantAttentionCache(storage, spec)
    cache.write_sequence("window", slots, key, value, 510)
    compressed = cache.recent["window"].compressed
    assert compressed >= 3 * spec.dquant_key_window
    query = torch.randn((1, heads, 256), dtype=dtype, device="npu")
    actual = cache._rans_attention_partials(query, table, compressed, 256**-0.5, chunk, chunk)
    monkeypatch.setattr(
        dquant_fused, "_paged_dquant_window_attention_kernel", dquant_fused._paged_dquant_fused_attention_kernel
    )
    expected = cache._rans_attention_partials(query, table, compressed, 256**-0.5, chunk, chunk)
    for result, reference_value in zip(actual, expected):
        torch.testing.assert_close(result, reference_value, atol=3e-4, rtol=2e-3)


def make_cache(dtype=torch.float16, recent=0, key_window=0):
    # Qwen target's actual full-attention layout: four KV heads of size 256.
    spec = DQuantAttentionSpec(
        block_size=4,
        num_kv_heads=4,
        head_size=256,
        dtype=dtype,
        dquant_recent_window=recent,
        dquant_key_window=key_window,
    )
    storage = [
        torch.zeros((6, spec.block_size, width), dtype=torch.uint8, device="npu")
        for width in (spec.key_record_bytes, spec.value_record_bytes)
    ]
    return DQuantAttentionCache(storage, spec), storage


def reference(query, key, value, scale):
    q_len, _, _ = query.shape
    kv_len = key.shape[0]
    mask = torch.arange(kv_len)[None, :] <= torch.arange(kv_len - q_len, kv_len)[:, None]
    return F.scaled_dot_product_attention(
        query.float().cpu().transpose(0, 1),
        key.float().cpu().transpose(0, 1).repeat_interleave(query.shape[1] // key.shape[1], 0),
        value.float().cpu().transpose(0, 1).repeat_interleave(query.shape[1] // value.shape[1], 0),
        attn_mask=mask,
        scale=scale,
    ).transpose(0, 1)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_chunked_prefill_decode_and_mixed_batch_match_decoded_reference(dtype):
    cache, storage = make_cache(dtype)
    torch.manual_seed(47)
    key = torch.randn((8, 4, 256), dtype=dtype, device="npu")
    value = torch.randn_like(key)
    # Request A has non-contiguous physical pages [2, 4]; B starts at page 1.
    tables = torch.tensor([[2, 4], [1, 3]], dtype=torch.int32, device="npu")
    slots = torch.tensor([8, 9, 10, 11, 16, 17, 4, 5], device="npu")
    cache.write(slots[:3], key[:3], value[:3])
    mask = torch.triu(torch.ones((2048, 2048), dtype=torch.bool, device="npu"), diagonal=1)
    scale = 256**-0.5

    def check(q_ends, seq_lens):
        query = torch.randn((q_ends[-1], 8, 256), dtype=dtype, device="npu")
        output = torch.empty_like(query)
        cache.attention(query, tables, q_ends, seq_lens, scale, output, mask)
        start = 0
        for i, (end, seq_len) in enumerate(zip(q_ends, seq_lens)):
            k, v = cache.read(tables[i], seq_len, dtype)
            expected = reference(query[start:end], k, v, scale)
            torch.testing.assert_close(output[start:end].float().cpu(), expected, atol=0.025, rtol=0.025)
            start = end
        assert torch.isfinite(output).all().item()

    check([3], [3])  # initial prefill
    cache.write(slots[3:5], key[3:5], value[3:5])
    check([2], [5])  # chunk attends to cached prefix
    cache.write(slots[5:], key[5:], value[5:])
    check([1, 3], [6, 2])  # mixed decode and prefill
    assert sum(x.numel() for x in storage) == 24 * (336 + 236)
    assert cache.key.storage_bytes + cache.value.storage_bytes == 24 * 572


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_decode_matches_dense_decode_without_kv_materialization(dtype, monkeypatch):
    spec = DQuantAttentionSpec(
        block_size=4,
        num_kv_heads=2,
        head_size=256,
        dtype=dtype,
        dquant_recent_window=3,
        dquant_key_window=4,
    )
    storage = [
        torch.zeros((6, 4, width), dtype=torch.uint8, device="npu")
        for width in (spec.key_record_bytes, spec.value_record_bytes)
    ]
    cache = DQuantAttentionCache(storage, spec)
    cache.set_decode_chunk_size(4)
    torch.manual_seed(20261028)
    table = torch.tensor([2, 0, 1, 3, 4, 5], dtype=torch.int32, device="npu")
    key = torch.randn((20, 2, 256), dtype=dtype, device="npu")
    value = torch.randn_like(key)
    positions = torch.arange(22, device="npu")
    slots = table[positions // 4].long() * 4 + positions % 4
    cache.write_sequence("a", slots[:20], key, value, 20)
    assert cache.recent["a"].compressed == 16
    query = torch.randn((1, 4, 256), dtype=dtype, device="npu")
    current = (
        torch.randn((1, 2, 256), dtype=dtype, device="npu"),
        torch.randn((1, 2, 256), dtype=dtype, device="npu"),
    )
    expected = cache._stream_decode_attention(query, table, 20, 256**-0.5, "a", current)
    monkeypatch.setenv("VLLM_ASCEND_DQUANT_FUSED_ATTENTION", "1")
    monkeypatch.setenv("VLLM_ASCEND_DQUANT_FUSED_CHUNK_TOKENS", "16")
    fused = DQuantAttentionCache(storage, spec)
    fused.set_decode_chunk_size(4)
    fused.recent.update(cache.recent)
    fused.recent_pool = cache.recent_pool
    assert fused.fused_decode and fused.mirror_cap == 0 and fused.host_mirror_cap == 0

    def forbid_dense_read(*args, **kwargs):
        raise AssertionError("Fused decode must not reconstruct a dense compressed prefix")

    monkeypatch.setattr(fused, "read", forbid_dense_read)
    monkeypatch.setattr(fused, "read_compressed_range", forbid_dense_read)
    output = torch.empty_like(query)
    fused.attention(
        query,
        table[None],
        [1],
        [21],
        256**-0.5,
        output,
        None,
        request_ids=["a"],
        updates=(slots[20:21], *current),
    )
    assert not fused.mirror and fused.host_mirror_storage_bytes == 0
    torch.testing.assert_close(output.float(), expected.float(), atol=0.02, rtol=0.02)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_parallel_decode_matches_independent_requests(dtype, monkeypatch):
    spec = DQuantAttentionSpec(
        block_size=4,
        num_kv_heads=2,
        head_size=256,
        dtype=dtype,
        dquant_recent_window=3,
        dquant_key_window=4,
    )
    storage = [
        torch.zeros((12, 4, width), dtype=torch.uint8, device="npu")
        for width in (spec.key_record_bytes, spec.value_record_bytes)
    ]
    monkeypatch.setenv("VLLM_ASCEND_DQUANT_FUSED_ATTENTION", "1")
    monkeypatch.setenv("VLLM_ASCEND_DQUANT_FUSED_CHUNK_TOKENS", "16")
    torch.manual_seed(2026102801)
    table = torch.tensor([2, 0, 1, 3, 4, 5], dtype=torch.int32, device="npu")
    second_table = torch.tensor([6, 7, 8, 9, 10], dtype=torch.int32, device="npu")
    positions = torch.arange(22, device="npu")
    slots = table[positions // 4].long() * 4 + positions % 4
    second_slots = second_table[positions[:20] // 4].long() * 4 + positions[:20] % 4
    current_slots = torch.tensor([20, 44], dtype=torch.int64, device="npu")
    key = torch.randn((20, 2, 256), dtype=dtype, device="npu")
    value = torch.randn_like(key)
    reference_cache = DQuantAttentionCache(storage, spec)
    reference_cache.set_decode_chunk_size(4)
    reference_cache.write_sequence("a", slots[:20], key, value, 20)
    reference_cache.write_sequence("b", second_slots, key.flip(0), value.flip(0), 20)
    query = torch.randn((2, 4, 256), dtype=dtype, device="npu")
    current_key = torch.randn((2, 2, 256), dtype=dtype, device="npu")
    current_value = torch.randn_like(current_key)
    request_metadata = (("a", table), ("b", second_table))
    expected = [
        reference_cache._stream_decode_attention(
            query[index : index + 1],
            request_table,
            20,
            256**-0.5,
            request,
            (current_key[index : index + 1], current_value[index : index + 1]),
        )
        for index, (request, request_table) in enumerate(request_metadata)
    ]
    fused = DQuantAttentionCache(storage, spec)
    fused.recent.update(reference_cache.recent)
    fused.recent_pool = reference_cache.recent_pool
    output = torch.empty_like(query)
    batch_calls = []
    original_batch = attention_module.paged_dquant_batch_attention_npu

    def track_batch(*args, **kwargs):
        batch_calls.append(1)
        return original_batch(*args, **kwargs)

    monkeypatch.setattr(attention_module, "paged_dquant_batch_attention_npu", track_batch)
    fused.attention(
        query,
        torch.stack(
            (
                table,
                torch.cat((second_table, torch.tensor([11], device="npu", dtype=torch.int32))),
            )
        ),
        [1, 2],
        [21, 21],
        256**-0.5,
        output,
        None,
        request_ids=["a", "b"],
        updates=(current_slots, current_key, current_value),
    )
    assert len(batch_calls) == 1
    torch.testing.assert_close(output.float(), torch.cat(expected).float(), atol=0.02, rtol=0.02)
    assert not fused.mirror and fused.host_mirror_storage_bytes == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_batched_window_specialization_matches_single_request_decode(monkeypatch, dtype):
    spec = DQuantAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=16,
        dtype=dtype,
        dquant_recent_window=15,
        dquant_key_window=16,
    )
    storage = [
        torch.zeros((16, 16, width), dtype=torch.uint8, device="npu")
        for width in (spec.key_record_bytes, spec.value_record_bytes)
    ]
    monkeypatch.setenv("VLLM_ASCEND_DQUANT_FUSED_ATTENTION", "1")
    monkeypatch.setenv("VLLM_ASCEND_DQUANT_FUSED_CHUNK_TOKENS", "16")
    monkeypatch.setenv("VLLM_ASCEND_DQUANT_FUSED_TILE_TOKENS", "16")
    torch.manual_seed(2026110201)
    tables = torch.tensor(
        [[0, 1, 2, 3, 4], [8, 9, 10, 11, 12]],
        dtype=torch.int32,
        device="npu",
    )
    lengths = [48, 64]
    cache = DQuantAttentionCache(storage, spec)
    for index, length in enumerate(lengths):
        positions = torch.arange(length, device="npu")
        slots = tables[index, positions // spec.block_size].long() * spec.block_size
        slots += positions % spec.block_size
        key = torch.randn((length, 2, 16), dtype=dtype, device="npu")
        value = torch.randn_like(key)
        cache.write_sequence(str(index), slots, key, value, length)

    queries = torch.randn((2, 4, 16), dtype=dtype, device="npu")
    current_key = torch.randn((2, 2, 16), dtype=dtype, device="npu")
    current_value = torch.randn_like(current_key)
    current_slots = torch.tensor([48, 192], dtype=torch.int64, device="npu")
    request_ids = ["0", "1"]
    expected = []
    for index, request_id in enumerate(request_ids):
        result = cache._fused_decode_attention(
            queries[index : index + 1],
            tables[index],
            lengths[index],
            16**-0.5,
            request_id,
            (current_key[index : index + 1], current_value[index : index + 1]),
        )
        assert result is not None
        expected.append(result)

    monkeypatch.setattr(
        cache,
        "_fused_decode_attention",
        lambda *args, **kwargs: pytest.fail("batched attention fell back to sequential decode"),
    )
    output = torch.empty_like(queries)
    cache.attention(
        queries,
        tables,
        [1, 2],
        [length + 1 for length in lengths],
        16**-0.5,
        output,
        None,
        request_ids=request_ids,
        updates=(current_slots, current_key, current_value),
    )
    torch.testing.assert_close(output.float(), torch.cat(expected).float(), atol=0.02, rtol=0.02)


def test_hierarchical_attention_merge_matches_sequential_merge():
    torch.manual_seed(2026102802)
    count, heads, width = 257, 4, 16
    parts = torch.randn((count, heads, width), dtype=torch.float32, device="npu")
    lses = torch.randn((count, heads), dtype=torch.float32, device="npu")
    expected_out, expected_lse = parts[0], lses[0]
    for index in range(1, count):
        expected_out, expected_lse = torch_npu.npu_attention_update(
            [expected_lse, lses[index]], [expected_out, parts[index]], 1
        )
    actual_out, actual_lse = merge_attention_partials(parts, lses)
    torch.testing.assert_close(actual_out, expected_out, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(actual_lse, expected_lse, atol=1e-4, rtol=1e-4)


def test_packed_rans_table_roundtrip():
    frequencies = (1, 126, 129)
    packed = make_packed_decode_lut(frequencies, "npu").cpu()
    begin = 0
    for symbol, frequency in enumerate(frequencies):
        entries = packed[begin : begin + frequency]
        assert torch.all((entries & 255) == symbol)
        assert torch.all(((entries >> 8) & 511) == frequency)
        assert torch.all(((entries >> 17) & 255) == begin)
        begin += frequency
    single = make_packed_decode_lut((256,), "npu").cpu()
    assert torch.all(((single >> 8) & 511) == 256)
    for invalid in ((), (1, 2), (0, 256), (-1, 257)):
        with pytest.raises(ValueError):
            make_packed_decode_lut(invalid, "npu")


def test_attention_merge_rejects_empty_or_mismatched_partials():
    with pytest.raises(ValueError, match="nonempty"):
        merge_attention_partials(torch.empty(0, 4, 16), torch.empty(0, 4))
    with pytest.raises(ValueError, match="matching LSE"):
        merge_attention_partials(torch.empty(2, 4, 16), torch.empty(2, 3))


def test_fused_rejects_invalid_layout_before_kernel_launch(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_DQUANT_FUSED_ATTENTION", "1")
    spec = DQuantAttentionSpec(
        block_size=128,
        num_kv_heads=2,
        head_size=256,
        dtype=torch.bfloat16,
        dquant_recent_window=128,
        dquant_key_window=64,
        dquant_key_container_bytes=155,
        dquant_value_container_bytes=109,
    )
    storage = [
        torch.zeros((2, 128, width), dtype=torch.uint8, device="npu")
        for width in (spec.key_record_bytes, spec.value_record_bytes)
    ]
    cache = DQuantAttentionCache(storage, spec)
    query = torch.zeros((1, 12, 256), dtype=spec.dtype, device="npu")
    table = torch.arange(2, dtype=torch.int32, device="npu")
    args = dict(
        query=query,
        block_table=table,
        compressed=128,
        scale=256**-0.5,
        key_storage=cache.key.storage,
        key_scales=cache.key.scales,
        key_offsets=cache.key.offsets,
        key_decode_lut=cache.fused_key_lut,
        value_storage=cache.value.storage,
        value_scales=cache.value.scales,
        value_offsets=cache.value.offsets,
        value_decode_lut=cache.fused_value_lut,
        block_size=128,
        head_size=256,
        num_kv_heads=2,
        key_container_bytes=155,
        value_container_bytes=109,
        key_window=64,
        mean_offset=cache.key.record_bytes,
        stripe_bytes=16,
    )
    invalid_arguments = (
        dict(tile_tokens=0),
        dict(tile_tokens=24),
        dict(chunk_tokens=0),
        dict(chunk_tokens=17),
        dict(compressed=127),
        dict(block_table=table[:0]),
        dict(key_decode_lut=cache.fused_key_lut[:128]),
        dict(stripe_bytes=2),
        dict(num_kv_heads=4),
        dict(query=query.expand(2, -1, -1)),
    )
    for invalid in invalid_arguments:
        with pytest.raises(ValueError):
            paged_dquant_attention_npu(**(args | invalid))


def test_padding_does_not_overwrite_last_slot_and_page_copy_survives_rebinding():
    cache, storage = make_cache()
    key = torch.randn((2, 4, 256), dtype=torch.float16, device="npu")
    value = torch.randn_like(key)
    cache.write(torch.tensor([0, -1], device="npu"), key, value)
    assert storage[0][-1, -1].count_nonzero().item() == 0
    assert storage[1][-1, -1].count_nonzero().item() == 0
    before = cache.read(torch.tensor([0], device="npu"), 1, torch.float16)
    for records in storage:
        records[3].copy_(records[0])
    rebound = DQuantAttentionCache(storage, cache.spec)
    after = rebound.read(torch.tensor([3], device="npu"), 1, torch.float16)
    for actual, expected in zip(after, before):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    cache.write(torch.tensor([0], device="npu"), key[1:], value[1:])
    overwritten = cache.read(torch.tensor([0], device="npu"), 1, torch.float16)
    assert not torch.equal(overwritten[0], before[0])


def test_rejects_invalid_window_and_incomplete_table():
    cache, storage = make_cache()
    with pytest.raises(ValueError, match="nonnegative"):
        replace(cache.spec, dquant_recent_window=-1)
    with pytest.raises(ValueError, match="divide"):
        replace(cache.spec, dquant_key_window=3)
    with pytest.raises(ValueError, match="cover"):
        cache.read(torch.tensor([0], device="npu"), 5, torch.float16)
    with pytest.raises(ValueError, match="shape"):
        cache.write(torch.tensor([0], device="npu"), torch.zeros((1, 2, 32)), torch.zeros((1, 2, 32)))


def test_backend_routes_to_compressed_cache(monkeypatch):
    import vllm_ascend.attention.attention_v1 as attention_module
    from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl

    monkeypatch.setattr(attention_module, "_EXTRA_CTX", SimpleNamespace(capturing=False))
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_DQUANT", "1")
    monkeypatch.setenv("VLLM_ASCEND_DQUANT_DECODE_CHUNK_SIZE", "8")
    _, storage = make_cache()
    impl = AscendAttentionBackendImpl.__new__(AscendAttentionBackendImpl)
    impl._dquant_cache = None
    impl._use_layer_aware_fia_graph_replay = False
    impl.num_kv_heads = 4
    impl.head_size = 256
    impl.scale = 256**-0.5
    query = torch.randn((2, 8, 256), dtype=torch.float16, device="npu")
    key = torch.randn((2, 4, 256), dtype=torch.float16, device="npu")
    value = torch.randn_like(key)
    output = torch.full_like(query, float("nan"))
    metadata = SimpleNamespace(
        causal=True,
        num_actual_tokens=2,
        slot_mapping=torch.tensor([8, 9], device="npu"),
        actual_seq_lengths_q=[2],
        seq_lens_list=[2],
        block_tables=torch.tensor([[2]], dtype=torch.int32, device="npu"),
        attn_mask=torch.triu(torch.ones((2048, 2048), dtype=torch.bool, device="npu"), diagonal=1),
    )
    layer = SimpleNamespace(_k_scale_float=1.0, _v_scale_float=1.0)
    result = impl.forward(layer, query, key, value, storage, metadata, output=output)
    assert result is output
    assert torch.isfinite(result).all().item()
    assert impl._dquant_cache is not None
    assert impl._dquant_cache.decode_chunk_size == 8


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("recent,window", [(3, 0), (0, 4), (3, 4)])
def test_recent_flush_centering_page_copy_and_slot_reuse(dtype, recent, window):
    cache, storage = make_cache(dtype, recent, window)
    torch.manual_seed(51)
    mean = torch.randn((1, 4, 256), device="npu", dtype=dtype) * 4
    key = torch.randn((12, 4, 256), device="npu", dtype=dtype) * 0.1 + mean
    value = torch.randn_like(key)
    table = torch.tensor([2, 4, 1], device="npu")
    slots = torch.tensor([8, 9, 10, 11, 16, 17, 18, 19, 4, 5, 6, 7], device="npu")
    for start, end in [(0, 2), (2, 7), (7, 8), (8, 12)]:
        cache.write_sequence("a", slots[start:end], key[start:end], value[start:end], end)
        actual_k, actual_v = cache.read(table, end, dtype, "a")
        state = cache.recent["a"]
        assert state.length <= cache.spec.recent_capacity
        assert state.length == end - state.compressed
        if state.length:
            torch.testing.assert_close(actual_k[-state.length :], key[end - state.length : end], atol=0, rtol=0)
            torch.testing.assert_close(actual_v[-state.length :], value[end - state.length : end], atol=0, rtol=0)
        if window and state.compressed:
            assert (
                actual_k[: state.compressed].float() - key[: state.compressed].float()
            ).square().mean().item() < 0.03
    if window:
        for records in storage:
            records[3].copy_(records[2])
        rebound = DQuantAttentionCache(storage, cache.spec)
        copied = rebound.read(torch.tensor([3], device="npu"), 4, dtype)
        original = cache.read(table, 12, dtype, "a")
        torch.testing.assert_close(copied[0], original[0][:4], atol=0, rtol=0)
        torch.testing.assert_close(copied[1], original[1][:4], atol=0, rtol=0)
    cache.release(["a"])
    assert not cache.recent and not cache.recent_pool.owners
    assert cache.recent_storage_bytes == cache.recent_pool.capacity * cache.spec.recent_state_bytes
    cache.write_sequence("b", slots[:2], key[:2] + 1, value[:2], 2)
    read_k, _ = cache.read(table, 2, dtype, "b")
    torch.testing.assert_close(read_k, key[:2] + 1, atol=0, rtol=0)
    with pytest.raises(ValueError, match="prefix"):
        cache.write_sequence("b", slots[2:3], key[2:3], value[2:3], 4)


def test_windowed_attention_uses_current_chunk_raw_and_checks_causality():
    cache, _ = make_cache(recent=2, key_window=4)
    table = torch.tensor([[2, 4]], dtype=torch.int32, device="npu")
    slots = torch.tensor([8, 9, 10, 11, 16, 17], device="npu")
    key = torch.randn((6, 4, 256), dtype=torch.float16, device="npu")
    value = torch.randn_like(key)
    query = torch.randn((6, 8, 256), dtype=torch.float16, device="npu")
    mask = torch.triu(torch.ones((2048, 2048), dtype=torch.bool, device="npu"), diagonal=1)
    output = torch.empty_like(query)
    cache.attention(query, table, [6], [6], 256**-0.5, output, mask, request_ids=["a"], updates=(slots, key, value))
    expected = reference(query, key, value, 256**-0.5)
    torch.testing.assert_close(output.float().cpu(), expected, atol=0.025, rtol=0.025)
    assert cache.recent["a"].compressed == 4


def test_recent_request_isolation_and_recomputation():
    cache, _ = make_cache(recent=3, key_window=4)
    key = torch.randn((8, 4, 256), dtype=torch.float16, device="npu")
    value = torch.randn_like(key)
    slots_a = torch.tensor([8, 9, 10, 11, 16, 17, 18, 19], device="npu")
    slots_b = torch.tensor([4, 5], device="npu")
    cache.write_sequence("a", slots_a[:5], key[:5], value[:5], 5)
    cache.write_sequence("b", slots_b, key[:2] + 2, value[:2], 2)
    cache.write_sequence("a", slots_a[5:], key[5:], value[5:], 8)
    actual, _ = cache.read(torch.tensor([1], device="npu"), 2, torch.float16, "b")
    torch.testing.assert_close(actual, key[:2] + 2, atol=0, rtol=0)
    assert cache.recent["a"].compressed == 4
    cache.write_sequence("a", slots_a[:2], key[:2] - 1, value[:2], 2)
    actual, _ = cache.read(torch.tensor([2], device="npu"), 2, torch.float16, "a")
    torch.testing.assert_close(actual, key[:2] - 1, atol=0, rtol=0)
    cache.release(["a", "b"])
    with pytest.raises(ValueError, match="missing"):
        cache.write_sequence("a", slots_a[2:3], key[2:3], value[2:3], 3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("recent,window", [(0, 0), (3, 0), (3, 4)])
def test_paged_decode_matches_reference_across_noncontiguous_pages(dtype, recent, window):
    cache, storage = make_cache(dtype, recent, window)
    legacy = DQuantAttentionCache(storage, cache.spec, paged_decode=False)
    torch.manual_seed(98)
    key = torch.randn((12, 4, 256), dtype=dtype, device="npu")
    value = torch.randn_like(key)
    table = torch.tensor([2, 4, 1], dtype=torch.int32, device="npu")
    slots = torch.tensor([8, 9, 10, 11, 16, 17, 18, 19, 4, 5, 6, 7], device="npu")
    for start, end in [(0, 2), (2, 7), (7, 8), (8, 12)]:
        cache.write_sequence("a", slots[start:end], key[start:end], value[start:end], end)
        legacy.recent = cache.recent
        request = "a" if cache.spec.uses_window_state else None
        expected = legacy.read(table, end, dtype, request)
        actual = cache.read(table, end, dtype, request)
        for result, reference_value in zip(actual, expected):
            torch.testing.assert_close(result, reference_value, atol=0, rtol=0)
    copied_table = torch.tensor([3, 4, 1], dtype=torch.int32, device="npu")
    for pages in storage:
        pages[3].copy_(pages[2])
    actual = cache.read(copied_table, 12, dtype, request)
    expected = legacy.read(table, 12, dtype, request)
    for result, reference_value in zip(actual, expected):
        torch.testing.assert_close(result, reference_value, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_matched_bf16_control_preserves_raw_cache_and_causal_chunks(dtype):
    from dquant_eval_worker import dense_control_forward

    torch.manual_seed(84)
    key = torch.randn((6, 4, 256), device="npu", dtype=dtype)
    value = torch.randn_like(key)
    query = torch.randn((6, 8, 256), device="npu", dtype=dtype)
    pages = tuple(torch.zeros((5, 4, 4, 256), device="npu", dtype=dtype) for _ in range(2))
    slots = torch.tensor([8, 9, 10, 11, 16, 17], device="npu")
    table = torch.tensor([[2, 4]], device="npu", dtype=torch.int32)
    mask = torch.triu(torch.ones((2048, 2048), device="npu", dtype=torch.bool), diagonal=1)
    impl = SimpleNamespace(num_kv_heads=4, scale=256**-0.5)
    for start, end in [(0, 3), (3, 5), (5, 6)]:
        metadata = SimpleNamespace(
            num_actual_tokens=end - start,
            causal=True,
            slot_mapping=slots[start:end],
            actual_seq_lengths_q=[end - start],
            seq_lens_list=[end],
            block_tables=table,
            attn_mask=mask,
        )
        output = torch.empty_like(query[start:end])
        dense_control_forward(
            impl,
            None,
            query[start:end],
            key[start:end],
            value[start:end],
            pages,
            metadata,
            output,
        )
        expected = reference(query[start:end], key[:end], value[:end], impl.scale)
        torch.testing.assert_close(output.float().cpu(), expected, atol=0.025, rtol=0.025)
        for stored, original in zip(pages, (key, value)):
            torch.testing.assert_close(stored.flatten(0, 1)[slots[:end]], original[:end], atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("recent,window", [(0, 0), (3, 0), (3, 4)])
def test_streaming_decode_matches_full_attention_and_range_reads(dtype, recent, window):
    cache, storage = make_cache(dtype, recent, window)
    stream = DQuantAttentionCache(storage, cache.spec, decode_chunk_size=4)
    torch.manual_seed(113)
    key = torch.randn((12, 4, 256), device="npu", dtype=dtype)
    value = torch.randn_like(key)
    query = torch.randn((1, 8, 256), device="npu", dtype=dtype)
    table = torch.tensor([[2, 4, 1]], device="npu", dtype=torch.int32)
    slots = torch.tensor([8, 9, 10, 11, 16, 17, 18, 19, 4, 5, 6, 7], device="npu")
    mask = torch.triu(torch.ones((2048, 2048), dtype=torch.bool, device="npu"), diagonal=1)
    for length in (2, 7, 11):
        cache.release(["a"])
        cache.write_sequence("a", slots[:length], key[:length], value[:length], length)
        stream.recent = cache.recent
        request = "a" if cache.spec.uses_window_state else None
        decoded = cache.read(table[0], length, dtype, request)
        compressed = cache.recent["a"].compressed if cache.spec.uses_window_state else length
        for start in range(0, compressed, 4):
            end = min(compressed, start + 4)
            actual = stream.read_compressed_range(table[0], start, end, dtype)
            for result, expected in zip(actual, decoded):
                torch.testing.assert_close(result, expected[start:end], atol=0, rtol=0)
        full, tiled = torch.empty_like(query), torch.empty_like(query)
        kwargs = {"request_ids": ["a"]} if cache.spec.uses_window_state else {}
        cache.attention(query, table, [1], [length], 256**-0.5, full, mask, **kwargs)
        stream.attention(query, table, [1], [length], 256**-0.5, tiled, mask, **kwargs)
        torch.testing.assert_close(tiled.float(), full.float(), atol=0.004, rtol=0.02)
        # The raw current token must participate before its later cache commit.
        if cache.spec.uses_window_state:
            expected = reference(
                query,
                torch.cat((decoded[0], key[length : length + 1])),
                torch.cat((decoded[1], value[length : length + 1])),
                256**-0.5,
            )
            stream.attention(
                query,
                table,
                [1],
                [length + 1],
                256**-0.5,
                tiled,
                mask,
                request_ids=["a"],
                updates=(slots[length : length + 1], key[length : length + 1], value[length : length + 1]),
            )
            torch.testing.assert_close(tiled.float().cpu(), expected, atol=0.004, rtol=0.02)
            assert stream.recent["a"].total == length + 1
    with pytest.raises(ValueError, match="nonnegative"):
        stream.set_decode_chunk_size(-1)
    if window:
        with pytest.raises(ValueError, match="divisible"):
            stream.set_decode_chunk_size(3)
        with pytest.raises(ValueError, match="aligned"):
            stream.read_compressed_range(table[0], 1, 4, dtype)


def test_streaming_lse_merge_handles_large_scores_and_unequal_blocks():
    cache, _ = make_cache(torch.bfloat16, recent=16)
    cache.set_decode_chunk_size(2)
    query = torch.ones((1, 8, 256), dtype=torch.bfloat16, device="npu")
    key = torch.full((7, 4, 256), -8, dtype=query.dtype, device="npu")
    key[:2] = 8
    value = torch.full_like(key, 7)
    value[:2] = 1
    table = torch.tensor([[2, 4]], dtype=torch.int32, device="npu")
    slots = torch.tensor([8, 9, 10, 11, 16, 17, 18], device="npu")
    cache.write_sequence("a", slots, key, value, 7)
    output = torch.empty_like(query)
    mask = torch.triu(torch.ones((2048, 2048), dtype=torch.bool, device="npu"), diagonal=1)
    cache.attention(query, table, [1], [7], 256**-0.5, output, mask, request_ids=["a"])
    torch.testing.assert_close(output, torch.ones_like(output), atol=0, rtol=0)
    with pytest.raises(ValueError, match="cover"):
        cache.attention(query, table[:, :1], [1], [7], 256**-0.5, output, mask, request_ids=["a"])
