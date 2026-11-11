"""NPU regression for stateful D-Quant decode graph replay."""

import os
from unittest.mock import patch

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.attention.dquant_attention import DQuantAttentionCache
from vllm_ascend.core.kv_cache_interface import DQuantAttentionSpec
from vllm_ascend.ops.triton.dquant_decode_pool import DQuantBatchSlots, DQuantGraphWorkspace


def make_cache(spec, capacity, blocks):
    storage = [
        torch.zeros((blocks, spec.block_size, width), device="npu", dtype=torch.uint8)
        for width in (spec.key_record_bytes, spec.value_record_bytes)
    ]
    return DQuantAttentionCache(storage, spec, max_num_seqs=capacity, output_chunk=64)


def replay(batch, capture_mode="global"):
    import faulthandler

    faulthandler.dump_traceback_later(90, repeat=True)
    torch.npu.set_device(0)
    import torch_npu._inductor  # noqa: F401

    torch.manual_seed(4401)
    spec = DQuantAttentionSpec(
        block_size=128,
        num_kv_heads=2,
        head_size=256,
        dtype=torch.bfloat16,
        dquant_recent_window=128,
        dquant_key_window=64,
        dquant_key_levels=8,
        dquant_value_levels=6,
        dquant_key_container_bytes=155,
        dquant_value_container_bytes=109,
        dquant_alpha=1.4,
    )
    capacity, max_length = batch, 512
    graph_cache = make_cache(spec, capacity, batch * 4)
    reference = make_cache(spec, capacity, batch * 4)
    workspace = DQuantGraphWorkspace(capacity, max_length)
    batch_slots = DQuantBatchSlots(capacity, torch.device("npu:0"))
    tables = torch.arange(batch * 4, device="npu", dtype=torch.int32).view(batch, 4)
    table_input = tables.clone()
    q = torch.randn((batch, 12, 256), device="npu", dtype=torch.bfloat16)
    k = torch.randn((batch, 2, 256), device="npu", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    current_slots = torch.zeros(batch, device="npu", dtype=torch.int64)
    out = torch.empty_like(q)
    metadata, _ = batch_slots.prepare([], set(), [], capturing=True)

    def forward():
        return graph_cache.graph_decode(q, table_input, metadata, 256**-0.5, out, (current_slots, k, v), workspace)

    for _ in range(2):
        forward()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode=capture_mode):
        forward()
    torch.npu.synchronize()
    print(f"Captured batch={batch}", flush=True)

    ids = [f"r{i}" for i in range(batch)]
    lengths = [191, 127, 0, 255][:batch] if batch <= 4 else [191 + i % 3 for i in range(batch)]
    _, owners = batch_slots.prepare(ids, set(ids), [length + 1 for length in lengths])
    graph_cache.bind_recent_slots(ids, owners, [0] * batch)
    for row, length in enumerate(lengths):
        if not length:
            continue
        pos = torch.arange(length, device="npu")
        slots = tables[row, pos // 128].long() * 128 + pos % 128
        key = torch.randn((length, 2, 256), device="npu", dtype=torch.bfloat16)
        val = torch.randn_like(key)
        for cache in (graph_cache, reference):
            cache.write_sequence(ids[row], slots, key, val, length)
    pointer = graph_cache.recent_pool.key.data_ptr()
    max_error = 0.0
    for step in range(70):
        order = list(reversed(range(batch))) if step % 2 else list(range(batch))
        # Drop a row temporarily, then reintroduce it without losing its slot.
        if batch > 1 and step % 5 == 2:
            order = order[:-1]
        request_ids = [ids[i] for i in order]
        metadata, owners = batch_slots.prepare(request_ids, set(ids), [lengths[i] + 1 for i in order])
        table_input.zero_()
        table_input[: len(order)].copy_(tables[order])
        q.normal_()
        k.normal_()
        v.normal_()
        current_slots.fill_(-1)
        for row, index in enumerate(order):
            length = lengths[index]
            current_slots[row] = tables[index, length // 128] * 128 + length % 128
        expected = []
        for row, index in enumerate(order):
            key, val = k[row : row + 1], v[row : row + 1]
            if lengths[index]:
                old_k, old_v = reference.read(tables[index], lengths[index], q.dtype, ids[index])
                key, val = torch.cat((old_k, key)), torch.cat((old_v, val))
            result, _ = torch_npu.npu_fused_infer_attention_score(
                query=q[row : row + 1],
                key=key,
                value=val,
                input_layout="TND",
                num_heads=12,
                num_key_value_heads=2,
                scale=256**-0.5,
                actual_seq_lengths=[1],
                actual_seq_lengths_kv=[key.shape[0]],
                sparse_mode=0,
            )
            expected.append(result.view(1, 12, 256))
        graph.replay()
        workspace.check_errors()
        actual = out[: len(order)]
        expected = torch.cat(expected)
        error = (actual.float() - expected.float()).abs().max().item()
        max_error = max(max_error, error)
        torch.testing.assert_close(actual, expected, atol=0.025, rtol=0.025)
        assert graph_cache.recent_pool.key.data_ptr() == pointer
        for row, index in enumerate(order):
            lengths[index] += 1
            reference.write_sequence(
                ids[index], current_slots[row : row + 1], k[row : row + 1], v[row : row + 1], lengths[index]
            )
            state = reference.recent[ids[index]]
            owner = batch_slots.owners[ids[index]]
            assert graph_cache.recent_pool.totals[owner].item() == lengths[index]
            torch.testing.assert_close(
                graph_cache.recent_pool.key[owner, : state.length], state.key[: state.length], atol=0, rtol=0
            )
            torch.testing.assert_close(
                graph_cache.recent_pool.value[owner, : state.length], state.value[: state.length], atol=0, rtol=0
            )
            if state.compressed:
                for actual_records, expected_records in zip(
                    (graph_cache.key, graph_cache.value), (reference.key, reference.value)
                ):
                    slots = tables[index, : state.compressed // 128 + 1]
                    pos = torch.arange(state.compressed, device="npu")
                    addresses = slots[pos // 128].long() * 128 + pos % 128
                    torch.testing.assert_close(
                        actual_records.storage[addresses], expected_records.storage[addresses], atol=0, rtol=0
                    )
        if step < 4 or step % 16 == 0:
            print(f"Batch={batch} step={step} max_error={max_error:.6f}", flush=True)

    # A finished request's slot can safely serve a fresh request at prefix zero.
    removed = ids[0]
    released_slot = batch_slots.owners[removed]
    graph_cache.release([removed])
    active = set(ids[1:]) | {"replacement"}
    batch_slots.prepare(["replacement"], active, [1])
    assert batch_slots.owners["replacement"] == released_slot
    current_slots[0] = tables[0, 0] * 128
    graph.replay()
    workspace.check_errors()
    torch.testing.assert_close(out[0], v[0].repeat_interleave(6, dim=0), atol=0, rtol=0)
    # A stale prefix must be rejected before its sampled output is returned.
    batch_slots.prepare(["replacement"], active, [3])
    graph.replay()
    try:
        workspace.check_errors()
    except RuntimeError as error:
        assert "code=2" in str(error)
    else:
        raise AssertionError("Stale decode prefix was accepted")
    print(f"PASS batch={batch}, 70 stateful replays, max_error={max_error:.6f}", flush=True)
    faulthandler.cancel_dump_traceback_later()


@pytest.mark.parametrize("batch", [1, 4, 32])
@pytest.mark.skipif(not torch.npu.is_available(), reason="Ascend NPU is unavailable")
def test_full_decode_replay(batch):
    with patch.dict(
        os.environ,
        {
            "VLLM_ASCEND_DQUANT_FUSED_ATTENTION": "1",
            "VLLM_ASCEND_DQUANT_FUSED_CHUNK_TOKENS": "64",
            "VLLM_ASCEND_DQUANT_FUSED_TILE_TOKENS": "64",
        },
    ):
        replay(batch)


if __name__ == "__main__":
    import sys

    test_full_decode_replay(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
