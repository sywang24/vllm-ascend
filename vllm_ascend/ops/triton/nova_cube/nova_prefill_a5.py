# SPDX-License-Identifier: Apache-2.0
"""Stage one request's prefill K/V for a single right-aligned FIA call."""

import math
from functools import lru_cache

import torch
import torch_npu

from vllm_ascend.ops.triton.nova_cube.nova_prefill_wide import bounded_fia_prefill as wide_prefill

_MAX_STAGE_TOKENS = 96 * 1024


@lru_cache(maxsize=8)
def _mask(device):
    return torch.ones((2048, 2048), device=device, dtype=torch.bool).triu_(1)


def bounded_fia_prefill(
    query,
    new_key,
    new_value,
    old_length,
    read_cold,
    read_exact,
    sink,
    recent,
    query_tile=2048,
    kv_tile=16384,
    scale=None,
):
    count, heads, groups, dim = query.shape
    if new_key.shape != (count, heads, dim) or new_value.shape != new_key.shape:
        raise ValueError("invalid new K/V geometry")
    if not 0 <= count <= 8192 or min(sink, recent, old_length) < 0 or kv_tile < 1:
        raise ValueError("invalid prefill chunk, history or tile size")
    if not count:
        return query.float(), query[..., 0].float()

    total = old_length + count
    if total > _MAX_STAGE_TOKENS:
        if count <= 4096:
            return wide_prefill(
                query,
                new_key,
                new_value,
                old_length,
                read_cold,
                read_exact,
                sink,
                recent,
                query_tile=query_tile,
                kv_tile=kv_tile,
                scale=scale,
            )
        raise ValueError("8K prefill requires an attention context of at most 96K")

    q = query.permute(1, 2, 0, 3).reshape(1, heads * groups, count, dim).contiguous()
    if old_length:
        # Each layer reuses the allocator's temporary space; nothing is added to the KV cache.
        k = torch.empty((1, heads, total, dim), dtype=q.dtype, device=q.device)
        v = torch.empty_like(k)

        def put(start, end, key, value):
            k[0, :, start:end, :].copy_(key.transpose(0, 1))
            v[0, :, start:end, :].copy_(value.transpose(0, 1))

        sink_end = min(old_length, sink)
        recent_start = max(sink_end, old_length - recent)
        for start, end, read in (
            (0, sink_end, read_exact),
            (sink_end, recent_start, read_cold),
            (recent_start, old_length, read_exact),
        ):
            for pos in range(start, end, kv_tile):
                stop = min(pos + kv_tile, end)
                put(pos, stop, *read(pos, stop))
        put(old_length, total, new_key, new_value)
    else:
        k = new_key.transpose(0, 1).unsqueeze(0).to(q.dtype).contiguous()
        v = new_value.transpose(0, 1).unsqueeze(0).to(q.dtype).contiguous()

    # Right-aligned causal masking shifts the diagonal by total - count.
    mask = _mask(q.device) if count > 1 else None
    part, part_lse = torch_npu.npu_fused_infer_attention_score(
        q,
        k,
        v,
        atten_mask=mask,
        input_layout="BNSD",
        actual_seq_lengths=[count],
        actual_seq_lengths_kv=[total],
        num_key_value_heads=heads,
        num_heads=heads * groups,
        scale=1 / math.sqrt(dim) if scale is None else scale,
        sparse_mode=3 if mask is not None else 0,
        pre_tokens=2147483647,
        next_tokens=0,
        inner_precise=1,
        softmax_lse_flag=True,
    )
    return (
        part.reshape(heads, groups, count, dim).permute(2, 0, 1, 3),
        part_lse.reshape(heads, groups, count).permute(2, 0, 1),
    )
