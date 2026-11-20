# SPDX-License-Identifier: Apache-2.0
"""Prefill in FIA layout, with one query packing per chunk and bounded history."""

import math

import torch
import torch_npu


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
    if not 0 <= count <= 4096 or min(sink, recent, old_length) < 0 or kv_tile < 1:
        raise ValueError("invalid prefill chunk, history or tile size")
    if not count:
        return query.float(), query[..., 0].float()
    scale = 1 / math.sqrt(dim) if scale is None else scale
    q = query.permute(1, 2, 0, 3).reshape(1, heads * groups, count, dim).contiguous()
    part = torch.empty_like(q)
    part_lse = torch.empty((1, heads * groups, count, 1), dtype=torch.float32, device=q.device)
    output = lse = None

    def accumulate(key, value, mask=None):
        nonlocal output, lse
        k = key.transpose(0, 1).unsqueeze(0).to(q.dtype).contiguous()
        v = value.transpose(0, 1).unsqueeze(0).to(q.dtype).contiguous()
        torch_npu.npu_fused_infer_attention_score.out(
            query=q,
            key=k,
            value=v,
            atten_mask=mask,
            input_layout="BNSD",
            actual_seq_lengths=[count],
            actual_seq_lengths_kv=[len(key)],
            num_key_value_heads=heads,
            num_heads=heads * groups,
            scale=scale,
            sparse_mode=3 if mask is not None else 0,
            pre_tokens=2147483647,
            next_tokens=0,
            inner_precise=1,
            softmax_lse_flag=True,
            out=[part, part_lse],
        )
        if output is None:
            output, lse = part.to(dtype=torch.float32, copy=True), part_lse.clone()
        else:
            combined = torch.logaddexp(lse, part_lse)
            output = output * torch.exp(lse - combined) + part.float() * torch.exp(part_lse - combined)
            lse = combined

    sink_end = min(old_length, sink)
    recent_start = max(sink_end, old_length - recent)
    for start, end, reader in (
        (0, sink_end, read_exact),
        (sink_end, recent_start, read_cold),
        (recent_start, old_length, read_exact),
    ):
        for pos in range(start, end, kv_tile):
            accumulate(*reader(pos, min(pos + kv_tile, end)))
    # All historical tokens precede the entire query chunk. Mask only its own K/V.
    mask = torch.ones((2048, 2048), device=q.device, dtype=torch.bool).triu_(1) if count > 1 else None
    accumulate(new_key, new_value, mask)
    return (
        output.reshape(heads, groups, count, dim).permute(2, 0, 1, 3),
        lse.reshape(heads, groups, count).permute(2, 0, 1),
    )
