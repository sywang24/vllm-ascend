# SPDX-License-Identifier: Apache-2.0
"""Experimental eager FIA prefill with bounded decoded history tiles."""

import math

import torch
import torch_npu


def bounded_fia_prefill(
    query, new_key, new_value, old_length, read_cold, read_exact, sink, recent, query_tile=32, kv_tile=2048, scale=None
):
    if query.ndim != 4 or new_key.ndim != 3 or new_value.shape != new_key.shape:
        raise ValueError("invalid query or key/value shape")
    count, heads, groups, dim = query.shape
    if new_key.shape != (count, heads, dim) or query_tile < 1 or query_tile > 2048:
        raise ValueError("invalid query geometry or tile size")
    if old_length < 0 or sink < 0 or recent < 0 or kv_tile < 1:
        raise ValueError("invalid history or tile size")
    if query.device != new_key.device or new_key.device != new_value.device:
        raise ValueError("query and key/value must be on the same device")
    scale = 1 / math.sqrt(dim) if scale is None else scale
    output = torch.zeros_like(query, dtype=torch.float32)
    lse = torch.full(query.shape[:-1], -torch.inf, dtype=torch.float32, device=query.device)
    mask = torch.ones((2048, 2048), device=query.device, dtype=torch.bool).triu(1)

    def accumulate(key, value, q0, q1, causal=False):
        if key.shape != value.shape or key.ndim != 3 or key.shape[1:] != (heads, dim):
            raise ValueError("invalid reader output")
        if key.device != query.device or value.device != query.device:
            raise ValueError("reader returned tensors on the wrong device")
        q = query[q0:q1].permute(1, 2, 0, 3).reshape(1, heads * groups, q1 - q0, dim).contiguous()
        k = key.permute(1, 0, 2).unsqueeze(0).to(query.dtype).contiguous()
        v = value.permute(1, 0, 2).unsqueeze(0).to(query.dtype).contiguous()
        part = torch.empty_like(q)
        part_lse = torch.empty((1, heads * groups, q1 - q0, 1), device=query.device, dtype=torch.float32)
        torch_npu.npu_fused_infer_attention_score.out(
            query=q,
            key=k,
            value=v,
            atten_mask=mask if causal else None,
            input_layout="BNSD",
            actual_seq_lengths=[q1 - q0],
            actual_seq_lengths_kv=[len(key)],
            num_key_value_heads=heads,
            num_heads=heads * groups,
            scale=scale,
            sparse_mode=3 if causal else 0,
            pre_tokens=2147483647,
            next_tokens=0,
            inner_precise=1,
            softmax_lse_flag=True,
            out=[part, part_lse],
        )
        part = part.reshape(heads, groups, q1 - q0, dim).permute(2, 0, 1, 3).float()
        part_lse = part_lse.reshape(heads, groups, q1 - q0).permute(2, 0, 1)
        combined = torch.logaddexp(lse[q0:q1], part_lse)
        output[q0:q1] = (
            output[q0:q1] * torch.exp(lse[q0:q1] - combined)[..., None]
            + part * torch.exp(part_lse - combined)[..., None]
        )
        lse[q0:q1] = combined

    sink_end = min(old_length, sink)
    recent_start = max(sink_end, old_length - recent)
    if count:
        for start, end, reader in (
            (0, sink_end, read_exact),
            (sink_end, recent_start, read_cold),
            (recent_start, old_length, read_exact),
        ):
            for pos in range(start, end, kv_tile):
                stop = min(end, pos + kv_tile)
                key, value = reader(pos, stop)
                if key.shape[0] != stop - pos:
                    raise ValueError("reader returned an incorrect number of rows")
                for q0 in range(0, count, query_tile):
                    accumulate(key, value, q0, min(count, q0 + query_tile))
        for q0 in range(0, count, query_tile):
            q1 = min(count, q0 + query_tile)
            for pos in range(0, q0, kv_tile):
                stop = min(q0, pos + kv_tile)
                accumulate(new_key[pos:stop], new_value[pos:stop], q0, q1)
            accumulate(new_key[q0:q1], new_value[q0:q1], q0, q1, causal=True)
    return output, lse
