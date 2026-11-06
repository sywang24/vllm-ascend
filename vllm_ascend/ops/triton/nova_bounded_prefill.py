# SPDX-License-Identifier: Apache-2.0
"""Bounded compressed/exact readers and the CPU prefill reference."""

import math

import torch


def planar_cold_reader(key_codes, value_codes, scales, page_table, codec):
    """Read and decode only the requested interval from planar compressed pages."""
    if key_codes.ndim != 4 or value_codes.shape != key_codes.shape:
        raise ValueError("invalid compressed code planes")
    pages, heads, _, page_size = key_codes.shape
    if scales.shape != (pages, heads, 3, page_size) or page_table.ndim != 1:
        raise ValueError("invalid compressed scales or page table")
    if any(x.device != key_codes.device for x in (value_codes, scales, page_table)):
        raise ValueError("compressed inputs must be on the same device")

    def read(start, end):
        if start < 0 or end <= start or end > page_table.numel() * page_size:
            raise ValueError("cold interval is outside the page table")
        positions = torch.arange(start, end, device=key_codes.device, dtype=torch.int64)
        pages_for_rows = page_table[positions // page_size].long()
        offsets = positions % page_size
        keys = key_codes[pages_for_rows, :, :, offsets].contiguous()
        values = value_codes[pages_for_rows, :, :, offsets].contiguous()
        row_scales = scales[pages_for_rows, :, :, offsets].contiguous()
        return codec.decode(keys, values, row_scales)

    return read


def exact_window_reader(key, value, request_slot, sink, ring, old_length):
    """Read a request's retained sink/ring rows without copying the full window."""
    if key.ndim != 4 or value.shape != key.shape or value.device != key.device:
        raise ValueError("invalid exact-window storage")
    if not 0 <= request_slot < key.shape[0] or sink < 0 or ring < 1 or old_length < 0:
        raise ValueError("invalid exact-window geometry")
    if key.shape[1] != sink + ring:
        raise ValueError("exact-window capacity does not match sink and ring")

    def read(start, end):
        if (
            start < 0
            or end <= start
            or end > old_length
            or (start < sink < end)
            or (start >= sink and start < max(sink, old_length - ring))
        ):
            raise ValueError("exact interval is outside the retained window")
        positions = torch.arange(start, end, device=key.device, dtype=torch.int64)
        offsets = torch.where(positions < sink, positions, sink + positions % ring)
        return key[request_slot, offsets], value[request_slot, offsets]

    return read


def bounded_prefill(
    query, new_key, new_value, old_length, read_cold, read_exact, sink, recent, query_tile=32, kv_tile=256, scale=None
):
    """Attend to an old segmented history and a causal, exact new chunk.

    Readers take an absolute half-open interval and return tensors shaped
    [rows, kv_heads, dim].  They must only materialize the requested rows.
    All old sink/recent rows and every new chunk row remain lossless.
    Temporary attention tensors are bounded by query_tile and kv_tile.
    The caller must retain the old exact window until every query tile has
    finished; writing the new chunk into its ring earlier can evict old rows.
    """
    if query.ndim != 4 or new_key.ndim != 3 or new_value.shape != new_key.shape:
        raise ValueError("invalid query or key/value shape")
    count, heads, groups, dim = query.shape
    if new_key.shape != (count, heads, dim):
        raise ValueError("new key/value geometry does not match query")
    if old_length < 0 or sink < 0 or recent < 0 or query_tile < 1 or kv_tile < 1:
        raise ValueError("invalid history or tile size")
    if query.device != new_key.device or new_key.device != new_value.device:
        raise ValueError("query and new key/value must be on the same device")
    scale = 1 / math.sqrt(dim) if scale is None else scale
    output = torch.zeros_like(query, dtype=torch.float32)
    maximum = torch.full(query.shape[:-1], -torch.inf, dtype=torch.float32, device=query.device)
    denominator = torch.zeros_like(maximum)
    sink_end = min(old_length, sink)
    recent_start = max(sink_end, old_length - recent)

    def accumulate(key, value, key_start=None):
        if key.shape != value.shape or key.ndim != 3 or key.shape[1:] != (heads, dim):
            raise ValueError("reader returned invalid key/value geometry")
        if key.device != query.device or value.device != query.device:
            raise ValueError("reader returned tensors on the wrong device")
        key, value = key.float(), value.float()
        for q0 in range(0, count, query_tile):
            q1 = min(count, q0 + query_tile)
            if key_start is not None and key_start >= q1:
                continue
            q = query[q0:q1].float()
            scores = torch.einsum("qhgd,khd->qhgk", q, key.float()) * scale
            if key_start is not None:
                query_positions = torch.arange(q0, q1, device=query.device)
                key_positions = torch.arange(key_start, key_start + key.shape[0], device=query.device)
                causal = key_positions[None, :] <= query_positions[:, None]
                scores = scores.masked_fill(~causal[:, None, None, :], -torch.inf)
            block_max = scores.amax(dim=-1)
            old_max = maximum[q0:q1]
            next_max = torch.maximum(old_max, block_max)
            safe_max = torch.where(torch.isfinite(next_max), next_max, torch.zeros_like(next_max))
            previous_weight = torch.exp(old_max - safe_max)
            probabilities = torch.exp(scores - safe_max[..., None])
            output[q0:q1] = output[q0:q1] * previous_weight[..., None] + torch.einsum(
                "qhgk,khd->qhgd", probabilities, value
            )
            denominator[q0:q1] = denominator[q0:q1] * previous_weight + probabilities.sum(dim=-1)
            maximum[q0:q1] = next_max

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
                accumulate(key, value)
        for pos in range(0, count, kv_tile):
            stop = min(count, pos + kv_tile)
            accumulate(new_key[pos:stop], new_value[pos:stop], pos)
    return output / denominator[..., None], maximum + denominator.log()
