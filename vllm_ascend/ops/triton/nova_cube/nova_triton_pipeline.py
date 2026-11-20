# SPDX-License-Identifier: Apache-2.0
"""Persistent AIV LUT scoring and Cube PV, consuming planar compressed KV."""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.nova_cube.nova_query_lut import _query_lut
from vllm_ascend.ops.triton.nova_cube.nova_triton_compressed import TritonCompressedDecode, _merge_compressed
from vllm_ascend.ops.triton.triton_utils import insert_slice


@triton.jit
def _score_persistent(
    LUT,
    MEAN,
    KC,
    SC,
    TABLE,
    STARTS,
    ENDS,
    SCORES,
    G: tl.constexpr,
    CAP: tl.constexpr,
    PAGE: tl.constexpr,
    KS: tl.constexpr,
    SS: tl.constexpr,
    T: tl.constexpr,
    SCALE: tl.constexpr,
    ALIGN: tl.constexpr,
    PARTS: tl.constexpr,
    H: tl.constexpr = 0,
    TS: tl.constexpr = 0,
):
    query_head = tl.program_id(1)
    head = query_head // G
    batch = head // H if H else 0
    head = head % H if H else head
    TABLE += batch * TS
    begin, end = tl.load(STARTS + batch), tl.load(ENDS + batch)
    origin = begin // ALIGN * ALIGN
    n, groups = tl.arange(0, T), tl.arange(0, 64)
    # Reuse this query head's complete LUT across token tiles on the same AIV.
    book = tl.load(LUT + query_head * 16384 + tl.arange(0, 16384)).to(tl.float32)
    mean = tl.load(MEAN + query_head)
    for tile in range(tl.program_id(0), CAP // T, PARTS):
        first = origin + tile * T
        if first < end:
            page = tl.load(TABLE + first // PAGE).to(tl.int64)
            within = first % PAGE + n
            code = tl.load(KC + page * KS + (head * 64 + groups[:, None]) * PAGE + within[None, :]).to(tl.int32)
            indices = groups[:, None] * 256 + code
            dots = tl.gather(book, indices.reshape(64 * T), 0).reshape(64, T)
            rho = tl.load(SC + page * SS + head * 3 * PAGE + within).to(tl.float32)
            score = tl.where((first + n >= begin) & (first + n < end), (tl.sum(dots, 0) * rho + mean) * SCALE, -1.0e30)
            tl.store(SCORES + query_head * CAP + tile * T + n, score)


@triton.jit
def _pv_codes(
    SCORES,
    VC,
    SC,
    TABLE,
    STARTS,
    ENDS,
    PART,
    MAXIMA,
    SUMS,
    G: tl.constexpr,
    CAP: tl.constexpr,
    PAGE: tl.constexpr,
    VS: tl.constexpr,
    SS: tl.constexpr,
    SPLITS: tl.constexpr,
    SPLIT: tl.constexpr,
    T: tl.constexpr,
    VD: tl.constexpr,
    ALIGN: tl.constexpr,
):
    task = tl.program_id(0)
    head, split = task // SPLITS, task % SPLITS
    begin, end = tl.load(STARTS), tl.load(ENDS)
    first = begin // ALIGN * ALIGN + split * SPLIT
    row, n, vg = tl.arange(0, 16), tl.arange(0, T), tl.arange(0, 64)
    dims = tl.program_id(1) * VD + tl.arange(0, VD)
    maximum = tl.full((16,), -1.0e30, tl.float32)
    total, bias = tl.zeros((16,), tl.float32), tl.zeros((16,), tl.float32)
    accum = tl.zeros((16, VD), tl.float32)
    for offset in range(0, tl.minimum(SPLIT, end - first), T):
        pos = first + offset + n
        valid = (pos >= begin) & (pos < end)
        score = tl.load(
            SCORES + (head * G + row[:, None]) * CAP + split * SPLIT + offset + n[None, :], row[:, None] < G, -1.0e30
        )
        next_maximum = tl.maximum(maximum, tl.max(score, 1))
        alpha = tl.exp(maximum - next_maximum)
        p = tl.where(valid[None, :] & (row[:, None] < G), tl.exp(score - next_maximum[:, None]), 0.0)
        page = tl.load(TABLE + (first + offset) // PAGE).to(tl.int64)
        # Keep modulo scalar so the compiler emits contiguous tile DMA.
        within = (first + offset) % PAGE + n
        scale_base = SC + page * SS + head * 3 * PAGE + within
        step = tl.load(scale_base + PAGE).to(tl.float32)
        low = tl.load(scale_base + 2 * PAGE).to(tl.float32)
        packed = tl.load(VC + page * VS + (head * 64 + vg[:, None]) * PAGE + within[None, :]).to(tl.int16)
        value = tl.zeros((VD, T), tl.float16)
        for local_component in tl.static_range(VD // 64):
            component = tl.program_id(1) * (VD // 64) + local_component
            part = ((packed >> (2 * component)) & 3).to(tl.float16)
            value = insert_slice(value, part, (local_component * 64, 0), (64, T), (1, 1))
        accum = accum * alpha[:, None] + tl.dot((p * step[None, :]).to(tl.float16), tl.trans(value))
        bias = bias * alpha + tl.sum(p * low[None, :], 1)
        total = total * alpha + tl.sum(p, 1)
        maximum = next_maximum
    state = (head * G + row) * SPLITS + split
    tl.store(PART + state[:, None] * 256 + dims[None, :], accum + bias[:, None], row[:, None] < G)
    if tl.program_id(1) == 0:
        tl.store(MAXIMA + state, maximum, row < G)
        tl.store(SUMS + state, total, row < G)


class TritonPipelineDecode(TritonCompressedDecode):
    """B1/D256 experiment. Inputs have the existing planar paged-cache layout."""

    def __init__(
        self,
        max_tokens,
        heads,
        groups,
        device,
        *,
        split_tokens=1024,
        tile_tokens=64,
        score_tile=128,
        score_parts=10,
        value_dims=256,
        launch_options=None,
        batch=1,
    ):
        super().__init__(
            max_tokens,
            heads,
            groups,
            device,
            split_tokens=split_tokens,
            tile_tokens=tile_tokens,
            launch_options=launch_options,
            cube_qk=False,
            value_dims=value_dims,
        )
        if score_tile not in (32, 64, 128) or split_tokens % score_tile or score_parts < 1:
            raise ValueError("unsupported score/PV tile geometry")
        self.score_tile, self.score_parts = score_tile, score_parts
        if batch < 1:
            raise ValueError("batch must be positive")
        self.batch = batch
        self.partial = torch.empty((batch * heads * groups, self.splits, 256), device=device, dtype=torch.float32)
        self.maximum = torch.empty((batch * heads * groups, self.splits), device=device, dtype=torch.float32)
        self.total = torch.empty_like(self.maximum)
        self.output = torch.empty((batch, heads, groups, 256), device=device, dtype=torch.float32)
        self.lse = torch.empty((batch, heads, groups), device=device, dtype=torch.float32)
        self.alignment = max(score_tile, tile_tokens)
        self.score_capacity = self.splits * split_tokens
        self.lut = torch.empty((batch, heads, groups, 64, 256), dtype=torch.float16, device=device)
        self.scores = torch.empty((batch * heads * groups, self.score_capacity), dtype=torch.float32, device=device)

    def prepare_lut(self, query, books):
        _query_lut[(64, self.batch * self.heads * self.groups)](
            query,
            self.prepare_books(books),
            self.lut,
            self.groups,
            self.heads,
            compile_mode="simd",
            num_stages=1,
            enable_auto_bind_sub_block=False,
            enable_fp_fusion=False,
        )
        return self.lut

    def __call__(self, query, mean, views, books, table, starts, ends, scale, *, lut=None):
        key, value, scales = views
        page = key.shape[-1]
        if (
            query.shape != (self.batch, self.heads, self.groups, 256)
            or query.dtype != torch.float32
            or not query.is_contiguous()
            or mean.shape != (self.batch, self.heads, self.groups)
            or mean.dtype != torch.float32
            or not mean.is_contiguous()
            or books.shape != (self.heads, 64, 256, 4)
            or table.ndim != 2
            or table.shape[0] != self.batch
            or table.stride(1) != 1
            or starts.shape != (self.batch,)
            or ends.shape != (self.batch,)
            or any(t.dtype != torch.int32 for t in (table, starts, ends))
            or key.shape[1:] != (self.heads, 64, page)
            or value.shape != key.shape
            or scales.shape != (key.shape[0], self.heads, 3, page)
            or key.dtype != torch.uint8
            or value.dtype != torch.uint8
            or scales.dtype != torch.bfloat16
            or any(t.stride(-1) != 1 for t in views)
            or any(t.stride(1) != 64 * page or t.stride(2) != page for t in (key, value))
            or scales.stride(1) != 3 * page
            or scales.stride(2) != page
            or page % self.alignment
            or self.max_tokens > table.shape[1] * page
        ):
            raise ValueError(
                "pipeline requires grouped FP32 queries, planar cache, and aligned pages: "
                f"query={query.shape}/{query.dtype}/{query.stride()} mean={mean.shape}/{mean.dtype}/{mean.stride()} "
                f"table={table.shape}/{table.dtype}/{table.stride()} key={key.shape}/{key.dtype}/{key.stride()} "
                f"value={value.shape}/{value.dtype}/{value.stride()} "
                f"scales={scales.shape}/{scales.dtype}/{scales.stride()} "
                f"bounds={starts.shape}/{starts.dtype},{ends.shape}/{ends.dtype} "
                f"page={page} align={self.alignment} max={self.max_tokens}"
            )
        if lut is None:
            lut = self.prepare_lut(query, books)
        elif lut.shape != self.lut.shape or lut.dtype != torch.float16 or not lut.is_contiguous():
            raise ValueError("invalid FP16 Query LUT")
        return self._attend(lut, mean, views, table, starts, ends, scale)

    def _attend(self, lut, mean, views, table, starts, ends, scale):
        key, value, scales = views
        page = key.shape[-1]
        _score_persistent[(self.score_parts, self.heads * self.groups)](
            lut,
            mean,
            key,
            scales,
            table,
            starts,
            ends,
            self.scores,
            self.groups,
            self.score_capacity,
            page,
            key.stride(0),
            scales.stride(0),
            self.score_tile,
            scale,
            self.alignment,
            self.score_parts,
            compile_mode="simd",
            num_stages=1,
            enable_auto_bind_sub_block=False,
        )
        _pv_codes[(self.heads * self.splits, 256 // self.value_dims)](
            self.scores,
            value,
            scales,
            table,
            starts,
            ends,
            self.partial,
            self.maximum,
            self.total,
            self.groups,
            self.score_capacity,
            page,
            value.stride(0),
            scales.stride(0),
            self.splits,
            self.split_tokens,
            self.tile_tokens,
            self.value_dims,
            self.alignment,
            **self.options,
        )
        _merge_compressed[(self.heads * self.groups, 4)](
            self.partial,
            self.maximum,
            self.total,
            self.output,
            self.lse,
            self.splits,
            min(64, triton.next_power_of_2(self.splits)),
            64,
            compile_mode="simd",
            enable_auto_bind_sub_block=False,
        )
        return self.output, self.lse

    def workspace_bytes(self):
        return (
            super().workspace_bytes()
            + self.scores.numel() * self.scores.element_size()
            + self.lut.numel() * self.lut.element_size()
        )
