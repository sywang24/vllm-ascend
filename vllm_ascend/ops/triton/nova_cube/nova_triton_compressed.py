# SPDX-License-Identifier: Apache-2.0
"""Paged NOVA attention with optional Cube QK and on-chip 2-bit V for Cube PV."""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import insert_slice


@triton.jit
def _pad_query(QUERY, PADDED, G: tl.constexpr):
    head = tl.program_id(0)
    row = tl.arange(0, 16)
    dim = tl.arange(0, 256)
    q = tl.load(QUERY + (head * G + row[:, None]) * 256 + dim[None, :], row[:, None] < G, 0).to(tl.float16)
    q = q.reshape(16, 64, 4).trans(0, 2, 1).reshape(16, 256)
    tl.store(PADDED + (head * 16 + row[:, None]) * 256 + dim[None, :], q)


@triton.jit
def _compressed_pv(
    LUT,
    QUERY,
    BOOKS,
    MEAN,
    KC,
    VC,
    SC,
    TABLE,
    STARTS,
    ENDS,
    PART,
    MAXIMA,
    SUMS,
    H: tl.constexpr,
    G: tl.constexpr,
    PAGE: tl.constexpr,
    KS: tl.constexpr,
    VS: tl.constexpr,
    SS: tl.constexpr,
    TS: tl.constexpr,
    SPLITS: tl.constexpr,
    SPLIT: tl.constexpr,
    T: tl.constexpr,
    C: tl.constexpr,
    SCALE: tl.constexpr,
    CUBE_QK: tl.constexpr,
    VD: tl.constexpr,
):
    task = tl.program_id(0)
    head, split = task // SPLITS, task % SPLITS
    begin, end = tl.load(STARTS), tl.load(ENDS)
    first = begin // T * T + split * SPLIT
    q = tl.arange(0, 16)
    n = tl.arange(0, T)
    d = tl.program_id(1) * VD + tl.arange(0, VD)
    kd = tl.arange(0, 256)
    m = tl.full((16,), -1.0e30, tl.float32)
    total = tl.zeros((16,), tl.float32)
    low_sum = tl.zeros((16,), tl.float32)
    acc = tl.zeros((16, VD), tl.float32)
    mean = tl.load(MEAN + head * G + q, q < G, 0)
    if CUBE_QK:
        query = tl.load(QUERY + (head * 16 + q[:, None]) * 256 + kd[None, :])
    for offset in range(0, tl.minimum(SPLIT, end - first), T):
        pos = first + offset + n
        valid = (pos >= begin) & (pos < end)
        physical_page = tl.load(TABLE + (first + offset) // PAGE, first + offset < end, 0).to(tl.int64)
        page = tl.full((T,), 0, tl.int64) + physical_page
        # The tile cannot cross a page. Keep modulo scalar so DMA stays contiguous.
        within = (first + offset) % PAGE + n
        score = tl.zeros((16, T), tl.float32)
        if CUBE_QK:
            decoded = tl.zeros((256, T), tl.float16)
            # Component-major books keep the local gather and Cube input aligned.
            for group_start in range(0, 64, C):
                groups = tl.arange(0, C)
                code = tl.load(
                    KC + physical_page * KS + (head * 64 + group_start + groups[:, None]) * PAGE + within[None, :]
                ).to(tl.int32)
                indices = groups[:, None] * 256 + code
                for component in tl.static_range(4):
                    book = tl.load(BOOKS + (head * 4 + component) * 16384 + group_start * 256 + tl.arange(0, C * 256))
                    part = tl.gather(book, indices.reshape(C * T), 0).reshape(C, T).to(tl.float16)
                    decoded = insert_slice(decoded, part, (component * 64 + group_start, 0), (C, T), (1, 1))
            score = tl.dot(query, decoded)
        else:
            for query_id in range(0, G):
                row = tl.zeros((T,), tl.float32)
                for group_start in range(0, 64, C):
                    cg = group_start + tl.arange(0, C)
                    code = tl.load(KC + physical_page * KS + (head * 64 + cg[:, None]) * PAGE + within[None, :]).to(
                        tl.int32
                    )
                    book = tl.load(LUT + ((head * G + query_id) * 64 + group_start) * 256 + tl.arange(0, C * 256))
                    indices = tl.arange(0, C)[:, None] * 256 + code
                    dots = tl.gather(book, indices.reshape(C * T), 0).reshape(C, T)
                    row += tl.sum(dots.to(tl.float32), 0)
                score = insert_slice(score, row[None, :], (query_id, 0), (1, T), (1, 1))
        scale_base = page * SS + head * 3 * PAGE + within
        rho = tl.load(SC + scale_base).to(tl.float32)
        step = tl.load(SC + scale_base + PAGE).to(tl.float32)
        low = tl.load(SC + scale_base + 2 * PAGE).to(tl.float32)
        score = tl.where(valid[None, :] & (q[:, None] < G), (score * rho[None, :] + mean[:, None]) * SCALE, -1.0e30)
        new_m = tl.maximum(m, tl.max(score, 1))
        alpha = tl.exp(m - new_m)
        p = tl.where(valid[None, :] & (q[:, None] < G), tl.exp(score - new_m[:, None]), 0.0)
        vg = tl.arange(0, 64)
        packed = tl.load(VC + physical_page * VS + (head * 64 + vg[:, None]) * PAGE + within[None, :]).to(tl.int16)
        decoded_value = tl.zeros((VD, T), tl.float16)
        for local_component in tl.static_range(VD // 64):
            component = tl.program_id(1) * (VD // 64) + local_component
            part = ((packed >> (2 * component)) & 3).to(tl.float16)
            decoded_value = insert_slice(decoded_value, part, (local_component * 64, 0), (64, T), (1, 1))
        value = tl.trans(decoded_value)
        weights = (p * step[None, :]).to(tl.float16)
        acc = acc * alpha[:, None] + tl.dot(weights, value)
        low_sum = low_sum * alpha + tl.sum(p * low[None, :], 1)
        total = total * alpha + tl.sum(p, 1)
        m = new_m
    state = (head * G + q) * SPLITS + split
    tl.store(PART + state[:, None] * 256 + d[None, :], acc + low_sum[:, None], q[:, None] < G)
    if tl.program_id(1) == 0:
        tl.store(MAXIMA + state, m, q < G)
        tl.store(SUMS + state, total, q < G)


@triton.jit
def _merge_compressed(PART, MAXIMA, SUMS, OUT, LSE, SPLITS: tl.constexpr, B: tl.constexpr, N: tl.constexpr):
    head = tl.program_id(0)
    dim = tl.program_id(1) * N + tl.arange(0, N)
    lane = tl.arange(0, B)
    peak = tl.full((), -1.0e30, tl.float32)
    for start in range(0, SPLITS, B):
        s = start + lane
        maximum = tl.load(MAXIMA + head * SPLITS + s, s < SPLITS, -1.0e30)
        peak = tl.maximum(peak, tl.max(maximum, 0))
    denominator = tl.full((), 0.0, tl.float32)
    numerator = tl.zeros((N,), tl.float32)
    for start in range(0, SPLITS, B):
        s = start + lane
        state = head * SPLITS + s
        total = tl.load(SUMS + state, s < SPLITS, 0)
        maximum = tl.load(MAXIMA + state, s < SPLITS, -1.0e30)
        factor = tl.exp(maximum - peak)
        denominator += tl.sum(total * factor, 0)
        part = tl.load(PART + state[:, None] * 256 + dim[None, :], s[:, None] < SPLITS, 0)
        numerator += tl.sum(part * factor[:, None], 0)
    value = numerator / tl.maximum(denominator, 1.0e-30)
    out_dim = dim % 64 * 4 + dim // 64
    tl.store(OUT + head * 256 + out_dim, value)
    if tl.program_id(1) == 0:
        tl.store(LSE + head, tl.where(denominator > 0, peak + tl.log(denominator), -float("inf")))


class TritonCompressedDecode:
    """B1/D256 experimental backend with only split-state external storage."""

    def __init__(
        self,
        max_tokens,
        heads,
        groups,
        device,
        *,
        split_tokens=512,
        tile_tokens=64,
        code_groups=16,
        launch_options=None,
        cube_qk=True,
        value_dims=256,
    ):
        if (
            max_tokens < 1
            or heads < 1
            or not 1 <= groups <= 16
            or tile_tokens not in (16, 32, 64, 128)
            or split_tokens < tile_tokens
            or split_tokens % tile_tokens
            or code_groups not in (1, 2, 4, 8, 16, 32, 64)
            or value_dims not in (64, 128, 256)
        ):
            raise ValueError("unsupported compressed Triton decode geometry")
        self.max_tokens, self.heads, self.groups = max_tokens, heads, groups
        self.split_tokens, self.tile_tokens, self.code_groups = split_tokens, tile_tokens, code_groups
        self.splits = triton.cdiv(max_tokens, split_tokens)
        self.options = {"num_stages": 1, "enable_auto_bind_sub_block": False}
        self.options.update(launch_options or {})
        if self.options.get("enable_dynamic_cv_pipeline"):
            raise ValueError("dynamic CV pipeline is unsupported on Ascend910B4")
        self.cube_qk = cube_qk
        self.value_dims = value_dims
        self._books = {}
        self.query_padded = torch.empty((heads, 16, 256), device=device, dtype=torch.float16) if cube_qk else None
        self.partial = torch.empty((heads * groups, self.splits, 256), device=device, dtype=torch.float32)
        self.maximum = torch.empty((heads * groups, self.splits), device=device, dtype=torch.float32)
        self.total = torch.empty_like(self.maximum)
        self.output = torch.empty((1, heads, groups, 256), device=device, dtype=torch.float32)
        self.lse = torch.empty((1, heads, groups), device=device, dtype=torch.float32)

    def prepare_lut(self, query, books):
        return (
            torch.einsum("bhgrd,hrcd->bhgrc", query.reshape(1, self.heads, self.groups, 64, 4), books.float())
            .half()
            .contiguous()
        )

    def prepare_books(self, books):
        if id(books) not in self._books:
            if torch.npu.is_current_stream_capturing():
                raise RuntimeError("prepare codebooks before graph capture")
            prepared = books.permute(0, 3, 1, 2).float().contiguous()
            self._books[id(books)] = (books, prepared)
        return self._books[id(books)][1]

    def __call__(self, query, mean, views, books, table, starts, ends, scale, *, lut=None):
        key, value, scales = views
        if (
            query.shape != (1, self.heads, self.groups, 256)
            or not query.is_contiguous()
            or mean.shape != (1, self.heads, self.groups)
            or books.shape != (self.heads, 64, 256, 4)
            or table.ndim != 2
            or table.shape[0] != 1
            or table.stride(1) != 1
            or starts.shape != (1,)
            or ends.shape != (1,)
            or key.shape[1:] != (self.heads, 64, key.shape[-1])
            or value.shape != key.shape
            or scales.shape != (key.shape[0], self.heads, 3, key.shape[-1])
            or key.dtype != torch.uint8
            or value.dtype != torch.uint8
            or scales.dtype != torch.bfloat16
            or any(t.stride(-1) != 1 for t in views)
            or key.stride(1) != 64 * key.shape[-1]
            or key.stride(2) != key.shape[-1]
            or value.stride(1) != 64 * key.shape[-1]
            or value.stride(2) != key.shape[-1]
            or scales.stride(1) != 3 * key.shape[-1]
            or scales.stride(2) != key.shape[-1]
            or key.shape[-1] % self.tile_tokens
            or self.max_tokens > table.shape[1] * key.shape[-1]
        ):
            raise ValueError("compressed Triton decode requires B1 and aligned pages")
        if lut is None and not self.cube_qk:
            lut = self.prepare_lut(query, books)
        if (
            (
                not self.cube_qk
                and (
                    lut.shape != (1, self.heads, self.groups, 64, 256)
                    or lut.dtype != torch.float16
                    or not lut.is_contiguous()
                )
            )
            or mean.dtype != torch.float32
            or not mean.is_contiguous()
            or any(t.dtype != torch.int32 for t in (table, starts, ends))
        ):
            raise ValueError("invalid LUT or metadata for compressed Triton decode")
        if self.cube_qk:
            _pad_query[(self.heads,)](
                query, self.query_padded, self.groups, compile_mode="simd", enable_auto_bind_sub_block=False
            )
        _compressed_pv[(self.heads * self.splits, 256 // self.value_dims)](
            query if self.cube_qk else lut,
            self.query_padded if self.cube_qk else query,
            self.prepare_books(books) if self.cube_qk else books,
            mean,
            key,
            value,
            scales,
            table,
            starts,
            ends,
            self.partial,
            self.maximum,
            self.total,
            self.heads,
            self.groups,
            key.shape[-1],
            key.stride(0),
            value.stride(0),
            scales.stride(0),
            table.stride(0),
            self.splits,
            self.split_tokens,
            self.tile_tokens,
            self.code_groups,
            scale,
            self.cube_qk,
            self.value_dims,
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
        tensors = list((self.partial, self.maximum, self.total, self.output, self.lse))
        if self.query_padded is not None:
            tensors.append(self.query_padded)
        return sum(t.numel() * t.element_size() for t in tensors)
