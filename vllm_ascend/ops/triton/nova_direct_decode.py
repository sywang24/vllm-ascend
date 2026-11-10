# SPDX-License-Identifier: Apache-2.0
"""Single-launch direct compressed attention, with bounded split outputs."""

import ctypes
import json
from pathlib import Path

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _merge(
    PART,
    MAXIMA,
    SUMS,
    STARTS,
    ENDS,
    OUT,
    LSE,
    D: tl.constexpr,
    HEAD_GROUPS: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
    SPLIT_TOKENS: tl.constexpr,
    TILE_TOKENS: tl.constexpr,
    VALUE_PLANAR: tl.constexpr = False,
):
    head = tl.program_id(0)
    base = head * SPLITS
    request = head // HEAD_GROUPS
    begin, end = tl.load(STARTS + request), tl.load(ENDS + request)
    active_splits = tl.where(begin < end, tl.cdiv(end - begin // TILE_TOKENS * TILE_TOKENS, SPLIT_TOKENS), 0)
    dims = tl.arange(0, D)
    part_dims = dims % 4 * 64 + dims // 4 if VALUE_PLANAR else dims
    maximum = tl.full((), -1.0e30, tl.float32)
    for start in range(0, SPLITS, BLOCK):
        splits = start + tl.arange(0, BLOCK)
        m = tl.load(MAXIMA + (base + splits) * 8, (splits < SPLITS) & (splits < active_splits), -1.0e30)
        maximum = tl.maximum(maximum, tl.max(m, 0))
    accum = tl.zeros((D,), tl.float32)
    denominator = tl.full((), 0.0, tl.float32)
    for start in range(0, SPLITS, BLOCK):
        splits = start + tl.arange(0, BLOCK)
        valid = (splits < SPLITS) & (splits < active_splits)
        m = tl.load(MAXIMA + (base + splits) * 8, valid, -1.0e30)
        total = tl.load(SUMS + (base + splits) * 8, valid, 0.0)
        weight = tl.exp(m - maximum)
        part = tl.load(PART + (base + splits[:, None]) * D + part_dims[None, :], valid[:, None], 0.0)
        accum += tl.sum(part * weight[:, None], 0)
        denominator += tl.sum(total * weight, 0)
    tl.store(OUT + head * D + dims, accum / tl.maximum(denominator, 1.0e-30))
    tl.store(LSE + head, tl.where(denominator > 0, maximum + tl.log(denominator), -float("inf")))


class DirectDecode:
    def __init__(
        self,
        capacity,
        heads,
        groups,
        device,
        library_root,
        max_tokens,
        split_tokens=1024,
        block_count=40,
        half_lut=True,
        tile_tokens=128,
    ):
        library_dir = Path(library_root) / "lib"
        manifest = library_dir / "nova_kernel_abi.json"
        if not manifest.is_file():
            raise ValueError(f"missing Nova kernel ABI manifest: rebuild {library_root}")
        abi = json.loads(manifest.read_text())
        expected_dtype = "float16" if half_lut else "float32"
        if (
            abi.get("abi_version") != 1
            or abi.get("head_dim") != 256
            or abi.get("lut_dtype") != expected_dtype
            or abi.get("tile_tokens") != tile_tokens
            or split_tokens > abi.get("max_split_tokens", 0)
            or not 1 <= groups <= abi.get("max_query_groups", 0)
        ):
            raise ValueError(
                f"Nova kernel ABI mismatch: {abi}; requested "
                f"LUT={expected_dtype}, tile={tile_tokens}, "
                f"split={split_tokens}, groups={groups}"
            )
        if (
            split_tokens not in (128, 256, 512, 1024)
            or not 1 <= block_count <= 80
            or tile_tokens != 128
            or not half_lut
            or capacity < 1
            or heads < 1
            or max_tokens < 1
        ):
            raise ValueError("direct decode requires half-LUT tile=128 and split=128/256/512/1024")
        self.capacity, self.heads, self.groups = capacity, heads, groups
        self.split_tokens, self.block_count = split_tokens, block_count
        self.half_lut = half_lut
        self.tile_tokens = tile_tokens
        self.page_offset_bits = abi.get("page_offset_bits", 32)
        self.max_splits = triton.cdiv(max_tokens, split_tokens)
        tasks = capacity * heads * groups * self.max_splits
        # The merge masks inactive splits, including graph padding rows.
        self.partial = torch.zeros((tasks, 256), device=device, dtype=torch.float32)
        self.maximum = torch.full((tasks, 8), -1.0e30, device=device, dtype=torch.float32)
        self.total = torch.zeros_like(self.maximum)
        self.output = torch.empty((capacity, heads, groups, 256), device=device, dtype=torch.float32)
        self.lse = torch.empty((capacity, heads, groups), device=device, dtype=torch.float32)
        self.group_offsets = (
            torch.arange(64, device=device, dtype=torch.int32)[:, None].expand(64, tile_tokens).contiguous()
            * 256
            * (2 if half_lut else 4)
        )
        dimensions = torch.arange(256, device=device, dtype=torch.int32)
        self.dimension_offsets = (dimensions % 4 * 64 + dimensions // 4) * 4
        self.library = ctypes.CDLL(str(library_dir / "libnova_attention_shared.so"))
        self.launch = self.library.aclrtlaunch_nova_attention_shared
        self.launch.argtypes = (
            [ctypes.c_uint32, ctypes.c_void_p] + [ctypes.c_void_p] * 13 + [ctypes.c_uint32] * 11 + [ctypes.c_float]
        )
        self.launch.restype = ctypes.c_int

    def prepare_lut(self, query, books):
        batch, heads, groups, _ = query.shape
        lut = torch.einsum("bhgrd,hrcd->bhgrc", query.reshape(batch, heads, groups, 64, 4), books.float()).contiguous()
        return lut.half() if self.half_lut else lut

    def __call__(
        self, query, mean, key_codes, value_codes, scales, books, table, starts, ends, max_tokens, scale, *, lut=None
    ):
        batch, heads, groups, dim = query.shape
        splits = triton.cdiv(max_tokens, self.split_tokens)
        if (
            not 0 < batch <= self.capacity
            or (heads, groups, dim) != (self.heads, self.groups, 256)
            or not 1 <= groups <= 6
            or key_codes.shape[-1] < self.tile_tokens
            or key_codes.shape[-1] % 32
            or table.stride(1) != 1
            or table.shape[1] > 2048
            or not 0 < max_tokens <= table.shape[1] * key_codes.shape[-1]
            or splits > self.max_splits
        ):
            raise ValueError("unsupported direct decode geometry")
        if (
            query.dtype != torch.float32
            or mean.dtype != torch.float32
            or table.dtype != torch.int32
            or starts.dtype != torch.int32
            or ends.dtype != torch.int32
        ):
            raise ValueError("direct decode requires FP32 query and INT32 metadata")
        if lut is None:
            lut = self.prepare_lut(query, books)
        elif (
            lut.shape != (batch, heads, groups, 64, 256)
            or lut.dtype != torch.float16
            or lut.device != query.device
            or not lut.is_contiguous()
        ):
            raise ValueError("invalid direct decode LUT")
        result = self.launch(
            self.block_count,
            torch.npu.current_stream().npu_stream,
            lut.data_ptr(),
            mean.data_ptr(),
            key_codes.data_ptr(),
            value_codes.data_ptr(),
            scales.data_ptr(),
            table.data_ptr(),
            starts.data_ptr(),
            ends.data_ptr(),
            self.group_offsets.data_ptr(),
            self.dimension_offsets.data_ptr(),
            self.partial.data_ptr(),
            self.maximum.data_ptr(),
            self.total.data_ptr(),
            batch,
            heads,
            groups,
            key_codes.shape[-1],
            table.shape[1],
            splits,
            self.split_tokens,
            key_codes.stride(0),
            scales.stride(0),
            table.stride(0),
            0,
            scale,
        )
        if result:
            raise RuntimeError(f"NOVA direct launch failed: {result}")
        output, lse = self.output[:batch], self.lse[:batch]
        _merge[(batch * heads * groups,)](
            self.partial,
            self.maximum,
            self.total,
            starts,
            ends,
            output,
            lse,
            dim,
            heads * groups,
            splits,
            16,
            self.split_tokens,
            self.tile_tokens,
        )
        return output, lse

    def workspace_bytes(self):
        return sum(
            t.numel() * t.element_size()
            for t in (
                self.partial,
                self.maximum,
                self.total,
                self.output,
                self.lse,
                self.group_offsets,
                self.dimension_offsets,
            )
        )
