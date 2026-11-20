# SPDX-License-Identifier: Apache-2.0
import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _query_lut(QUERY, BOOKS, LUT, G: tl.constexpr, H: tl.constexpr = 0):
    group, query_head = tl.program_id(0), tl.program_id(1)
    head = query_head // G
    head = head % H if H else head
    code = tl.arange(0, 256)
    dot = tl.zeros((256,), tl.float32)
    for component in tl.static_range(4):
        query = tl.load(QUERY + query_head * 256 + group * 4 + component)
        book = tl.load(BOOKS + (head * 4 + component) * 16384 + group * 256 + code)
        dot += book * query
    tl.store(LUT + (query_head * 64 + group) * 256 + code, dot.to(tl.float16))


class TritonQueryLut:
    """The returned LUT is scratch storage valid until the next call."""

    def __init__(self, books, groups, max_batch=1):
        if books.ndim != 4 or books.shape[1:] != (64, 256, 4) or not 1 <= groups <= 16:
            raise ValueError("Query LUT requires Hx64x256x4 books and 1..16 query groups")
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("prepare the LUT workspace before graph capture")
        self.source_books = books
        self.heads, self.groups = books.shape[0], groups
        self.max_batch = max_batch
        self.books = books.permute(0, 3, 1, 2).float().contiguous()
        self.lut = torch.empty((max_batch, self.heads, groups, 64, 256), device=books.device, dtype=torch.float16)

    def __call__(self, query):
        batch = query.shape[0]
        if (
            not 1 <= batch <= self.max_batch
            or query.shape != (batch, self.heads, self.groups, 256)
            or query.dtype != torch.float32
            or not query.is_contiguous()
            or query.device != self.lut.device
        ):
            raise ValueError("Query LUT requires a contiguous B1 FP32 grouped query on the codebook device")
        _query_lut[(64, batch * self.heads * self.groups)](
            query,
            self.books,
            self.lut,
            self.groups,
            self.heads,
            compile_mode="simd",
            num_stages=1,
            enable_auto_bind_sub_block=False,
            enable_fp_fusion=False,
        )
        return self.lut[:batch]

    def workspace_bytes(self):
        return sum(t.numel() * t.element_size() for t in (self.books, self.lut))
