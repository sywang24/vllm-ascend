# SPDX-License-Identifier: Apache-2.0
"""Measured A5 compressed QK, fixed-split Softmax and Cube PV."""

import torch
from vllm.triton_utils import triton

from vllm_ascend.ops.triton.nova_cube.nova_decode_merge_a5 import merge_one_pass
from vllm_ascend.ops.triton.nova_cube.nova_split_kernels import _pv_prepared, _split_weights
from vllm_ascend.ops.triton.nova_cube.nova_triton_compressed import _merge_compressed
from vllm_ascend.ops.triton.nova_cube.nova_triton_pipeline import TritonPipelineDecode, _score_persistent


class TritonFixedSplitDecode(TritonPipelineDecode):
    def __init__(
        self,
        *args,
        feed_tokens=128,
        prepared_weights=True,
        score_heads=0,
        fuse_score_weights=False,
        merge_dims=0,
        score_stages=1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if (
            feed_tokens != 128
            or not prepared_weights
            or score_heads
            or fuse_score_weights
            or score_stages != 1
            or merge_dims not in (0, 64)
            or self.split_tokens != 1024
        ):
            raise ValueError("unsupported experimental Cube configuration")
        self.feed_tokens = feed_tokens
        self.alignment = max(self.alignment, feed_tokens)
        self.merge_dims = merge_dims
        self.weights = torch.empty_like(self.scores, dtype=torch.float16)
        self.bias = torch.empty_like(self.total)

    def _attend(self, lut, mean, views, table, starts, ends, scale):
        """Run lookup scoring, per-split Softmax/PV, then stable split reduction."""
        key, value, scales = views
        page = key.shape[-1]
        count = self.batch * self.heads * self.groups
        _score_persistent[(self.score_parts, count)](
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
            self.heads,
            table.stride(0),
            compile_mode="simd",
            num_stages=1,
            enable_auto_bind_sub_block=False,
        )
        _split_weights[(count, self.splits)](
            self.scores,
            scales,
            table,
            starts,
            ends,
            self.weights,
            self.bias,
            self.maximum,
            self.total,
            self.groups,
            self.score_capacity,
            page,
            scales.stride(0),
            self.splits,
            self.split_tokens,
            self.feed_tokens,
            self.alignment,
            self.heads,
            table.stride(0),
            compile_mode="simd",
            num_stages=1,
            enable_auto_bind_sub_block=False,
        )
        _pv_prepared[(self.batch * self.heads * self.splits, 256 // self.value_dims)](
            self.weights,
            value,
            table,
            starts,
            ends,
            self.bias,
            self.partial,
            self.groups,
            self.score_capacity,
            page,
            value.stride(0),
            self.splits,
            self.split_tokens,
            self.tile_tokens,
            self.feed_tokens,
            self.value_dims,
            self.alignment,
            self.heads,
            table.stride(0),
            **self.options,
        )
        if self.merge_dims and self.splits <= 128:
            merge_one_pass[(count, 256 // self.merge_dims)](
                self.partial,
                self.maximum,
                self.total,
                self.output,
                self.lse,
                self.splits,
                triton.next_power_of_2(self.splits),
                self.merge_dims,
                compile_mode="simd",
                enable_auto_bind_sub_block=False,
            )
        else:
            _merge_compressed[(count, 4)](
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
        return super().workspace_bytes() + sum(t.numel() * t.element_size() for t in (self.weights, self.bias))
