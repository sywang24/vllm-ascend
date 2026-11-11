import os
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import torch
from vllm.config import CUDAGraphMode
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_ascend.core.kv_cache_interface import (
    DQuantAttentionSpec,
    dquant_enabled,
    require_dquant_attention_backend,
)


class DQuantSpecTests(unittest.TestCase):
    def test_compact_per_token_records(self):
        spec = DQuantAttentionSpec.from_spec(
            FullAttentionSpec(
                block_size=128,
                num_kv_heads=8,
                head_size=128,
                dtype=torch.float16,
            )
        )

        self.assertEqual(spec.stream_length, 1024)
        self.assertEqual(spec.key_page_size_bytes, 128 * 336)
        self.assertEqual(spec.value_page_size_bytes, 128 * 236)
        self.assertEqual(spec.real_page_size_bytes, 128 * 572)
        self.assertEqual(spec.page_size_bytes, spec.real_page_size_bytes)

    def test_preserves_scheduler_shape(self):
        original = FullAttentionSpec(
            block_size=64,
            num_kv_heads=4,
            head_size=128,
            dtype=torch.bfloat16,
            sliding_window=4096,
        )
        spec = DQuantAttentionSpec.from_spec(original)

        self.assertEqual(spec.block_size, original.block_size)
        self.assertEqual(spec.num_kv_heads, original.num_kv_heads)
        self.assertEqual(spec.head_size, original.head_size)
        self.assertEqual(spec.sliding_window, original.sliding_window)

    def test_compact_page_does_not_inherit_dense_padding(self):
        original = FullAttentionSpec(
            block_size=128,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.float16,
            page_size_padded=300_000,
        )
        spec = DQuantAttentionSpec.from_spec(original)

        self.assertEqual(spec.page_size_bytes, spec.real_page_size_bytes)
        self.assertEqual(spec.page_size_bytes, 128 * (336 + 236))

    def test_rejects_different_value_head_size(self):
        original = FullAttentionSpec(
            block_size=128,
            num_kv_heads=8,
            head_size=128,
            head_size_v=64,
            dtype=torch.float16,
        )
        with self.assertRaisesRegex(ValueError, "equal key and value head sizes"):
            DQuantAttentionSpec.from_spec(original)

    def test_switch_is_opt_in(self):
        old_value = os.environ.pop("VLLM_ASCEND_ENABLE_DQUANT", None)
        try:
            self.assertFalse(dquant_enabled())
            os.environ["VLLM_ASCEND_ENABLE_DQUANT"] = "true"
            self.assertTrue(dquant_enabled())
        finally:
            if old_value is None:
                os.environ.pop("VLLM_ASCEND_ENABLE_DQUANT", None)
            else:
                os.environ["VLLM_ASCEND_ENABLE_DQUANT"] = old_value

    def test_backend_requirement_fails_before_cache_allocation(self):
        with self.assertRaisesRegex(RuntimeError, "allocation has not started"):
            require_dquant_attention_backend()

    def test_backend_requirement_accepts_tp_but_rejects_context_parallelism(self):
        config = SimpleNamespace(
            model_config=SimpleNamespace(enforce_eager=True),
            parallel_config=SimpleNamespace(tensor_parallel_size=1),
            kv_transfer_config=None,
            speculative_config=None,
            quant_config=None,
            cache_config=SimpleNamespace(cache_dtype="auto", enable_prefix_caching=False),
        )
        require_dquant_attention_backend(config)
        config.parallel_config.tensor_parallel_size = 2
        require_dquant_attention_backend(config)
        config.parallel_config.decode_context_parallel_size = 2
        with self.assertRaisesRegex(RuntimeError, "PP/PCP/DCP"):
            require_dquant_attention_backend(config)

    def test_backend_requirement_accepts_breakable_piecewise_graph(self):
        config = SimpleNamespace(
            model_config=SimpleNamespace(enforce_eager=False),
            parallel_config=SimpleNamespace(tensor_parallel_size=2),
            compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.PIECEWISE),
            kv_transfer_config=None,
            speculative_config=None,
            quant_config=None,
            cache_config=SimpleNamespace(cache_dtype="auto", enable_prefix_caching=False),
        )
        with (
            patch(
                "vllm_ascend.core.kv_cache_interface.breakable_cudagraph.is_breakable_cudagraph_enabled",
                return_value=True,
            ),
            patch("vllm_ascend.envs.VLLM_ASCEND_DQUANT_RECENT_WINDOW", 0),
            patch("vllm_ascend.envs.VLLM_ASCEND_DQUANT_KEY_WINDOW", 0),
        ):
            require_dquant_attention_backend(config)

    def test_backend_requirement_rejects_monolithic_graph(self):
        config = SimpleNamespace(
            model_config=SimpleNamespace(enforce_eager=False),
            parallel_config=SimpleNamespace(tensor_parallel_size=2),
            compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY),
            kv_transfer_config=None,
            speculative_config=None,
            quant_config=None,
            cache_config=SimpleNamespace(cache_dtype="auto"),
        )
        for mode in (
            CUDAGraphMode.NONE,
            CUDAGraphMode.FULL,
            CUDAGraphMode.FULL_AND_PIECEWISE,
        ):
            config.compilation_config.cudagraph_mode = mode
            with (
                self.subTest(mode=mode),
                patch(
                    "vllm_ascend.core.kv_cache_interface.breakable_cudagraph.is_breakable_cudagraph_enabled",
                    return_value=True,
                ),
                self.assertRaisesRegex(RuntimeError, "breakable PIECEWISE"),
            ):
                require_dquant_attention_backend(config)
        config.compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
        with (
            patch(
                "vllm_ascend.core.kv_cache_interface.breakable_cudagraph.is_breakable_cudagraph_enabled",
                return_value=False,
            ),
            self.assertRaisesRegex(RuntimeError, "breakable PIECEWISE"),
        ):
            require_dquant_attention_backend(config)

    def test_merge_preserves_nondefault_compression_layout(self):
        spec = DQuantAttentionSpec(
            block_size=128,
            num_kv_heads=4,
            head_size=256,
            dtype=torch.bfloat16,
            dquant_key_container_bytes=355,
            dquant_value_container_bytes=255,
            dquant_key_levels=7,
            dquant_value_levels=5,
            dquant_stream_length=1024,
        )
        merged = DQuantAttentionSpec.merge([spec, spec])
        self.assertEqual(merged, spec)
        self.assertEqual(merged.page_size_bytes, 128 * (360 + 260))

    def test_scheduler_padding_is_accounted_for(self):
        spec = DQuantAttentionSpec(block_size=128, num_kv_heads=4, head_size=256, dtype=torch.bfloat16)
        padded = replace(spec, page_size_padded=100_000)
        self.assertEqual(padded.page_size_bytes, 100_000)
        self.assertEqual(padded.real_page_size_bytes, 128 * 572)

    def test_centering_means_and_recent_memory_are_accounted_for(self):
        spec = DQuantAttentionSpec(
            block_size=128,
            num_kv_heads=4,
            head_size=256,
            dtype=torch.bfloat16,
            dquant_recent_window=64,
            dquant_key_window=64,
        )
        self.assertEqual(spec.mean_stripe_bytes, 32)
        self.assertEqual(spec.key_record_bytes, 368)
        self.assertEqual(spec.real_page_size_bytes, 128 * 604)
        self.assertEqual(spec.recent_capacity, 127)
        self.assertEqual(spec.recent_state_bytes, 128 * (4096 + 8) + 4)
        self.assertEqual(DQuantAttentionSpec.merge([spec, spec]), spec)

    def test_window_state_rejects_prefix_caching(self):
        config = SimpleNamespace(
            model_config=SimpleNamespace(enforce_eager=True),
            parallel_config=SimpleNamespace(tensor_parallel_size=1),
            kv_transfer_config=None,
            speculative_config=None,
            quant_config=None,
            cache_config=SimpleNamespace(cache_dtype="auto", enable_prefix_caching=True),
        )
        from unittest.mock import patch

        with (
            patch("vllm_ascend.envs.VLLM_ASCEND_DQUANT_RECENT_WINDOW", 64),
            self.assertRaisesRegex(RuntimeError, "prefix_caching"),
        ):
            require_dquant_attention_backend(config)

    def test_full_decode_only_accepts_validated_fused_layout(self):
        config = SimpleNamespace(
            model_config=SimpleNamespace(
                enforce_eager=False,
                get_num_kv_heads=lambda parallel: 2,
                get_head_size=lambda: 256,
            ),
            parallel_config=SimpleNamespace(tensor_parallel_size=2),
            compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY),
            kv_transfer_config=None,
            speculative_config=None,
            quant_config=None,
            cache_config=SimpleNamespace(cache_dtype="auto", enable_prefix_caching=False),
        )
        with (
            patch("vllm_ascend.envs.VLLM_ASCEND_DQUANT_FUSED_ATTENTION", True),
            patch("vllm_ascend.envs.VLLM_ASCEND_DQUANT_SYMBOL_CACHE", False),
            patch("vllm_ascend.envs.VLLM_ASCEND_DQUANT_KEY_WINDOW", 64),
            patch("vllm_ascend.envs.VLLM_ASCEND_DQUANT_RECENT_WINDOW", 128),
        ):
            require_dquant_attention_backend(config)
            config.model_config.get_num_kv_heads = lambda parallel: 4
            with self.assertRaisesRegex(RuntimeError, "two KV heads"):
                require_dquant_attention_backend(config)
            config.model_config.get_num_kv_heads = lambda parallel: 2
            with (
                patch("vllm_ascend.envs.VLLM_ASCEND_DQUANT_FUSED_ATTENTION", False),
                self.assertRaisesRegex(RuntimeError, "fused rANS"),
            ):
                require_dquant_attention_backend(config)


if __name__ == "__main__":
    unittest.main()
