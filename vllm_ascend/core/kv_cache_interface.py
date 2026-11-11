# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from collections.abc import Iterable
from dataclasses import dataclass, replace

import torch
from typing_extensions import Self
from vllm.compilation import breakable_cudagraph
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager, SlidingWindowManager
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

from vllm_ascend import envs


def dquant_enabled(vllm_config: VllmConfig | None = None) -> bool:
    """Return whether the experimental D-Quant cache format was requested.

    The switch is intentionally environment based until vLLM grows a public
    cache-format option.  Keeping the default off prevents a compressed cache
    from being handed to an unmodified attention backend.
    """
    del vllm_config
    return envs.VLLM_ASCEND_ENABLE_DQUANT


def require_dquant_attention_backend(vllm_config: VllmConfig | None = None) -> None:
    """Reject unvalidated execution modes before cache allocation."""
    if vllm_config is None:
        raise RuntimeError("D-Quant configuration is missing; KV cache allocation has not started.")
    breakable_piecewise = False
    full_decode = (
        not vllm_config.model_config.enforce_eager
        and vllm_config.compilation_config.cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY
    )
    if not vllm_config.model_config.enforce_eager:
        breakable_piecewise = (
            breakable_cudagraph.is_breakable_cudagraph_enabled()
            and vllm_config.compilation_config.cudagraph_mode == CUDAGraphMode.PIECEWISE
        )
    if not vllm_config.model_config.enforce_eager and not (breakable_piecewise or full_decode):
        raise RuntimeError(
            "D-Quant requires eager, breakable PIECEWISE, or FULL_DECODE_ONLY execution; "
            "KV cache allocation has not started."
        )
    parallel = vllm_config.parallel_config
    if full_decode and (
        not envs.VLLM_ASCEND_DQUANT_FUSED_ATTENTION
        or envs.VLLM_ASCEND_DQUANT_SYMBOL_CACHE
        or envs.VLLM_ASCEND_DQUANT_KEY_WINDOW != 64
        or envs.VLLM_ASCEND_DQUANT_RECENT_WINDOW != 128
        or vllm_config.model_config.get_num_kv_heads(parallel) != 2
        or vllm_config.model_config.get_head_size() != 256
    ):
        raise RuntimeError(
            "D-Quant FULL_DECODE_ONLY requires fused rANS, two KV heads, head_size=256, recent=128, key_window=64"
        )
    # Tensor parallelism is supported: each rank owns a contiguous shard of the
    # KV heads, so the per-token record shrinks with the local stream length.
    # PP/PCP/DCP still replicate scheduler-visible token ids and are rejected.
    if any(
        getattr(parallel, name, 1) != 1
        for name in (
            "pipeline_parallel_size",
            "prefill_context_parallel_size",
            "decode_context_parallel_size",
        )
    ):
        raise RuntimeError("The D-Quant eager bridge supports TP but not PP/PCP/DCP.")
    if vllm_config.kv_transfer_config is not None or vllm_config.speculative_config is not None:
        raise RuntimeError("D-Quant does not yet support KV transfer or speculative decoding.")
    if vllm_config.quant_config is not None or vllm_config.cache_config.cache_dtype != "auto":
        raise RuntimeError("D-Quant requires unquantized model weights and cache_dtype=auto.")
    if envs.VLLM_ASCEND_DQUANT_RECENT_WINDOW or envs.VLLM_ASCEND_DQUANT_KEY_WINDOW:
        if vllm_config.cache_config.enable_prefix_caching:
            raise RuntimeError("D-Quant window state currently requires enable_prefix_caching=False.")
        if getattr(parallel, "enable_dbo", False):
            raise RuntimeError("D-Quant window state does not support dual batch overlap.")


# Reference record sizes from the paper: a 1024-value stream uses a 331-byte Key
# container (2.59 bit/value) and a 231-byte Value container (1.81 bit/value).
# With tensor parallelism the local stream shrinks, so the fixed container must
# shrink proportionally to keep the same effective bit rate.
_REFERENCE_STREAM_LENGTH = 1024
_KEY_CONTAINER_BYTES_PER_VALUE = 331 / _REFERENCE_STREAM_LENGTH
_VALUE_CONTAINER_BYTES_PER_VALUE = 231 / _REFERENCE_STREAM_LENGTH


def dquant_container_bytes(stream_length: int) -> tuple[int, int]:
    """Scale the paper container sizes to one rank-local KV head shard."""
    if stream_length <= 0:
        raise ValueError("D-Quant stream length must be positive")
    key = max(8, int(round(stream_length * _KEY_CONTAINER_BYTES_PER_VALUE)))
    value = max(8, int(round(stream_length * _VALUE_CONTAINER_BYTES_PER_VALUE)))
    return key | 1, value | 1


@dataclass(frozen=True, kw_only=True)
class DQuantAttentionSpec(FullAttentionSpec):
    """Paged D-Quant layout for one dense full-attention layer.

    One scheduler token maps to one K and one V fixed-size rANS record.  The
    record contains an even-padded payload and two FP16 values (scale/offset),
    so ``real_page_size_bytes`` is the compact physical page size.  The class
    deliberately remains a ``FullAttentionSpec`` subclass: scheduling,
    prefix-cache hashing and block allocation keep their normal semantics while
    the Ascend worker can select a D-Quant-aware allocator/backend.
    """

    dquant_key_container_bytes: int = 331
    dquant_value_container_bytes: int = 231
    dquant_key_levels: int = 8
    dquant_value_levels: int = 6
    dquant_alpha: float = 1.4
    dquant_recent_window: int = 0
    dquant_key_window: int = 0
    dquant_stream_length: int | None = None
    dquant_symbol_cache: bool = False

    def __post_init__(self):
        super().__post_init__()
        for label, levels in (("key", self.dquant_key_levels), ("value", self.dquant_value_levels)):
            if not isinstance(levels, int) or not 2 <= levels <= 256:
                raise ValueError(f"D-Quant {label} levels must be an integer between 2 and 256")
        for label, size in (
            ("key", self.dquant_key_container_bytes),
            ("value", self.dquant_value_container_bytes),
        ):
            if not isinstance(size, int) or size < 4:
                raise ValueError(f"D-Quant {label} payload must contain the 4-byte rANS state")
        if not math.isfinite(self.dquant_alpha) or self.dquant_alpha <= 0:
            raise ValueError("D-Quant alpha must be finite and positive")
        if self.dquant_recent_window < 0 or self.dquant_key_window < 0:
            raise ValueError("D-Quant windows must be nonnegative")
        if self.dquant_key_window and self.block_size % self.dquant_key_window:
            raise ValueError("D-Quant key window must divide the physical block size")
        if self.dquant_symbol_cache and (
            max(self.dquant_key_levels, self.dquant_value_levels) > 16
            or self.stream_length % 2
            or self.dquant_key_window % 16
            or not self.dquant_key_window
            or self.mean_stripe_bytes % 4
        ):
            raise ValueError(
                "Symbol cache needs <=16 levels, even streams, 16-aligned windows and four-byte mean stripes"
            )

    @property
    def uses_window_state(self) -> bool:
        return bool(self.dquant_recent_window or self.dquant_key_window)

    @property
    def mean_stripe_bytes(self) -> int:
        if not self.dquant_key_window:
            return 0
        size = self.stream_length * get_dtype_size(self.dtype)
        return 2 * cdiv(size, 2 * self.dquant_key_window)

    @property
    def recent_capacity(self) -> int:
        return self.dquant_recent_window + max(1, self.dquant_key_window) - 1

    @property
    def recent_state_bytes(self) -> int:
        return (self.recent_capacity + 1) * (2 * self.stream_length * get_dtype_size(self.dtype) + 8) + 4

    @property
    def stream_length(self) -> int:
        return self.dquant_stream_length or (self.num_kv_heads * self.head_size)

    @property
    def key_record_bytes(self) -> int:
        return ((self.dquant_key_container_bytes + 1) & ~1) + 4 + self.mean_stripe_bytes + self.symbol_bytes

    @property
    def value_record_bytes(self) -> int:
        return ((self.dquant_value_container_bytes + 1) & ~1) + 4 + self.symbol_bytes

    @property
    def symbol_bytes(self) -> int:
        return self.stream_length // 2 if self.dquant_symbol_cache else 0

    @property
    def key_page_size_bytes(self) -> int:
        return self.block_size * self.key_record_bytes

    @property
    def value_page_size_bytes(self) -> int:
        return self.block_size * self.value_record_bytes

    @property
    def real_page_size_bytes(self) -> int:
        return self.key_page_size_bytes + self.value_page_size_bytes

    @property
    def page_size_bytes(self) -> int:
        return max(self.real_page_size_bytes, self.page_size_padded or 0)

    @classmethod
    def from_spec(cls, spec: FullAttentionSpec) -> "DQuantAttentionSpec":
        """Convert a regular full-attention spec without changing its block size."""
        if spec.head_size_v not in (None, spec.head_size):
            raise ValueError("D-Quant KV records currently require equal key and value head sizes.")
        stream_length = spec.num_kv_heads * spec.head_size
        key_container_bytes, value_container_bytes = dquant_container_bytes(stream_length)
        if envs.VLLM_ASCEND_DQUANT_KEY_BYTES < 0 or envs.VLLM_ASCEND_DQUANT_VALUE_BYTES < 0:
            raise ValueError("D-Quant payload overrides must be nonnegative; zero selects the default")
        if envs.VLLM_ASCEND_DQUANT_KEY_BYTES:
            key_container_bytes = envs.VLLM_ASCEND_DQUANT_KEY_BYTES | 1
        if envs.VLLM_ASCEND_DQUANT_VALUE_BYTES:
            value_container_bytes = envs.VLLM_ASCEND_DQUANT_VALUE_BYTES | 1
        return cls(
            block_size=spec.block_size,
            num_kv_heads=spec.num_kv_heads,
            head_size=spec.head_size,
            dtype=spec.dtype,
            kv_quant_mode=spec.kv_quant_mode,
            # The base spec may carry a dense-page override. D-Quant's physical
            # allocation is determined by its fixed record sizes instead.
            page_size_padded=None,
            head_size_v=spec.head_size_v,
            sliding_window=spec.sliding_window,
            attention_chunk_size=spec.attention_chunk_size,
            non_causal=spec.non_causal,
            dquant_stream_length=stream_length,
            dquant_key_container_bytes=key_container_bytes,
            dquant_value_container_bytes=value_container_bytes,
            dquant_key_levels=envs.VLLM_ASCEND_DQUANT_KEY_LEVELS,
            dquant_value_levels=envs.VLLM_ASCEND_DQUANT_VALUE_LEVELS,
            dquant_alpha=envs.VLLM_ASCEND_DQUANT_ALPHA,
            dquant_recent_window=envs.VLLM_ASCEND_DQUANT_RECENT_WINDOW,
            dquant_key_window=envs.VLLM_ASCEND_DQUANT_KEY_WINDOW,
            dquant_symbol_cache=envs.VLLM_ASCEND_DQUANT_SYMBOL_CACHE,
        )

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        assert all(isinstance(spec, DQuantAttentionSpec) for spec in specs)
        layout = {
            (
                spec.dquant_key_container_bytes,
                spec.dquant_value_container_bytes,
                spec.dquant_key_levels,
                spec.dquant_value_levels,
                spec.dquant_alpha,
                spec.dquant_recent_window,
                spec.dquant_key_window,
                spec.dquant_stream_length,
                spec.dquant_symbol_cache,
            )
            for spec in specs
        }
        assert len(layout) == 1, "D-Quant layers in one KV cache group must share one record layout."
        merged = super().merge(specs)
        first = specs[0]
        return replace(
            merged,
            dquant_key_container_bytes=first.dquant_key_container_bytes,
            dquant_value_container_bytes=first.dquant_value_container_bytes,
            dquant_key_levels=first.dquant_key_levels,
            dquant_value_levels=first.dquant_value_levels,
            dquant_alpha=first.dquant_alpha,
            dquant_recent_window=first.dquant_recent_window,
            dquant_key_window=first.dquant_key_window,
            dquant_stream_length=first.dquant_stream_length,
            dquant_symbol_cache=first.dquant_symbol_cache,
        )

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        max_model_len = vllm_config.model_config.max_model_len
        return ((max_model_len + self.block_size - 1) // self.block_size) * self.page_size_bytes


def get_kv_cache_compression_ratio(kv_cache_spec: KVCacheSpec) -> int:
    """Return the MLA compression ratio across vLLM cache-spec APIs."""
    return kv_cache_spec.tokens_per_state


def get_storage_block_size(kv_cache_spec: KVCacheSpec) -> int:
    """Return the physical token rows represented by one scheduler block."""
    if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
        storage_block_sizes = {get_storage_block_size(spec) for spec in kv_cache_spec.kv_cache_specs.values()}
        assert len(storage_block_sizes) == 1, "All specs in one KV cache group must use the same storage block size."
        return storage_block_sizes.pop()
    # vLLM #53906 added an optional MLA storage-view override. It is not
    # Ascend's derived number of physical rows per logical block.
    if isinstance(kv_cache_spec, AscendMLAAttentionSpec):
        return kv_cache_spec.block_size // kv_cache_spec.tokens_per_state
    if isinstance(kv_cache_spec, MLAAttentionSpec):
        storage_block_size = kv_cache_spec.storage_block_size
        return kv_cache_spec.block_size if storage_block_size is None else storage_block_size
    return getattr(kv_cache_spec, "storage_block_size", kv_cache_spec.block_size)


def is_circular_kv_cache_spec(kv_cache_spec: KVCacheSpec) -> bool:
    """Resolve the fixed-block addressing capability through grouped specs."""
    if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
        specs = tuple(kv_cache_spec.kv_cache_specs.values())
        return bool(specs) and all(is_circular_kv_cache_spec(spec) for spec in specs)
    return isinstance(kv_cache_spec, CircularBufferSpec) or getattr(kv_cache_spec, "is_circular", False)


def is_prefix_cacheable(kv_cache_spec: KVCacheSpec) -> bool:
    """Resolve cacheability through uniform wrappers.

    ``prefix_cacheable`` is the upstream API on the lanes that define it; the
    release lane predates it, so an unset attribute means cacheable.
    """
    if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
        return all(is_prefix_cacheable(spec) for spec in kv_cache_spec.kv_cache_specs.values())
    return bool(getattr(kv_cache_spec, "prefix_cacheable", True)) and bool(
        getattr(kv_cache_spec, "participates_in_prefix_caching", True)
    )


def requires_padded_page_layout(kv_cache_specs: Iterable[KVCacheSpec]) -> bool:
    """Whether state caches advance by a padded page size shared with attention.

    Ascend mixed pools can bind one physical page to both an attention cache and a
    recurrent state cache. The attention side of such a pool is addressed by an explicit
    physical block stride (``indexes_kv_by_block_stride``) and the state caches are
    padded to that same page size, so every view over the shared page has to advance by
    the padded page size: otherwise a state update for one scheduler block ID would land
    in the pages owned by another block ID.

    The capability belongs to the pool rather than to a single spec, because hybrid
    models also pad Mamba pages through ``cache_config.mamba_page_size_padded`` while
    keeping the packed contiguous state layout, so padding alone does not require this
    layout.
    """
    specs = list(kv_cache_specs)
    state_specs = [spec for spec in specs if isinstance(spec, MambaSpec)]
    if not state_specs:
        return False
    if not any(getattr(spec, "page_size_padded", None) is not None for spec in state_specs):
        return False
    return any(getattr(spec, "indexes_kv_by_block_stride", False) for spec in specs)


def supports_component_major_mla_pd(vllm_config: VllmConfig) -> bool:
    """Whether PD transfer can preserve a component-major MLA cache layout.

    A component-major MLA page packs NoPE and RoPE into one physical slot:
    ``[all NoPE][all RoPE][padding]``.  MooncakeV2 pull connectors transfer
    that page as one raw block when the published base address, block length,
    and block stride describe the complete slot.  Other connectors still use
    the legacy separate NoPE/RoPE protocol.

    Without PD, local layout selection is unrestricted. With supported PD,
    both endpoints must agree on the page layout: raw transfers carry no
    metadata distinguishing component-major from token-fused FlashMLA pages.
    This helper identifies connector support, not cross-endpoint agreement.
    """
    kv_transfer_config = vllm_config.kv_transfer_config
    if kv_transfer_config is None:
        return True

    # Connectors loaded through a custom module path are not identified by the
    # registry name, so conservatively fall back to the legacy layout.
    if getattr(kv_transfer_config, "kv_connector_module_path", None) is not None:
        return False

    return getattr(kv_transfer_config, "kv_connector", None) in {
        "MooncakeConnectorV2",
        "MooncakePullConnector",
    }


@dataclass(frozen=True, kw_only=True)
class AscendMLAAttentionSpec(MLAAttentionSpec):
    """MLA cache spec with Ascend-specific layout metadata.

    For SFA, this spec describes only the main MLA cache. The indexer K
    tensor, its quantization scale, and DCP replication are described by a
    separate :class:`AscendSFAIndexerCacheSpec`.
    """

    # Per-rank query heads used by FlashMLA layout selection. Auxiliary MLA
    # caches without query heads use the one-latent-head cache contract.
    num_heads: int = 1
    scale_dim: int = 0
    scale_dtype: torch.dtype = torch.int8
    # Sparse C8 changes the main cache into one packed byte tensor. Keep that
    # main-cache property here; indexer-specific C8 properties belong to the
    # indexer spec.
    cache_sparse_sfa_c8: bool = False
    cache_sparse_sfa_turboquant: bool = False
    store_on_host: bool = False
    # Ascend kernels consume padded pages through an explicit physical block
    # stride. vLLM main removed this field from AttentionSpec, but it remains
    # part of the Ascend runner/backend contract.
    indexes_kv_by_block_stride: bool = False

    @property
    def uses_packed_sfa_main_cache(self) -> bool:
        """Whether the SFA main cache is stored in one packed tensor."""
        return self.cache_sparse_sfa_c8 or self.cache_sparse_sfa_turboquant

    @property
    def supports_single_raw_backing(self) -> bool:
        """Whether this geometry can be exposed from one raw MLA backing."""
        return (
            get_kv_cache_compression_ratio(self) == 1
            and self.model_version is None
            and not self.indexes_kv_by_block_stride
            and not self.cache_sparse_sfa_c8
        )

    @property
    def real_page_size_bytes(self) -> int:
        return (
            get_storage_block_size(self)
            * self.num_kv_heads
            * (self.head_size * get_dtype_size(self.dtype) + self.scale_dim * get_dtype_size(self.scale_dtype))
        )

    @property
    def unpadded_page_size_bytes(self) -> int:
        return self.real_page_size_bytes

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        # Uniformity probes may include indexer specs, which share the upstream
        # MLA base class but do not have the main cache's layout metadata.
        assert all(isinstance(spec, AscendMLAAttentionSpec) for spec in specs), (
            "All attention layers in the same KV cache group must be AscendMLAAttentionSpec."
        )
        ascend_layouts = {
            (
                spec.scale_dim,
                spec.num_heads,
                spec.scale_dtype,
                spec.cache_sparse_sfa_c8,
                spec.cache_sparse_sfa_turboquant,
                spec.store_on_host,
                spec.alignment,
                get_kv_cache_compression_ratio(spec),
                spec.indexes_kv_by_block_stride,
            )
            for spec in specs
        }
        assert len(ascend_layouts) == 1, (
            "All attention layers in the same KV cache group must use the same Ascend KV cache layout."
        )
        non_causal_multi_token_decode_set = set(spec.non_causal_multi_token_decode for spec in specs)
        assert len(non_causal_multi_token_decode_set) == 1, (
            "Causal target layers and non-causal multi-token draft layers must use separate KV cache groups."
        )
        first_spec = specs[0]
        merged = super().merge(specs)
        return replace(
            merged,
            scale_dim=first_spec.scale_dim,
            num_heads=first_spec.num_heads,
            scale_dtype=first_spec.scale_dtype,
            alignment=first_spec.alignment,
            cache_sparse_sfa_c8=first_spec.cache_sparse_sfa_c8,
            cache_sparse_sfa_turboquant=first_spec.cache_sparse_sfa_turboquant,
            store_on_host=first_spec.store_on_host,
            indexes_kv_by_block_stride=first_spec.indexes_kv_by_block_stride,
        )

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        max_model_len = vllm_config.model_config.max_model_len
        dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        # Note(hc): each dcp rank only need save
        # (max_model_len//dcp_world_size) tokens locally.
        if dcp_world_size > 1:
            max_model_len = cdiv(max_model_len, dcp_world_size)
        return cdiv(max_model_len, self.block_size) * self.page_size_bytes


@dataclass(frozen=True, kw_only=True)
class AscendSFAIndexerCacheSpec(MLAAttentionSpec):
    """KV cache spec for SFA indexer K/scale cache.

    The scheduler should treat this as a full-attention-compatible cache so it
    can share block ids with the MLA cache in the same UniformType group. The
    model runner still allocates it as an independent physical cache tensor.
    """

    scale_dim: int = 0
    scale_dtype: torch.dtype = torch.int8
    cache_sparse_li_c8: bool = False
    cache_sparse_li_c4: bool = False
    cache_dtype_str: str | None = None
    sfa_dcp_replicated_indexer_size: int = 1

    @property
    def page_size_bytes(self) -> int:
        return self.real_page_size_bytes

    @property
    def real_page_size_bytes(self) -> int:
        num_heads_per_page = self.block_size * self.num_kv_heads
        return (
            self.sfa_dcp_replicated_indexer_size
            * num_heads_per_page
            * (self.head_size * get_dtype_size(self.dtype) + self.scale_dim * get_dtype_size(self.scale_dtype))
        )

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        assert all(isinstance(spec, AscendSFAIndexerCacheSpec) for spec in specs), (
            "All attention layers in the same KV cache group must be AscendSFAIndexerCacheSpec."
        )
        cache_dtype_str_set = set(spec.cache_dtype_str for spec in specs)
        dtype_set = set(spec.dtype for spec in specs)
        scale_dim_set = set(spec.scale_dim for spec in specs)
        scale_dtype_set = set(spec.scale_dtype for spec in specs)
        cache_sparse_li_c8_set = set(spec.cache_sparse_li_c8 for spec in specs)
        cache_sparse_li_c4_set = set(spec.cache_sparse_li_c4 for spec in specs)
        sfa_dcp_replicated_indexer_size_set = set(spec.sfa_dcp_replicated_indexer_size for spec in specs)
        assert (
            len(cache_dtype_str_set) == 1
            and len(dtype_set) == 1
            and len(scale_dim_set) == 1
            and len(scale_dtype_set) == 1
            and len(cache_sparse_li_c8_set) == 1
            and len(cache_sparse_li_c4_set) == 1
            and len(sfa_dcp_replicated_indexer_size_set) == 1
        ), (
            "All SFA indexer cache layers in the same KV cache group must use "
            "the same dtype, scale layout, quantization method, LI quant flags "
            "and DCP replication size."
        )
        return cls(
            block_size=specs[0].block_size,
            num_kv_heads=specs[0].num_kv_heads,
            head_size=specs[0].head_size,
            dtype=dtype_set.pop(),
            cache_dtype_str=cache_dtype_str_set.pop(),
            scale_dim=scale_dim_set.pop(),
            scale_dtype=scale_dtype_set.pop(),
            cache_sparse_li_c8=cache_sparse_li_c8_set.pop(),
            cache_sparse_li_c4=cache_sparse_li_c4_set.pop(),
            sfa_dcp_replicated_indexer_size=sfa_dcp_replicated_indexer_size_set.pop(),
        )


@dataclass(frozen=True, kw_only=True)
class AscendSlidingWindowMLASpec(SlidingWindowMLASpec):
    """Sliding window attention with MLA cache format."""

    cache_dtype_str: str | None = None
    # DeepseekV4-only: see MLAAttentionSpec.model_version.
    alignment: int | None = None  # Default to None for no padding.
    compress_ratio: int = 1
    model_version: str | None = None
    indexes_kv_by_block_stride: bool = False

    def __post_init__(self):
        pass

    @property
    def storage_block_size(self) -> int:
        return self.block_size // self.compress_ratio

    @property
    def real_page_size_bytes(self) -> int:
        return self.storage_block_size * self.num_kv_heads * self.head_size * get_dtype_size(self.dtype)

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        assert all(isinstance(spec, AscendSlidingWindowMLASpec) for spec in specs), (
            "All attention layers in the same KV cache group must be AscendSlidingWindowMLASpec."
        )
        cache_dtype_str_set = set(spec.cache_dtype_str for spec in specs)
        compress_ratio_set = set(spec.compress_ratio for spec in specs)
        model_version_set = set(spec.model_version for spec in specs)
        sliding_window_set = set(spec.sliding_window for spec in specs)
        assert (
            len(cache_dtype_str_set) == 1
            and len(compress_ratio_set) == 1
            and len(model_version_set) == 1
            and len(sliding_window_set) == 1
        ), (
            "All attention layers in the same KV cache group must use the same "
            "quantization method, compress ratio, model version and sliding "
            "window size."
        )
        return cls(
            block_size=specs[0].block_size,
            num_kv_heads=specs[0].num_kv_heads,
            head_size=specs[0].head_size,
            dtype=specs[0].dtype,
            page_size_padded=specs[0].page_size_padded,
            sliding_window=sliding_window_set.pop(),
            cache_dtype_str=cache_dtype_str_set.pop(),
            compress_ratio=compress_ratio_set.pop(),
            model_version=model_version_set.pop(),
        )


@dataclass(frozen=True, kw_only=True)
class AscendIndexerKPoolTailSpec(SlidingWindowSpec):
    """One fixed FP32 ``[ring_capacity, 2 * head_size]`` page per request.

    ``block_size`` is the physical ring capacity; ``compress_ratio`` defines
    pool boundaries independently. The dedicated manager never slides or grows.
    """

    compress_ratio: int
    model_version: str = "glm5_next"
    cache_role: str = "indexer_tail"
    indexes_kv_by_block_stride: bool = True

    @property
    def is_circular(self) -> bool:
        return True

    @property
    def prefix_cacheable(self) -> bool:
        return False

    @property
    def unpadded_page_size_bytes(self) -> int:
        # The tail stores a fixed K/gate pair; the parent handles page padding.
        return 2 * self.block_size * self.head_size * get_dtype_size(self.dtype)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.dtype != torch.float32:
            raise ValueError(f"Indexer K-pool tail must use FP32, got {self.dtype}.")
        if self.num_kv_heads != 1 or self.head_size <= 0:
            raise ValueError("Indexer K-pool tail requires one K/gate pair with positive head size.")
        if self.compress_ratio <= 1 or self.block_size < self.compress_ratio:
            raise ValueError("Tail ring capacity must be at least the compression ratio, which must exceed one.")
        if self.sliding_window != self.compress_ratio:
            raise ValueError("Tail sliding_window describes the logical pool size and must equal compress_ratio.")

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        assert all(isinstance(spec, cls) for spec in specs)
        return super().merge(specs)

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        del vllm_config
        return self.page_size_bytes

    def max_admission_blocks_per_request(self, max_in_flight_tokens: int, max_model_len: int) -> int:
        return 1

    def max_num_blocks_per_req(self, vllm_config: VllmConfig, max_len: int) -> int:
        return 1

    def is_uniform_with_collection(self, kv_cache_specs: dict[str, KVCacheSpec]) -> bool:
        return all(
            isinstance(spec, AscendIndexerKPoolTailSpec)
            and spec.block_size == self.block_size
            and spec.compress_ratio == self.compress_ratio
            for spec in kv_cache_specs.values()
        )


@dataclass(frozen=True, kw_only=True)
class AscendCircularBufferSpec(CircularBufferSpec):
    """Circular state whose physical pages participate in Ascend KV zeroing."""

    def is_uniform_with_collection(self, kv_cache_specs: dict[str, KVCacheSpec]) -> bool:
        return all(isinstance(spec, AscendCircularBufferSpec) for spec in kv_cache_specs.values())


def register_ascend_kv_cache_specs() -> None:
    KVCacheSpecRegistry.register(
        kvcache_spec_cls=DQuantAttentionSpec,
        manager_class=FullAttentionManager,
        uniform_type_base_spec=FullAttentionSpec,
    )
    from vllm_ascend.core.single_type_kv_cache_manager import AscendCircularBufferManager
    from vllm_ascend.models.glm5next.kv_cache import KpoolTailManager

    KVCacheSpecRegistry.register(
        kvcache_spec_cls=AscendCircularBufferSpec,
        manager_class=AscendCircularBufferManager,
        uniform_type_base_spec=AscendCircularBufferSpec,
    )

    KVCacheSpecRegistry.register(
        kvcache_spec_cls=AscendMLAAttentionSpec,
        manager_class=FullAttentionManager,
        uniform_type_base_spec=FullAttentionSpec,
    )
    KVCacheSpecRegistry.register(
        kvcache_spec_cls=AscendSFAIndexerCacheSpec,
        manager_class=FullAttentionManager,
        uniform_type_base_spec=FullAttentionSpec,
    )
    KVCacheSpecRegistry.register(
        kvcache_spec_cls=AscendSlidingWindowMLASpec,
        manager_class=SlidingWindowManager,
        uniform_type_base_spec=SlidingWindowMLASpec,
    )

    KVCacheSpecRegistry.register(
        kvcache_spec_cls=AscendIndexerKPoolTailSpec,
        manager_class=KpoolTailManager,
        uniform_type_base_spec=AscendIndexerKPoolTailSpec,
    )
