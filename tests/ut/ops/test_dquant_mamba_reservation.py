"""Regression for false external-load reservations in hybrid D-Quant runs."""

from unittest.mock import patch

import pytest
import torch
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_ascend.patch.platform.patch_mamba_manager import AscendMambaManager


@pytest.mark.parametrize(
    "local,total,main,tracked,expected",
    [
        (15, 15, 16, True, 0),  # incremental decode, no external KV
        (128, 128, 216, True, 0),  # second prefill chunk
        (128, 128, 129, False, 0),  # local prefix hit
        (0, 128, 129, False, 1),  # synchronous external load
        (64, 128, 129, False, 1),  # mixed local and external hit
        (0, 128, 128, False, 0),  # asynchronous load only
        (0, 128, 129, True, 0),  # existing allocation needs no load slot
    ],
)
def test_only_new_synchronous_external_loads_reserve_extra_state(local, total, main, tracked, expected):
    manager = AscendMambaManager.__new__(AscendMambaManager)
    manager.block_size = 512
    manager.num_cached_block = {"request": 0} if tracked else {}
    parent = AscendMambaManager.__mro__[1]
    with patch.object(parent, "get_num_blocks_to_allocate", return_value=0):
        assert (
            manager.get_num_blocks_to_allocate(
                request_id="request",
                num_tokens=main,
                new_computed_blocks=[],
                total_computed_tokens=total,
                num_local_computed_tokens=local,
                num_tokens_main_model=main,
            )
            == expected
        )


def test_running_request_advances_without_spare_state_blocks():
    # A hybrid request can occupy every non-null block in its cache pool.
    # Neither decode nor another prefill chunk needs a new state page here.
    pool = BlockPool(num_gpu_blocks=2, enable_caching=False, hash_block_size=512)
    spec = MambaSpec(
        block_size=512,
        shapes=((3, 8), (2, 4, 4)),
        dtypes=(torch.bfloat16, torch.float32),
        mamba_cache_mode="none",
    )
    manager = AscendMambaManager(
        spec,
        pool,
        enable_caching=False,
        kv_cache_group_id=0,
        scheduler_block_size=512,
    )
    manager.add_local_computed_blocks("request", [], 0, 0)
    blocks = manager.allocate_new_blocks("request", 15, 15)
    assert len(blocks) == 1
    assert pool.get_num_free_blocks() == 0
    for local, main in ((15, 16), (128, 212)):
        required = manager.get_num_blocks_to_allocate(
            "request",
            main,
            [],
            local,
            local,
            main,
        )
        assert required == 0
        assert required <= pool.get_num_free_blocks()
        assert manager.allocate_new_blocks("request", main, main) == []
