# SPDX-License-Identifier: Apache-2.0
import pytest

from vllm_ascend.ascend_config import NovaKVConfig


def test_capture_configuration():
    assert NovaKVConfig(capture_path="/capture").artifact_path is None


def test_cube_configuration_without_direct_library():
    options = NovaKVConfig(artifact_path="/artifacts", backend="cube", prefill_query_tile=8192)
    assert options.direct_library_root is None
    assert options.backend == "cube"


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"artifact_path": "/artifacts"},
        {"capture_path": "/capture", "artifact_path": "/artifacts"},
        {"capture_path": "/capture", "recent_tokens": 0},
        {"capture_path": "/capture", "codec_batch_size": 0},
        {"capture_path": "/capture", "direct_split_tokens": 2048},
        {"capture_path": "/capture", "prefill_query_tile": 2049},
        {"capture_path": "/capture", "unknown_option": True},
        {"artifact_path": "/artifacts", "backend": "unknown"},
        {"capture_path": "/capture", "backend": "cube"},
        {"artifact_path": "/artifacts", "backend": "cube", "direct_split_tokens": 512},
        {"artifact_path": "/artifacts", "backend": "cube", "prefill_query_tile": 8193},
    ],
)
def test_invalid_configuration(options):
    with pytest.raises(ValueError):
        NovaKVConfig(**options)
