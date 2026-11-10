# SPDX-License-Identifier: Apache-2.0
import json
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.ops.triton import nova_direct_decode as decode


def write_manifest(tmp_path, **overrides):
    manifest = dict(
        abi_version=1, head_dim=256, lut_dtype="float16", tile_tokens=128, max_split_tokens=1024, max_query_groups=6
    )
    manifest.update(overrides)
    library = tmp_path / "lib"
    library.mkdir()
    (library / "nova_kernel_abi.json").write_text(json.dumps(manifest))


@pytest.mark.parametrize(
    "overrides",
    [
        {"abi_version": 2},
        {"lut_dtype": "float32"},
        {"tile_tokens": 64},
        {"head_dim": 128},
        {"max_split_tokens": 512},
        {"max_query_groups": 0},
    ],
)
def test_rejects_incompatible_kernel_before_loading(tmp_path, monkeypatch, overrides):
    write_manifest(tmp_path, **overrides)

    def unexpected_load(*args):
        pytest.fail("incompatible library must not be loaded")

    monkeypatch.setattr(decode.ctypes, "CDLL", unexpected_load)
    with pytest.raises(ValueError, match="ABI mismatch"):
        decode.DirectDecode(1, 1, 1, "cpu", str(tmp_path), 1024)


def test_missing_manifest_rejected(tmp_path):
    with pytest.raises(ValueError, match="manifest"):
        decode.DirectDecode(1, 1, 1, "cpu", str(tmp_path), 1024)


def test_workspace_accounts_for_allocated_buffers(tmp_path, monkeypatch):
    write_manifest(tmp_path)
    launch = SimpleNamespace()
    monkeypatch.setattr(decode.ctypes, "CDLL", lambda path: SimpleNamespace(aclrtlaunch_nova_attention_shared=launch))
    workspace = decode.DirectDecode(2, 1, 3, "cpu", str(tmp_path), 2048)
    assert workspace.partial.shape == (12, 256)
    expected = sum(
        tensor.numel() * tensor.element_size()
        for tensor in vars(workspace).values()
        if isinstance(tensor, torch.Tensor)
    )
    assert workspace.workspace_bytes() == expected


def test_lookup_table_matches_float_reference(tmp_path, monkeypatch):
    write_manifest(tmp_path)
    monkeypatch.setattr(
        decode.ctypes, "CDLL", lambda path: SimpleNamespace(aclrtlaunch_nova_attention_shared=SimpleNamespace())
    )
    workspace = decode.DirectDecode(2, 1, 3, "cpu", str(tmp_path), 2048)
    query = torch.randn(2, 1, 3, 256)
    books = torch.randn(1, 64, 256, 4).half()
    expected = torch.einsum("bhgrd,hrcd->bhgrc", query.reshape(2, 1, 3, 64, 4), books.float()).half()
    actual = workspace.prepare_lut(query, books)
    assert actual.dtype == torch.float16
    torch.testing.assert_close(actual, expected)
