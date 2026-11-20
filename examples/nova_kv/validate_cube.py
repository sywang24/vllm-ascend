# SPDX-License-Identifier: Apache-2.0
"""Check the relocated Cube path against the measured implementation on A5.

Run inside the pinned experiment container with its codebooks and original
benchmark modules available on PYTHONPATH. Output contains no model weights.
"""

import argparse
import json
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.nova_cube.nova_triton_fixed_split import TritonFixedSplitDecode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # The reference is supplied by the pinned, hardware-validated experiment.
    from bench_decode_a5 import make_case
    from nova_triton_fixed_split import TritonFixedSplitDecode as Reference

    from vllm_ascend.quantization.methods.kv_cache.nova_kv.codec import NovaCodec

    device = torch.device("npu:0")
    torch.npu.set_device(device)
    torch.manual_seed(51923)
    codec = NovaCodec.load("/nova-artifacts/layer_03.pt", device, 0, 2, 2)
    results = []
    for lengths in ([30000], [90000, 90349, 90599, 91000], [0, 64, 257, 4096]):
        batch = len(lengths)
        capacity = max(max(lengths), 1024)
        data = make_case(batch, lengths, 2, 6, 256, codec, device)
        kwargs = dict(
            batch=batch,
            score_parts=4 if batch == 4 else 10,
            tile_tokens=128,
            value_dims=128,
            merge_dims=64 if batch == 4 else 0,
            launch_options={"num_stages": 2 if batch == 4 else 1},
        )
        reference = Reference(capacity, 2, 6, device, **kwargs)
        relocated = TritonFixedSplitDecode(capacity, 2, 6, device, **kwargs)
        expected, lse = reference(*data[:-1], lut=data[-1])
        actual, actual_lse = relocated(*data[:-1], lut=data[-1])
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(actual_lse, lse, rtol=1e-5, atol=1e-5)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            relocated(*data[:-1], lut=data[-1])
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        results.append(dict(lengths=lengths, max_abs_error=(actual - expected).abs().max().item(), graph_replay=True))
    args.output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
