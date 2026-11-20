# SPDX-License-Identifier: Apache-2.0
"""Exercise packaged Cube selection, chunked prefill and captured B1/B4 Decode."""

import argparse
import json
import os
import random
from pathlib import Path

from vllm import LLM, SamplingParams


def main():
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    llm = LLM(
        model=args.model,
        tensor_parallel_size=2,
        dtype="bfloat16",
        kv_cache_dtype="bfloat16",
        seed=1,
        max_model_len=91024,
        max_num_seqs=4,
        max_num_batched_tokens=8192,
        gpu_memory_utilization=0.90,
        block_size=128,
        enable_prefix_caching=False,
        async_scheduling=False,
        enable_chunked_prefill=True,
        language_model_only=True,
        trust_remote_code=True,
        generation_config="vllm",
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 3, 4]},
        additional_config={
            "nova_kv": {"backend": "cube", "artifact_path": args.artifacts, "sink_tokens": 64, "recent_tokens": 256}
        },
    )
    rng = random.Random(20261120)
    rows = []
    for length, batch in ((30000, 1), (90000, 4)):
        prompts = [{"prompt_token_ids": [rng.randrange(1000, 50000) for _ in range(length)]} for _ in range(batch)]
        output = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=16, ignore_eos=True), use_tqdm=False)
        if len(output) != batch or any(len(row.outputs[0].token_ids) != 16 for row in output):
            raise AssertionError("incomplete Cube smoke output")
        rows.append(dict(length=length, batch=batch, output_tokens=[len(row.outputs[0].token_ids) for row in output]))
    args.output.write_text(json.dumps(rows, indent=2))
    llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
