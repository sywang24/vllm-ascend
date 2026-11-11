import importlib.util
import math
import random
import sys
import unittest
from pathlib import Path

codec_path = next(
    parent / "vllm_ascend/quantization/methods/kv_cache/dquant_codec.py"
    for parent in Path(__file__).resolve().parents
    if (parent / "vllm_ascend/quantization/methods/kv_cache/dquant_codec.py").is_file()
)
spec = importlib.util.spec_from_file_location("dquant_codec", codec_path)
assert spec is not None and spec.loader is not None
codec = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = codec
spec.loader.exec_module(codec)


class DriftTests(unittest.TestCase):
    def setUp(self):
        self.table = codec.frequencies_from_probabilities([0.28, 0.22, 0.16, 0.12, 0.10, 0.07, 0.04, 0.01])

    def test_rate_budget_is_respected(self):
        values = [-1.0 + index / 8 for index in range(17)]
        symbols = codec._drift(values, 2.0 / 7, -1.0, self.table, 48.0)
        coded_rate = sum(self.table.rate_bits[symbol] for symbol in symbols)
        self.assertLessEqual(coded_rate, 48.0)

    def test_zero_distortion_range_uses_cheapest_symbol(self):
        values = [3.0] * 16
        symbols = codec._drift(values, 0.0, 3.0, self.table, 64.0)
        cheapest_symbol = min(range(len(self.table.frequencies)), key=self.table.rate_bits.__getitem__)
        self.assertEqual(symbols, [cheapest_symbol] * len(values))

    def test_fp16_refit_stream_stays_within_fixed_container(self):
        rng = random.Random(31)
        values = [rng.gauss(0.0, 1.0) for _ in range(1024)]
        table = codec.gaussian_order_statistic_table(1024, 8)
        stream = codec.encode_stream(values, table, container_bytes=331)
        self.assertEqual(len(stream.payload), 331)
        self.assertTrue(all(math.isfinite(value) for value in codec.decode_stream(stream, table)))


if __name__ == "__main__":
    unittest.main()
