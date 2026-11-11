# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import math
import random
import sys
import unittest
from pathlib import Path

_module_path = next(
    (
        parent / "vllm_ascend/quantization/methods/kv_cache/dquant_codec.py"
        for parent in Path(__file__).resolve().parents
        if (parent / "vllm_ascend/quantization/methods/kv_cache/dquant_codec.py").is_file()
    ),
    Path(__file__).resolve().parent / "dquant.py",
)
_spec = importlib.util.spec_from_file_location("dquant_codec", _module_path)
assert _spec is not None and _spec.loader is not None
dquant_codec = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = dquant_codec
_spec.loader.exec_module(dquant_codec)

RansTable = dquant_codec.RansTable
DQuantCodec = dquant_codec.DQuantCodec
DQuantCodecConfig = dquant_codec.DQuantCodecConfig
DQuantPagedCache = dquant_codec.DQuantPagedCache
decode_stream = dquant_codec.decode_stream
encode_stream = dquant_codec.encode_stream
frequencies_from_probabilities = dquant_codec.frequencies_from_probabilities
gaussian_order_statistic_probabilities = dquant_codec.gaussian_order_statistic_probabilities
gaussian_order_statistic_table = dquant_codec.gaussian_order_statistic_table
hadamard_rotate = dquant_codec.hadamard_rotate
remove_windowed_key_mean = dquant_codec.remove_windowed_key_mean
sharpen_probabilities = dquant_codec.sharpen_probabilities


class DQuantCodecTests(unittest.TestCase):
    def setUp(self):
        self.table = frequencies_from_probabilities([0.28, 0.22, 0.16, 0.12, 0.10, 0.07, 0.04, 0.01])

    def test_frequency_table_is_positive_and_normalized(self):
        self.assertEqual(sum(self.table.frequencies), 256)
        self.assertTrue(all(freq > 0 for freq in self.table.frequencies))

    def test_fixed_container_round_trip(self):
        rng = random.Random(7)
        values = [math.sin(index / 7) + rng.uniform(-0.01, 0.01) for index in range(128)]

        stream = encode_stream(values, self.table, container_bytes=96)
        decoded = decode_stream(stream, self.table)

        self.assertEqual(len(stream.payload), 96)
        self.assertEqual(len(decoded), len(values))
        self.assertLess(max(abs(a - b) for a, b in zip(values, decoded)), 0.8)

    def test_encode_is_deterministic(self):
        values = [float((index * 17) % 31) for index in range(64)]
        first = encode_stream(values, self.table, container_bytes=64)
        second = encode_stream(values, self.table, container_bytes=64)
        self.assertEqual(first, second)

    def test_rejects_container_below_minimum_rate(self):
        with self.assertRaises(ValueError):
            encode_stream([0.0] * 16, self.table, container_bytes=4)

    def test_rans_round_trip_without_quantizer(self):
        _rans_decode = dquant_codec._rans_decode
        _rans_encode = dquant_codec._rans_encode

        symbols = [0, 4, 1, 7, 2, 2, 0, 5, 3, 1] * 9
        encoded = _rans_encode(symbols, self.table)
        self.assertEqual(_rans_decode(encoded, len(symbols), self.table), symbols)

    def test_paper_default_key_and_value_stream_budgets(self):
        rng = random.Random(13)
        values = [rng.gauss(0.0, 1.0) for _ in range(1024)]
        codec = DQuantCodec(DQuantCodecConfig(1024, 331, 231))
        key_stream = codec.encode_key(values)
        value_stream = codec.encode_value(values)
        self.assertEqual(len(key_stream.payload), 331)
        self.assertEqual(len(value_stream.payload), 231)
        self.assertEqual(len(codec.decode_key(key_stream)), len(values))
        self.assertEqual(len(codec.decode_value(value_stream)), len(values))

    def test_codec_config_rejects_container_without_rans_state(self):
        with self.assertRaises(ValueError):
            DQuantCodecConfig(1024, 3, 231)

    def test_codec_rejects_mismatched_stream_length(self):
        codec = DQuantCodec(DQuantCodecConfig(8, 8, 8))
        with self.assertRaises(ValueError):
            codec.encode_key([0.0] * 7)

    def test_paged_cache_round_trip_and_storage_accounting(self):
        codec = DQuantCodec(DQuantCodecConfig(32, 24, 20))
        cache = DQuantPagedCache(codec, capacity=4)
        key = [math.sin(index / 3) for index in range(32)]
        value = [math.cos(index / 4) for index in range(32)]
        cache.write(1, key, value)

        decoded_key, decoded_value = cache.read(1)
        self.assertEqual(cache.bytes_per_slot, 24 + 20 + 8)
        self.assertEqual(cache.storage_bytes, 4 * cache.bytes_per_slot)
        self.assertAlmostEqual(cache.bits_per_value, 52 * 8 / 64)
        self.assertLess(max(abs(a - b) for a, b in zip(key, decoded_key)), 0.5)
        self.assertLess(max(abs(a - b) for a, b in zip(value, decoded_value)), 0.8)

    def test_paged_cache_copy_uses_snapshot_semantics(self):
        codec = DQuantCodec(DQuantCodecConfig(16, 16, 16))
        cache = DQuantPagedCache(codec, capacity=3)
        first = [float(index) / 8 for index in range(16)]
        second = [-float(index) / 5 for index in range(16)]
        cache.write(0, first, first)
        cache.write(1, second, second)

        cache.copy_slots([(0, 1), (1, 2)])
        copied_first, _ = cache.read(1)
        copied_second, _ = cache.read(2)
        expected_first = codec.decode_key(codec.encode_key(first))
        expected_second = codec.decode_key(codec.encode_key(second))
        self.assertEqual(copied_first, expected_first)
        self.assertEqual(copied_second, expected_second)

    def test_paged_cache_copies_blocks_with_vllm_mapping_order(self):
        codec = DQuantCodec(DQuantCodecConfig(8, 8, 8))
        cache = DQuantPagedCache(codec, capacity=4)
        values = [[float(index + slot * 10) for index in range(8)] for slot in range(4)]
        for slot, stream in enumerate(values):
            cache.write(slot, stream, stream)

        cache.copy_blocks([(0, 1), (1, 0)], block_size=2)

        for slot in range(2):
            self.assertEqual(cache.read(slot)[0], codec.decode_key(codec.encode_key(values[slot + 2])))
            self.assertEqual(cache.read(slot + 2)[0], codec.decode_key(codec.encode_key(values[slot])))

    def test_paged_cache_rejects_uninitialized_slots(self):
        cache = DQuantPagedCache(DQuantCodec(DQuantCodecConfig(8, 8, 8)), capacity=2)
        with self.assertRaises(ValueError):
            cache.read(0)


class DQuantPreprocessTests(unittest.TestCase):
    def test_hadamard_transform_is_orthonormal_and_self_inverse(self):
        values = [0.25, -1.0, 2.5, 0.75, -0.5, 1.25, 3.0, -2.0]
        rotated = hadamard_rotate(values, block_size=4)
        restored = hadamard_rotate(rotated, block_size=4)
        self.assertTrue(all(abs(a - b) < 1e-12 for a, b in zip(values, restored)))

    def test_key_mean_is_removed_and_returned_per_window(self):
        keys = [[1.0, 4.0], [3.0, 8.0], [10.0, -2.0]]
        centered, means = remove_windowed_key_mean(keys, window_size=2)
        self.assertEqual(means, [[2.0, 6.0], [10.0, -2.0]])
        self.assertEqual(centered, [[-1.0, -2.0], [1.0, 2.0], [0.0, 0.0]])

    def test_rejects_invalid_rans_frequency_table(self):
        with self.assertRaises(ValueError):
            RansTable((2, 3))

    def test_probability_sharpening_preserves_normalization(self):
        sharpened = sharpen_probabilities([0.5, 0.3, 0.2], alpha=1.4)
        self.assertAlmostEqual(sum(sharpened), 1.0)
        self.assertGreater(sharpened[0], 0.5)
        self.assertLess(sharpened[-1], 0.2)

    def test_rejects_invalid_sharpening_parameters(self):
        with self.assertRaises(ValueError):
            sharpen_probabilities([0.5, 0.5], alpha=0.0)

    def test_gaussian_order_statistic_model_is_normalized_and_symmetric(self):
        probabilities = gaussian_order_statistic_probabilities(stream_length=1024, levels=8)
        self.assertAlmostEqual(sum(probabilities), 1.0)
        for left, right in zip(probabilities, reversed(probabilities)):
            self.assertAlmostEqual(left, right, places=12)
        table = gaussian_order_statistic_table(stream_length=1024, levels=8)
        self.assertEqual(len(table.frequencies), 8)
        self.assertEqual(table.total, 256)

    def test_gaussian_order_statistic_model_rejects_invalid_shape(self):
        with self.assertRaises(ValueError):
            gaussian_order_statistic_probabilities(stream_length=1, levels=8)


if __name__ == "__main__":
    unittest.main()
