import unittest

import numpy as np

from src.features import RAW_STATS, raw_feature_block


class FeatureTests(unittest.TestCase):
    def test_normalized_line_length_is_duration_independent(self):
        short = np.array([[0.0, 1.0, 0.0, 1.0]])
        long = np.array([[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]])
        index = RAW_STATS.index("line_length")
        short_value = raw_feature_block(short, line_length="normalized")[index, 0]
        long_value = raw_feature_block(long, line_length="normalized")[index, 0]
        self.assertAlmostEqual(short_value, long_value)

    def test_cumulative_line_length_remains_available(self):
        signal = np.array([[0.0, 1.0, 0.0, 1.0]])
        index = RAW_STATS.index("line_length")
        value = raw_feature_block(signal, line_length="sum")[index, 0]
        self.assertAlmostEqual(value, 3.0)


if __name__ == "__main__":
    unittest.main()
