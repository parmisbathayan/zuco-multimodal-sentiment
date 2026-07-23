import unittest

import numpy as np
import pandas as pd

from src.closeout_analysis import (
    CONTROLS,
    PRIORITY_SUBSETS,
    _cluster_accuracy_interval,
    evaluate_stop_decision,
)


class CloseoutAnalysisTests(unittest.TestCase):
    def test_cluster_bootstrap_keeps_sentence_as_sampling_unit(self):
        frame = pd.DataFrame(
            {
                "sentence_id": np.repeat([10, 20, 30], 3),
                "delta_correct": np.ones(9),
            }
        )
        low, high = _cluster_accuracy_interval(
            frame, samples=100, rng=np.random.default_rng(42)
        )
        self.assertEqual(low, 1.0)
        self.assertEqual(high, 1.0)

    def test_stop_decision_requires_every_control(self):
        rows = []
        for control in CONTROLS:
            rows.append(
                {
                    "text_mode": "finetune",
                    "control": control,
                    "subset_name": PRIORITY_SUBSETS[0],
                    "delta_accuracy": 0.02,
                    "delta_accuracy_ci95_low": 0.001,
                    "n_seeds_aligned_better": 2,
                }
            )
        passed = evaluate_stop_decision(pd.DataFrame(rows), minimum_delta=0.015)
        self.assertTrue(passed["alignment_specific_priority_subset_detected"])

        missing_control = evaluate_stop_decision(
            pd.DataFrame(rows[:-1]), minimum_delta=0.015
        )
        self.assertFalse(
            missing_control["alignment_specific_priority_subset_detected"]
        )


if __name__ == "__main__":
    unittest.main()
