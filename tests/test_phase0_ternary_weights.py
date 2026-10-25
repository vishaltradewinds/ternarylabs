import unittest
import numpy as np

from experiments.phase0_ternary_weights import quantize_ternary, run


class TernaryWeightTests(unittest.TestCase):
    def test_codes_are_ternary(self):
        weights = np.array([[0.0, 1.0, -2.0, 0.2]], dtype=np.float32)
        codes, scales = quantize_ternary(weights, threshold_ratio=0.5)
        self.assertTrue(set(np.unique(codes)).issubset({-1, 0, 1}))
        self.assertEqual(codes.shape, weights.shape)
        self.assertEqual(scales.shape, (1, 1))

    def test_seeded_run_is_reproducible_for_quality_metrics(self):
        a = run(seed=123, batch=4, in_features=8, out_features=6)
        b = run(seed=123, batch=4, in_features=8, out_features=6)
        for key in ("ternary_nonzero_fraction", "output_mae", "output_rmse",
                    "output_relative_l2_error", "fp32_weight_bytes",
                    "ternary_2bit_packed_plus_fp32_scales_bytes"):
            self.assertEqual(a["metrics"][key], b["metrics"][key])

    def test_packed_estimate_is_smaller_for_default_shape(self):
        result = run(seed=123, batch=4, in_features=256, out_features=256)
        metrics = result["metrics"]
        self.assertLess(metrics["ternary_2bit_packed_plus_fp32_scales_bytes"],
                        metrics["fp32_weight_bytes"])


if __name__ == "__main__":
    unittest.main()
