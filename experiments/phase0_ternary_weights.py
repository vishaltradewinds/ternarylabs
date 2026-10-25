#!/usr/bin/env python3
"""Phase 0: reproducible ternary-weight representation experiment.

This is a quality/storage experiment, NOT a hardware-speed benchmark. NumPy's
matrix multiply still runs on the host's normal numeric backend.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time

import numpy as np


def quantize_ternary(weights: np.ndarray, threshold_ratio: float = 0.7):
    """Quantize each output row to {-1, 0, +1} with one FP32 scale per row."""
    if weights.ndim != 2:
        raise ValueError("weights must be a 2D matrix")
    if not 0.0 <= threshold_ratio:
        raise ValueError("threshold_ratio must be non-negative")

    row_threshold = threshold_ratio * np.mean(np.abs(weights), axis=1, keepdims=True)
    codes = np.where(weights > row_threshold, 1,
             np.where(weights < -row_threshold, -1, 0)).astype(np.int8)

    counts = np.count_nonzero(codes, axis=1, keepdims=True)
    sums = np.sum(np.abs(weights) * (codes != 0), axis=1, keepdims=True)
    scales = np.divide(sums, counts, out=np.zeros_like(sums), where=counts != 0).astype(np.float32)
    return codes, scales


def run(seed: int = 20261025, batch: int = 32, in_features: int = 256,
        out_features: int = 256, threshold_ratio: float = 0.7) -> dict:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((batch, in_features), dtype=np.float32)
    w = rng.standard_normal((in_features, out_features), dtype=np.float32)

    # FP32 reference.
    t0 = time.perf_counter()
    y_ref = x @ w
    reference_seconds = time.perf_counter() - t0

    # Quantize each output column independently.
    codes, scales = quantize_ternary(w.T, threshold_ratio=threshold_ratio)
    w_approx = (codes.astype(np.float32) * scales).T

    t1 = time.perf_counter()
    y_ternary_reference = x @ w_approx
    ternary_host_seconds = time.perf_counter() - t1

    # Check the ternary representation's reconstruction is internally consistent.
    y_from_codes = x @ (codes.astype(np.float32) * scales).T
    np.testing.assert_allclose(y_ternary_reference, y_from_codes, rtol=1e-5, atol=1e-5)

    error = y_ternary_reference - y_ref
    ref_norm = float(np.linalg.norm(y_ref))
    nonzero = int(np.count_nonzero(codes))
    total = int(codes.size)
    fp32_weight_bytes = int(w.nbytes)
    # Theoretical 2-bit packing plus one FP32 scale per output feature.
    ternary_packed_bytes = int(np.ceil(total * 2 / 8) + out_features * np.dtype(np.float32).itemsize)

    return {
        "experiment": "phase0_ternary_weight_quality_storage",
        "seed": seed,
        "shape": {"batch": batch, "in_features": in_features, "out_features": out_features},
        "threshold_ratio": threshold_ratio,
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform()},
        "metrics": {
            "ternary_nonzero_fraction": nonzero / total,
            "fp32_weight_bytes": fp32_weight_bytes,
            "ternary_2bit_packed_plus_fp32_scales_bytes": ternary_packed_bytes,
            "theoretical_weight_storage_reduction_ratio": fp32_weight_bytes / ternary_packed_bytes,
            "output_mae": float(np.mean(np.abs(error))),
            "output_rmse": float(np.sqrt(np.mean(np.square(error)))),
            "output_relative_l2_error": float(np.linalg.norm(error) / ref_norm) if ref_norm else 0.0,
            "fp32_matmul_seconds_single_run": reference_seconds,
            "ternary_dequantized_matmul_seconds_single_run": ternary_host_seconds,
        },
        "interpretation_guardrails": [
            "Storage is a theoretical estimate: actual packing format and metadata can change the result.",
            "The ternary host timing is not a ternary-kernel speed measurement; it uses ordinary NumPy matmul.",
            "No energy, silicon-area, or hardware-speed claim is made.",
            "Quality results depend on seed, dimensions, distribution, and threshold ratio.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--in-features", type=int, default=256)
    parser.add_argument("--out-features", type=int, default=256)
    parser.add_argument("--threshold-ratio", type=float, default=0.7)
    parser.add_argument("--output", help="Optional path to write JSON results")
    args = parser.parse_args()

    result = run(args.seed, args.batch, args.in_features, args.out_features, args.threshold_ratio)
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")


if __name__ == "__main__":
    main()
