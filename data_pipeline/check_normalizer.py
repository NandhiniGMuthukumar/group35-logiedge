"""Check a saved normalizer on its calibration capture without refitting it."""

import argparse
from pathlib import Path

import numpy as np

from data_pipeline.normalization import DEFAULT_STATS, FrozenNormalizer, file_sha256, metadata_path


def check_normalizer(path=DEFAULT_STATS):
    path = Path(path)
    before = (file_sha256(path), file_sha256(metadata_path(path)))
    normalizer = FrozenNormalizer.load(path)
    feature_path = Path(normalizer.metadata["source_feature_npy"])
    if file_sha256(feature_path) != normalizer.metadata["source_feature_npy_sha256"]:
        raise ValueError("Baseline feature file differs from its recorded hash")
    values = np.load(feature_path, allow_pickle=False)
    if values.shape != (normalizer.metadata["window_count"], 6):
        raise ValueError("Baseline feature array shape differs from metadata")
    transformed = normalizer.transform(values)
    shifted = normalizer.transform(values, mean_shift_sigma=3.0)
    means = transformed.astype(np.float64).mean(axis=0)
    stds = transformed.astype(np.float64).std(axis=0, ddof=0)
    expected_stds = np.asarray(normalizer.metadata["raw_std"]) / normalizer.std
    if not np.allclose(means, 0, atol=1e-5):
        raise ValueError("Calibration normalized means are not near zero")
    if not np.allclose(stds, expected_stds, atol=1e-5):
        raise ValueError("Calibration normalized standard deviations are inconsistent")
    if not np.allclose(shifted - transformed, -3.0, atol=1e-5):
        raise ValueError("Experimental mean shift does not have the expected effect")
    after = (file_sha256(path), file_sha256(metadata_path(path)))
    if before != after:
        raise ValueError("Transforming features changed the saved statistics")
    print(f"Loaded statistics: {path}")
    print(f"Baseline feature shape: {values.shape}")
    print("Normalized baseline means:", [round(float(value), 6) for value in means])
    print("Normalized baseline stds:", [round(float(value), 6) for value in stds])
    print("First normalized vector:", [round(float(value), 6) for value in transformed[0]])
    print("Floored features:", normalizer.metadata["floored_features"])
    print("Statistics and metadata unchanged after transforms:", before == after)
    print("3-sigma mean-shift arithmetic: checked")
    print("Model accuracy comparison: pending model training and held-out evaluation")
    return transformed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    args = parser.parse_args(argv)
    try:
        check_normalizer(args.stats)
    except (OSError, ValueError, KeyError, TypeError, EOFError) as exc:
        print(f"Normalizer check stopped: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
