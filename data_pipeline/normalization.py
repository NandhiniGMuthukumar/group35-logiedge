"""Load and apply fixed training statistics; this runtime class never fits data."""

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from data_pipeline.preprocessing import FEATURE_NAMES, RECIPE

DEFAULT_STATS = Path("data_pipeline/training_stats.npy")
STD_FLOOR = 1e-8


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metadata_path(stats_path):
    return Path(stats_path).with_suffix(".meta.json")


@dataclass(frozen=True)
class FrozenNormalizer:
    mean: np.ndarray
    std: np.ndarray
    metadata: dict = field(default_factory=dict, repr=False)
    stats_path: Path | None = None
    stats_sha256: str = ""

    def __post_init__(self):
        mean = np.array(self.mean, dtype=np.float64, copy=True)
        std = np.array(self.std, dtype=np.float64, copy=True)
        if mean.shape != (6,) or std.shape != (6,):
            raise ValueError("Normalization needs one mean and std for each of six features")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std <= 0):
            raise ValueError("Statistics must be finite, with positive standard deviations")
        mean.setflags(write=False)
        std.setflags(write=False)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)

    @classmethod
    def load(cls, stats_path=DEFAULT_STATS):
        path = Path(stats_path)
        meta = json.loads(metadata_path(path).read_text(encoding="utf-8"))
        if meta.get("schema_version") != 1 or meta.get("stats_rows") != ["mean", "std"]:
            raise ValueError("Unsupported statistics format")
        if meta.get("feature_names") != list(FEATURE_NAMES):
            raise ValueError("Statistics feature order does not match the preprocessing code")
        if meta.get("recipe") != RECIPE:
            raise ValueError("Statistics preprocessing recipe does not match the runtime")
        digest = file_sha256(path)
        if digest != meta.get("stats_sha256"):
            raise ValueError("Statistics file hash differs from its saved metadata")
        matrix = np.load(path, allow_pickle=False)
        if matrix.shape != (2, 6) or matrix.dtype.kind != "f":
            raise ValueError("Statistics file must be a floating-point array of shape (2, 6)")
        return cls(matrix[0], matrix[1], meta, path, digest)

    def transform(self, features, mean_shift_sigma=0.0):
        """Apply z=(x-mean)/std without changing any fitted statistic.

        The optional mean shift is solely for the required later experiment:
        +3 sigma means shifting the saved means by +3*std, with std unchanged.
        """
        values = np.asarray(features, dtype=np.float64)
        if (values.ndim not in (1, 2) or values.shape[-1] != 6 or
                (values.ndim == 2 and values.shape[0] == 0)):
            raise ValueError("Expected a six-value vector or a nonempty (N, 6) matrix")
        if not np.all(np.isfinite(values)) or not math.isfinite(mean_shift_sigma):
            raise ValueError("Features and experimental shift must be finite")
        result = (values - (self.mean + mean_shift_sigma * self.std)) / self.std
        if not np.all(np.isfinite(result)) or np.any(np.abs(result) > np.finfo(np.float32).max):
            raise ValueError("Normalized features are outside the supported finite range")
        return result.astype(np.float32)
