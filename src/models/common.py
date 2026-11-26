"""Pieces shared by the fold models (M0, M1, M2)."""
from __future__ import annotations

import numpy as np

MODALITIES = ("eda", "bvp")
FIT_PARTS = ("eda", "bvp", "feature_fusion")        # parts that need training
CONDITIONS = ("eda", "bvp", "feature_fusion", "prediction_fusion")


def add_prediction_fusion(probs: dict, w: float = 0.5) -> dict:
    """Prediction fusion = w * p_EDA + (1 - w) * p_BVP (common experiment: w = 0.5)."""
    if "eda" in probs and "bvp" in probs:
        probs["prediction_fusion"] = w * probs["eda"] + (1 - w) * probs["bvp"]
    return probs


class Standardizer:
    """One mean / std per modality, fitted on training windows only."""

    def fit(self, x: np.ndarray) -> "Standardizer":
        self.mean = float(np.mean(x))
        self.std = float(np.std(x)) or 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def state(self) -> dict:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_state(cls, s: dict) -> "Standardizer":
        obj = cls()
        obj.mean, obj.std = s["mean"], s["std"]
        return obj


class FitContext:
    """What a fold model may see while fitting: training subjects only."""

    def __init__(self, processed_dir, train_subjects: list[str], seed: int, device: str):
        self.processed_dir = processed_dir
        self.train_subjects = train_subjects   # may contain repeats (bootstrap)
        self.seed = seed
        self.device = device
