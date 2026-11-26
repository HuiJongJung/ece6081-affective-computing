"""M0: handcrafted features + Logistic Regression (fixed setting)."""
from __future__ import annotations

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import handcrafted
from src.models.common import FIT_PARTS, add_prediction_fusion


def make_lr(params: dict) -> Pipeline:
    # keep_empty_features: a column that is all-NaN in training stays (filled with 0) so
    # feature counts never depend on which subjects are in the fold
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(**params)),
    ])


class M0:
    name = "m0"

    def __init__(self, cfg: dict, seed: int):
        self.params = dict(cfg["baseline"]["params"])
        self.pipes: dict[str, Pipeline] = {}

    @staticmethod
    def _inputs(eda, bvp) -> dict:
        fe = handcrafted.compute(eda, "eda")
        fb = handcrafted.compute(bvp, "bvp")
        return {"eda": fe, "bvp": fb, "feature_fusion": np.hstack([fe, fb])}

    def fit(self, data, idx, parts=FIT_PARTS, ctx=None):
        X = self._inputs(data.eda[idx], data.bvp[idx])
        for p in parts:
            self.pipes[p] = make_lr(self.params).fit(X[p], data.y[idx])
        return self

    def predict(self, eda, bvp) -> dict:
        X = self._inputs(eda, bvp)
        probs = {p: pipe.predict_proba(X[p])[:, 1] for p, pipe in self.pipes.items()}
        return add_prediction_fusion(probs)

    def save(self, path):
        joblib.dump(self.pipes, path.with_suffix(".joblib"))

    def load(self, path):
        self.pipes = joblib.load(path.with_suffix(".joblib"))
        return self
