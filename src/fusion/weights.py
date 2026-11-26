"""RQ3: choose the EDA weight w for p = w * p_EDA + (1 - w) * p_BVP on inner validation only."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.eval.metrics import macro_f1


def score_weights(oof: pd.DataFrame, grid, threshold: float = 0.5) -> pd.DataFrame:
    """oof: subject, y_true, p_eda, p_bvp for inner-validation windows.
    Score = mean over inner-validation subjects of per-subject Macro-F1."""
    rows = []
    for w in grid:
        p = w * oof.p_eda + (1 - w) * oof.p_bvp
        pred = (p >= threshold).astype(int)
        per_subject = [macro_f1(oof.y_true[g.index], pred[g.index]) for _, g in oof.groupby("subject")]
        rows.append(dict(w=float(w), inner_macro_f1=float(np.mean(per_subject))))
    return pd.DataFrame(rows)


def choose_weight(scores: pd.DataFrame, tie_break: str = "closest_to_0.5") -> float:
    best = scores.inner_macro_f1.max()
    tied = scores[np.isclose(scores.inner_macro_f1, best)]
    if tie_break == "closest_to_0.5":
        return float(tied.iloc[(tied.w - 0.5).abs().argsort()].w.iloc[0])
    if tie_break == "smallest":
        return float(tied.w.min())
    raise ValueError(f"unknown tie_break {tie_break}")
