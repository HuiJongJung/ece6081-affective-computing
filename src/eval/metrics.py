"""Participant-level metrics and the tables required by the guide.

Prediction frames have one row per (window, condition):
subject, window_id, orig_label, y_true, model, run, condition, prob, y_pred
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ORIG = {1: "baseline", 2: "stress", 3: "amusement"}


def macro_f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, average="macro", labels=[0, 1], zero_division=0)


def subject_scores(pred: pd.DataFrame, keys=("model", "run", "condition")) -> pd.DataFrame:
    keys = [k for k in keys if k in pred.columns]
    return (pred.groupby(keys + ["subject"], sort=False)
                .apply(lambda g: macro_f1(g.y_true, g.y_pred), include_groups=False)
                .rename("macro_f1").reset_index())


def summarize(scores: pd.DataFrame, keys=("model", "run", "condition")) -> pd.DataFrame:
    keys = [k for k in keys if k in scores.columns]
    return (scores.groupby(keys, sort=False)["macro_f1"]
                  .agg(macro_f1_mean="mean", macro_f1_sd=lambda s: s.std(ddof=1), n_subjects="count")
                  .reset_index())


def error_table(pred: pd.DataFrame, keys=("model", "run", "condition"), per_subject=False) -> pd.DataFrame:
    """Baseline false positives, Amusement false positives, Stress misses, each as n / N."""
    keys = [k for k in keys if k in pred.columns] + (["subject"] if per_subject else [])

    def counts(g):
        out = {}
        for code, name, wrong in ((1, "baseline_fp", 1), (3, "amusement_fp", 1), (2, "stress_miss", 0)):
            sel = g.orig_label == code
            n, N = int((g.y_pred[sel] == wrong).sum()), int(sel.sum())
            out[f"{name}_n"], out[f"{name}_N"] = n, N
            out[f"{name}_pct"] = 100.0 * n / N if N else np.nan
        return pd.Series(out)

    out = pred.groupby(keys, sort=False).apply(counts, include_groups=False).reset_index()
    int_cols = [c for c in out.columns if c.endswith(("_n", "_N"))]
    out[int_cols] = out[int_cols].astype(int)
    return out


def fusion_diff(scores: pd.DataFrame, keys=("model", "run")) -> pd.DataFrame:
    """Per participant: each fusion minus each single-signal model (same model, same run)."""
    keys = [k for k in keys if k in scores.columns]
    wide = scores.pivot_table(index=keys + ["subject"], columns="condition", values="macro_f1").reset_index()
    for fusion in ("feature_fusion", "prediction_fusion"):
        for single in ("eda", "bvp"):
            wide[f"{fusion}_minus_{single}"] = wide[fusion] - wide[single]
    return wide


def error_overlap(pred: pd.DataFrame, how: str = "jaccard", empty_union: str = "nan") -> pd.DataFrame:
    """Per participant overlap of the EDA and BVP single models' error sets."""
    rows = []
    for sid, g in pred.groupby("subject", sort=False):
        e = g[g.condition == "eda"].set_index("window_id")
        b = g[g.condition == "bvp"].set_index("window_id").loc[e.index]
        err_e, err_b = e.y_pred != e.y_true, b.y_pred != b.y_true
        inter, union = int((err_e & err_b).sum()), int((err_e | err_b).sum())
        if how != "jaccard":
            raise ValueError(f"unknown overlap metric {how}")
        val = inter / union if union else (1.0 if empty_union == "one" else np.nan)
        rows.append(dict(subject=sid, eda_errors=int(err_e.sum()), bvp_errors=int(err_b.sum()),
                         both_errors=inter, either_errors=union, error_overlap=val))
    return pd.DataFrame(rows)


def improved_worse(diff: pd.Series) -> str:
    return f"{int((diff > 0).sum())} / {int((diff < 0).sum())}"
