"""LOSO runner shared by every model, and helpers to reload fold models."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.models.common import FitContext
from src.utils import resolve, set_seed


def get_model(name: str):
    if name == "m0":
        from src.models.baseline import M0
        return M0
    if name == "m1":
        from src.models.deep import M1
        return M1
    if name == "m2":
        from src.models.jepa import M2
        return M2
    raise ValueError(f"unknown model {name}")


def pick_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def run_name(model: str, seed: int, bootstrap: bool = False) -> str:
    return f"{model}_seed{seed}" + ("_boot" if bootstrap else "")


def subject_index(data, subjects) -> np.ndarray:
    """Window indices of the given subjects; a subject listed twice appears twice (bootstrap)."""
    return np.concatenate([np.flatnonzero(data.subject == s) for s in subjects])


def training_subjects(subjects: list[str], test: str, seed: int, fold: int, bootstrap: bool) -> list[str]:
    train = [s for s in subjects if s != test]
    if bootstrap:
        rng = np.random.default_rng([seed, fold])
        train = list(rng.choice(train, size=len(train), replace=True))
    return train


def prediction_frame(data, idx, probs: dict, model: str, run: str, threshold: float) -> pd.DataFrame:
    base = data.index_frame().iloc[idx].reset_index(drop=True)
    frames = []
    for cond, p in probs.items():
        f = base.copy()
        f.insert(0, "run", run)
        f.insert(0, "model", model)
        f["condition"] = cond
        f["prob"] = p
        f["y_pred"] = (p >= threshold).astype(int)
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def fold_path(cfg, run: str, test: str) -> Path:
    d = resolve(cfg["paths"]["models_dir"]) / run
    d.mkdir(parents=True, exist_ok=True)
    return d / test


def run_loso(model_name: str, cfg: dict, data, seed: int, bootstrap: bool = False, log=print):
    """Train on 14 subjects, evaluate the held-out one, for all 15; return predictions and split log."""
    subjects = cfg["data"]["subjects"]
    run = run_name(model_name, seed, bootstrap)
    device = pick_device()
    preds, splits = [], []
    for fold, test in enumerate(subjects):
        train = training_subjects(subjects, test, seed, fold, bootstrap)
        idx = subject_index(data, train)
        te = np.flatnonzero(data.subject == test)
        set_seed(seed * 1000 + fold)
        ctx = FitContext(resolve(cfg["paths"]["processed_dir"]), train, seed * 1000 + fold, device)
        model = get_model(model_name)(cfg, seed).fit(data, idx, ctx=ctx)
        model.save(fold_path(cfg, run, test))
        probs = model.predict(data.eda[te], data.bvp[te])
        preds.append(prediction_frame(data, te, probs, model_name, run, cfg["evaluation"]["threshold"]))
        splits.append(dict(run=run, fold=fold, test_subject=test, train_subjects=" ".join(train),
                           n_train_windows=len(idx), n_test_windows=len(te)))
        log(f"[{run}] fold {fold + 1:2d}/{len(subjects)} test={test} train_windows={len(idx)} test_windows={len(te)}")
    return pd.concat(preds, ignore_index=True), pd.DataFrame(splits)


def load_fold_model(model_name: str, cfg: dict, run: str, test: str, seed: int):
    model = get_model(model_name)(cfg, seed).load(fold_path(cfg, run, test))
    model.device = pick_device()
    return model
