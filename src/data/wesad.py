"""WESAD loading and the common 60 s window index.

Every model, condition and RQ reads the same windows.npz, so all of them
evaluate exactly the same windows.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def load_subject(raw_dir: str | Path, sid: str):
    """Return wrist EDA (4 Hz), wrist BVP (64 Hz) and the 700 Hz label array."""
    with open(Path(raw_dir) / sid / f"{sid}.pkl", "rb") as f:
        d = pickle.load(f, encoding="latin1")
    eda = np.asarray(d["signal"]["wrist"]["EDA"], dtype=np.float64).ravel()
    bvp = np.asarray(d["signal"]["wrist"]["BVP"], dtype=np.float64).ravel()
    label = np.asarray(d["label"]).ravel().astype(int)
    return eda, bvp, label


def window_subject(sid, eda, bvp, label, cfg):
    """Cut non-overlapping windows from the recording start.

    Returns (kept, log): kept is a list of dicts with the window arrays,
    log is one row per candidate window with the keep/exclude reason.
    """
    fs_e = cfg["data"]["signals"]["eda"]["fs"]
    fs_b = cfg["data"]["signals"]["bvp"]["fs"]
    fs_l = cfg["data"]["label_fs"]
    length = cfg["windows"]["length_s"]
    step = cfg["windows"]["step_s"]
    label_map = {int(k): int(v) for k, v in cfg["labels"]["map"].items()}

    duration = min(len(eda) / fs_e, len(bvp) / fs_b, len(label) / fs_l)
    kept, log = [], []
    k = 0
    while True:
        t0 = k * step
        if t0 >= duration:
            break
        if t0 + length > duration:
            log.append(dict(subject=sid, window_id=k, start_s=t0, orig_label=-1,
                            status="excluded", reason="incomplete_tail"))
            break
        lab = label[int(round(t0 * fs_l)): int(round((t0 + length) * fs_l))]
        uniq = np.unique(lab)
        if len(uniq) > 1:
            reason, orig = "label_transition", -1
        elif int(uniq[0]) not in label_map:
            reason, orig = f"other_label_{int(uniq[0])}", int(uniq[0])
        else:
            reason, orig = "", int(uniq[0])
        if reason:
            log.append(dict(subject=sid, window_id=k, start_s=t0, orig_label=orig,
                            status="excluded", reason=reason))
        else:
            e = eda[int(t0 * fs_e): int(t0 * fs_e) + length * fs_e]
            b = bvp[int(t0 * fs_b): int(t0 * fs_b) + length * fs_b]
            kept.append(dict(subject=sid, window_id=k, start_s=t0, orig_label=orig,
                             y=label_map[orig], eda=e, bvp=b))
            log.append(dict(subject=sid, window_id=k, start_s=t0, orig_label=orig,
                            status="kept", reason=""))
        k += 1
    return kept, log


@dataclass
class WindowData:
    eda: np.ndarray        # (N, 240)
    bvp: np.ndarray        # (N, 3840)
    y: np.ndarray          # (N,) binary label
    orig_label: np.ndarray # (N,) 1 baseline / 2 stress / 3 amusement
    subject: np.ndarray    # (N,) "S2" ...
    window_id: np.ndarray  # (N,) index from the recording start

    def index_frame(self) -> pd.DataFrame:
        return pd.DataFrame(dict(subject=self.subject, window_id=self.window_id,
                                 orig_label=self.orig_label, y_true=self.y))


def save_windows(path: Path, kept: list[dict]) -> None:
    np.savez_compressed(
        path,
        eda=np.stack([w["eda"] for w in kept]).astype(np.float32),
        bvp=np.stack([w["bvp"] for w in kept]).astype(np.float32),
        y=np.array([w["y"] for w in kept], dtype=np.int64),
        orig_label=np.array([w["orig_label"] for w in kept], dtype=np.int64),
        subject=np.array([w["subject"] for w in kept]),
        window_id=np.array([w["window_id"] for w in kept], dtype=np.int64),
    )


def load_windows(processed_dir: str | Path) -> WindowData:
    d = np.load(Path(processed_dir) / "windows.npz")
    return WindowData(eda=d["eda"], bvp=d["bvp"], y=d["y"], orig_label=d["orig_label"],
                      subject=d["subject"].astype(str), window_id=d["window_id"])


def save_continuous(processed_dir: Path, sid: str, eda: np.ndarray, bvp: np.ndarray, fs_e: int, fs_b: int):
    """Whole wrist recording (label-free), used only for JEPA pre-training on training subjects."""
    n = int(min(len(eda) / fs_e, len(bvp) / fs_b) * fs_e)   # whole seconds aligned to EDA samples
    np.savez_compressed(processed_dir / "continuous" / f"{sid}.npz",
                        eda=eda[:n].astype(np.float32), bvp=bvp[: n * (fs_b // fs_e)].astype(np.float32))


def load_continuous(processed_dir: str | Path, sid: str):
    d = np.load(Path(processed_dir) / "continuous" / f"{sid}.npz")
    return d["eda"], d["bvp"]
