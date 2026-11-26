"""M0 handcrafted features.

A window whose feature cannot be computed (e.g. too few BVP peaks) gets NaN;
the window is kept and the LR pipeline imputes it with the training median.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt, find_peaks

EDA_FS, BVP_FS = 4, 64

EDA_FEATURES = ["eda_mean", "eda_std", "eda_min", "eda_max", "eda_range", "eda_slope",
                "eda_absdiff_mean", "tonic_mean", "phasic_std", "scr_count_per_min",
                "scr_amp_mean"]
BVP_FEATURES = ["bvp_std", "bvp_iqr", "hr_mean", "hr_std", "ibi_mean", "sdnn", "rmssd",
                "pnn50", "n_beats"]

_BVP_B, _BVP_A = butter(3, [0.7, 3.5], btype="bandpass", fs=BVP_FS)


def _slope(x: np.ndarray, fs: float) -> float:
    t = np.arange(len(x)) / fs
    return float(np.polyfit(t, x, 1)[0])


def eda_features(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    tonic = uniform_filter1d(x, size=10 * EDA_FS, mode="nearest")   # 10 s moving average
    phasic = x - tonic
    peaks, props = find_peaks(phasic, prominence=0.01)
    minutes = len(x) / EDA_FS / 60
    return np.array([
        x.mean(), x.std(), x.min(), x.max(), np.ptp(x), _slope(x, EDA_FS),
        np.abs(np.diff(x)).mean(), tonic.mean(), phasic.std(), len(peaks) / minutes,
        props["prominences"].mean() if len(peaks) else 0.0,
    ])


def bvp_features(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.full(len(BVP_FEATURES), np.nan)
    out[0] = x.std()
    out[1] = np.subtract(*np.percentile(x, [75, 25]))
    try:
        f = filtfilt(_BVP_B, _BVP_A, x)
        peaks, _ = find_peaks(f, distance=int(0.33 * BVP_FS), prominence=0.3 * f.std())
    except ValueError:
        return out
    ibi = np.diff(peaks) / BVP_FS
    ibi = ibi[(ibi > 0.33) & (ibi < 1.5)]          # 40-180 bpm
    out[8] = len(peaks)
    if len(ibi) < 5:
        return out
    hr = 60.0 / ibi
    d = np.diff(ibi)
    out[2:8] = [hr.mean(), hr.std(), ibi.mean(), ibi.std(), np.sqrt(np.mean(d ** 2)),
                np.mean(np.abs(d) > 0.05)]
    return out


def compute(X: np.ndarray, kind: str) -> np.ndarray:
    fn = eda_features if kind == "eda" else bvp_features
    return np.stack([fn(x) for x in X])
