"""RQ2: corrupt the raw BVP of test windows (no retraining)."""
from __future__ import annotations

import numpy as np


def corrupt_bvp(bvp: np.ndarray, subjects: np.ndarray, window_ids: np.ndarray,
                snr_db: float, kind: str, seed: int) -> np.ndarray:
    """Add noise at a fixed SNR relative to each window's clean BVP power.

    The random stream depends only on (seed, subject, window_id, SNR), so every model
    sees exactly the same corrupted signal.
    """
    out = np.empty_like(bvp)
    for i, (x, sid, wid) in enumerate(zip(bvp, subjects, window_ids)):
        rng = np.random.default_rng([seed, int(str(sid).lstrip("S")), int(wid), int(round(snr_db * 10)) + 10_000])
        power = np.mean((x - x.mean()) ** 2)
        sigma = np.sqrt(power / 10 ** (snr_db / 10))
        if kind == "gaussian":
            noise = rng.normal(0.0, sigma, size=x.shape)
        else:
            raise ValueError(f"unknown noise type {kind}")
        out[i] = x + noise
    return out.astype(bvp.dtype)
