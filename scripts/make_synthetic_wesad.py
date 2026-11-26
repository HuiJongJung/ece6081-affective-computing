"""Write a small fake WESAD (same pickle layout) to check that the pipeline runs end to end.

Synthetic results are only a run check and must never be reported as WESAD results.

  python scripts/make_synthetic_wesad.py
  python scripts/make_windows.py --raw-dir data/synthetic/raw/WESAD --processed-dir data/synthetic/processed \
      --results-dir results/synthetic --models-dir outputs/synthetic/models
"""
import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.utils import load_config, resolve

# (label, seconds): 0 = transient, 1 baseline, 2 stress, 3 amusement, 4 meditation
PROTOCOL = [(0, 200), (1, 1200), (0, 150), (2, 600), (0, 150), (3, 400), (4, 300), (0, 100)]


def subject_signals(rng: np.random.Generator):
    labels = np.concatenate([np.full(s, l) for l, s in PROTOCOL])          # 1 Hz
    n = len(labels)
    t4 = np.arange(n * 4) / 4
    lab4 = np.repeat(labels, 4)
    # EDA: subject level + stress shift + slow drift + SCRs (more frequent in stress)
    level = rng.uniform(1, 5) + rng.uniform(0.3, 1.5) * (lab4 == 2) + 0.2 * (lab4 == 3)
    eda = level + 0.3 * np.sin(2 * np.pi * t4 / rng.uniform(300, 900))
    rate = np.where(lab4 == 2, 5, np.where(lab4 == 3, 3, 2)) / 60 / 4
    kernel = np.exp(-np.arange(40) / 8)
    eda += np.convolve(rng.random(len(t4)) < rate, kernel)[: len(t4)] * rng.uniform(0.05, 0.3)
    eda += rng.normal(0, 0.02, len(t4))
    # BVP: pulse wave whose rate rises in stress
    lab64 = np.repeat(labels, 64)
    hr = rng.uniform(60, 80) + rng.uniform(5, 15) * (lab64 == 2) + 3 * (lab64 == 3)
    hr = hr + np.cumsum(rng.normal(0, 0.02, len(lab64)))
    phase = 2 * np.pi * np.cumsum(hr / 60) / 64
    amp = rng.uniform(30, 80)
    bvp = amp * (np.sin(phase) + 0.4 * np.sin(2 * phase + 1.0)) + rng.normal(0, 0.3 * amp, len(lab64))
    label700 = np.repeat(labels, 700).astype(np.int8)
    return eda[:, None], bvp[:, None], label700


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data/synthetic/raw/WESAD")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    out = resolve(args.out)
    rng = np.random.default_rng(args.seed)
    for sid in load_config()["data"]["subjects"]:
        eda, bvp, label = subject_signals(rng)
        (out / sid).mkdir(parents=True, exist_ok=True)
        with open(out / sid / f"{sid}.pkl", "wb") as f:
            pickle.dump({"subject": sid, "label": label,
                         "signal": {"wrist": {"EDA": eda, "BVP": bvp}}}, f)
    print(f"synthetic WESAD -> {out}")


if __name__ == "__main__":
    main()
