"""Build the common 60 s window index shared by every model, condition and RQ.

Outputs
  data/processed/windows.npz              EDA (N,240), BVP (N,3840), labels, subject, window_id
  data/processed/continuous/<S>.npz       whole wrist recording (label-free, for JEPA pre-training)
  results/windows/window_log.csv          every candidate window with keep / exclude reason
  results/windows/window_summary.csv      per-subject counts by original condition and exclusion reason
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data.wesad import load_subject, save_continuous, save_windows, window_subject
from src.utils import common_parser, config_from_args, resolve


def main():
    args = common_parser(__doc__).parse_args()
    cfg = config_from_args(args)
    raw = resolve(cfg["paths"]["raw_dir"])
    out = resolve(cfg["paths"]["processed_dir"])
    res = resolve(cfg["paths"]["results_dir"]) / "windows"
    (out / "continuous").mkdir(parents=True, exist_ok=True)
    res.mkdir(parents=True, exist_ok=True)
    fs_e, fs_b = cfg["data"]["signals"]["eda"]["fs"], cfg["data"]["signals"]["bvp"]["fs"]

    kept_all, log_all = [], []
    for sid in cfg["data"]["subjects"]:
        eda, bvp, label = load_subject(raw, sid)
        kept, log = window_subject(sid, eda, bvp, label, cfg)
        save_continuous(out, sid, eda, bvp, fs_e, fs_b)
        kept_all += kept
        log_all += log
        print(f"{sid}: kept {len(kept)} windows, excluded {sum(r['status'] == 'excluded' for r in log)}")

    save_windows(out / "windows.npz", kept_all)
    log_df = pd.DataFrame(log_all)
    log_df.to_csv(res / "window_log.csv", index=False)

    names = {int(k): v for k, v in cfg["labels"]["names"].items()}
    kept_df = log_df[log_df.status == "kept"].assign(cond=lambda d: d.orig_label.map(names))
    summary = pd.concat([
        kept_df.pivot_table(index="subject", columns="cond", values="window_id", aggfunc="count"),
        log_df[log_df.status == "excluded"].pivot_table(index="subject", columns="reason",
                                                         values="window_id", aggfunc="count"),
    ], axis=1).reindex(cfg["data"]["subjects"]).fillna(0).astype(int)
    summary.loc["total"] = summary.sum()
    summary.to_csv(res / "window_summary.csv")
    print(summary.to_string())
    print(f"\nwindows.npz: {len(kept_all)} windows -> {out}")


if __name__ == "__main__":
    main()
