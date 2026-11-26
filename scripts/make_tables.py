"""Tables from the common runs: results table 1 (common experiment) and table 2 (RQ1).

Reads results/predictions/m{0,1,2}_seed<seed>.csv (whichever exist).

Outputs
  results/scores/scores_per_subject.csv        participant x model x condition Macro-F1
  results/scores/fusion_diff.csv               fusion minus EDA / BVP per participant
  results/scores/error_table.csv               Baseline FP, Amusement FP, Stress miss (n / N) per model x condition
  results/scores/error_table_per_subject.csv   same, per participant
  results/tables/table1_common.csv             results table 1
  results/tables/table2_rq1.csv                results table 2 (RQ1)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.eval.loso import run_name
from src.eval.metrics import (error_overlap, error_table, fusion_diff, improved_worse, subject_scores,
                              summarize)
from src.models.common import CONDITIONS
from src.utils import common_parser, config_from_args, resolve


def fmt_pm(mean, sd):
    return f"{mean:.3f} ± {sd:.3f}"


def fmt_err(row, name):
    return f"{row[name + '_n']} / {row[name + '_N']} ({row[name + '_pct']:.1f}%)"


def main():
    p = common_parser(__doc__)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()
    cfg = config_from_args(args)
    seed = args.seed if args.seed is not None else cfg["evaluation"]["seed"]
    res = resolve(cfg["paths"]["results_dir"])
    for sub in ("scores", "tables"):
        (res / sub).mkdir(parents=True, exist_ok=True)

    frames = []
    for m in ("m0", "m1", "m2"):
        f = res / "predictions" / f"{run_name(m, seed)}.csv"
        if f.exists():
            frames.append(pd.read_csv(f))
        else:
            print(f"missing {f.name} (skip)")
    if not frames:
        sys.exit("no predictions found; run scripts/run_common.py first")
    pred = pd.concat(frames, ignore_index=True)
    order = {c: i for i, c in enumerate(CONDITIONS)}

    scores = subject_scores(pred)
    scores.to_csv(res / "scores" / "scores_per_subject.csv", index=False)
    fusion_diff(scores).to_csv(res / "scores" / "fusion_diff.csv", index=False)
    err = error_table(pred)
    err.to_csv(res / "scores" / "error_table.csv", index=False)
    error_table(pred, per_subject=True).to_csv(res / "scores" / "error_table_per_subject.csv", index=False)

    t1 = summarize(scores).merge(err, on=["model", "run", "condition"])
    t1 = t1.sort_values(["model", "condition"], key=lambda s: s.map(order) if s.name == "condition" else s)
    table1 = pd.DataFrame({
        "model": t1.model.str.upper(), "condition": t1.condition,
        "macro_f1 (mean ± SD)": [fmt_pm(a, b) for a, b in zip(t1.macro_f1_mean, t1.macro_f1_sd)],
        "baseline_fp n/N (%)": [fmt_err(r, "baseline_fp") for _, r in t1.iterrows()],
        "amusement_fp n/N (%)": [fmt_err(r, "amusement_fp") for _, r in t1.iterrows()],
        "stress_miss n/N (%)": [fmt_err(r, "stress_miss") for _, r in t1.iterrows()],
    })
    table1.to_csv(res / "tables" / "table1_common.csv", index=False)
    print(table1.to_string(index=False))

    # RQ1: per-participant F1 of the four conditions + EDA/BVP error overlap
    rq1_run = run_name(cfg["rq1"]["model"], seed)
    pr = pred[pred.run == rq1_run]
    if pr.empty:
        print(f"RQ1 skipped: {rq1_run} not found")
        return
    wide = scores[scores.run == rq1_run].pivot(index="subject", columns="condition", values="macro_f1")
    wide = wide.reindex(cfg["data"]["subjects"])[list(CONDITIONS)]
    ov = error_overlap(pr, cfg["rq1"]["overlap"], cfg["rq1"]["empty_union"]).set_index("subject")
    t2 = wide.join(ov)
    t2.to_csv(res / "scores" / "rq1_per_subject.csv")
    table2 = t2[list(CONDITIONS) + ["error_overlap"]].round(3).astype(object)
    table2.loc["mean ± SD"] = [fmt_pm(t2[c].mean(), t2[c].std(ddof=1)) for c in table2.columns]
    table2.loc["improved / worse"] = ["—", "—",
                                      f"vs EDA {improved_worse(t2.feature_fusion - t2.eda)}, "
                                      f"vs BVP {improved_worse(t2.feature_fusion - t2.bvp)}",
                                      f"vs EDA {improved_worse(t2.prediction_fusion - t2.eda)}, "
                                      f"vs BVP {improved_worse(t2.prediction_fusion - t2.bvp)}",
                                      "—"]
    table2.to_csv(res / "tables" / "table2_rq1.csv")
    print(table2.to_string())


if __name__ == "__main__":
    main()
