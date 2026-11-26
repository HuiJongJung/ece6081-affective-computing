"""RQ4: is the fusion gain just an ensemble effect?

Every configuration uses two models of the same type and budget on the same windows:
  EDA+EDA = EDA #1 + EDA #2,  BVP+BVP = BVP #1 + BVP #2,  EDA+BVP = EDA #1 + BVP #1 (= prediction fusion).
#1 is the common run (evaluation.seed); #2 is a second run with rq4.second_seed (and bootstrap if set).
Members are never selected using the evaluation results.

  python scripts/run_common.py --model m2 --seed 43 [--bootstrap]
  python scripts/run_rq4.py

Outputs
  results/predictions/rq4_members_<model>.csv   member probabilities per window + training condition
  results/scores/rq4_per_subject.csv
  results/tables/table5_rq4.csv                 results table 5
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.eval.loso import run_name
from src.eval.metrics import improved_worse, macro_f1
from src.utils import common_parser, config_from_args, resolve


def main():
    p = common_parser(__doc__)
    p.add_argument("--model", default=None, help="default: rq4.model")
    args = p.parse_args()
    cfg = config_from_args(args)
    model = args.model or cfg["rq4"]["model"]
    thr = cfg["evaluation"]["threshold"]
    run_a = run_name(model, cfg["evaluation"]["seed"])
    run_b = run_name(model, cfg["rq4"]["second_seed"], cfg["rq4"]["bootstrap"])
    res = resolve(cfg["paths"]["results_dir"])
    fb = res / "predictions" / f"{run_b}.csv"
    if not fb.exists():
        sys.exit(f"missing {fb.name}: run  python scripts/run_common.py --model {model} "
                 f"--seed {cfg['rq4']['second_seed']}" + (" --bootstrap" if cfg["rq4"]["bootstrap"] else ""))

    key = ["subject", "window_id", "orig_label", "y_true"]
    wide = {}
    for tag, run in (("1", run_a), ("2", run_b)):
        d = pd.read_csv(res / "predictions" / f"{run}.csv")
        wide[tag] = d[d.condition.isin(["eda", "bvp"])].pivot_table(index=key, columns="condition", values="prob")
    m = wide["1"].add_suffix("_1").join(wide["2"].add_suffix("_2")).reset_index()
    m["run_1"], m["run_2"] = run_a, run_b
    m["seed_1"], m["seed_2"] = cfg["evaluation"]["seed"], cfg["rq4"]["second_seed"]
    m["bootstrap_2"] = cfg["rq4"]["bootstrap"]

    configs = {  # name -> member probability columns
        "EDA #1": ["eda_1"], "EDA #2": ["eda_2"], "EDA + EDA": ["eda_1", "eda_2"],
        "BVP #1": ["bvp_1"], "BVP #2": ["bvp_2"], "BVP + BVP": ["bvp_1", "bvp_2"],
        "EDA + BVP (prediction fusion)": ["eda_1", "bvp_1"],
    }
    per_subject = []
    for sid, g in m.groupby("subject", sort=False):
        for name, cols in configs.items():
            pred = (g[cols].mean(axis=1) >= thr).astype(int)
            dis = (float(((g[cols[0]] >= thr) != (g[cols[1]] >= thr)).mean()) if len(cols) == 2 else np.nan)
            per_subject.append(dict(subject=sid, config=name, macro_f1=macro_f1(g.y_true, pred), disagreement=dis))
    ps = pd.DataFrame(per_subject)

    for sub in ("predictions", "scores", "tables"):
        (res / sub).mkdir(parents=True, exist_ok=True)
    m.to_csv(res / "predictions" / f"rq4_members_{model}.csv", index=False)
    ps.to_csv(res / "scores" / "rq4_per_subject.csv", index=False)

    f1 = ps.pivot(index="subject", columns="config", values="macro_f1")
    ref = f1["EDA + BVP (prediction fusion)"]
    rows = []
    for name, cols in configs.items():
        s = ps[ps.config == name]
        rows.append({
            "config": name,
            "macro_f1 (mean ± SD)": f"{s.macro_f1.mean():.3f} ± {s.macro_f1.std(ddof=1):.3f}",
            "member disagreement": f"{s.disagreement.mean():.3f}" if len(cols) == 2 else "—",
            "EDA+BVP better / worse (subjects)": "—" if name.startswith("EDA + BVP") else improved_worse(ref - f1[name]),
        })
    table = pd.DataFrame(rows)
    table.to_csv(res / "tables" / "table5_rq4.csv", index=False)
    print(f"members: {run_a} and {run_b}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
