"""RQ3: is 0.5 : 0.5 the right prediction-fusion weight?

For each LOSO fold, w* is chosen only from the 14 training subjects: participant-level GroupKFold
(evaluation.inner.n_splits) gives out-of-fold p_EDA / p_BVP, and w* maximizes the mean per-subject
Macro-F1 over the inner-validation subjects. The held-out subject is then scored with 0.5 and with w*
using the outer fold's p_EDA / p_BVP from the common run. Requires scripts/run_common.py first.

Outputs
  results/predictions/rq3_<run>.csv     test windows with p (0.5) and p (w*)
  results/scores/rq3_weight_scores.csv  inner score of every w per fold
  results/splits/rq3_inner_<run>.csv    inner split per fold
  results/tables/table4_rq3.csv         results table 4
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.data.wesad import load_windows
from src.eval.loso import get_model, load_fold_model, pick_device, run_name, subject_index
from src.eval.metrics import improved_worse, macro_f1
from src.fusion.weights import choose_weight, score_weights
from src.models.common import FitContext
from src.utils import common_parser, config_from_args, resolve, set_seed


def main():
    p = common_parser(__doc__)
    p.add_argument("--model", default=None, help="default: rq3.model")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()
    cfg = config_from_args(args)
    model = args.model or cfg["rq3"]["model"]
    seed = args.seed if args.seed is not None else cfg["evaluation"]["seed"]
    run = run_name(model, seed)
    thr = cfg["evaluation"]["threshold"]
    reuse = cfg["rq3"]["inner_pretrain"] == "reuse_outer" and model == "m2"

    res = resolve(cfg["paths"]["results_dir"])
    common = pd.read_csv(res / "predictions" / f"{run}.csv")
    outer = common[common.condition.isin(["eda", "bvp"])].pivot_table(
        index=["subject", "window_id", "orig_label", "y_true"], columns="condition", values="prob").reset_index()

    data = load_windows(resolve(cfg["paths"]["processed_dir"]))
    subjects = cfg["data"]["subjects"]
    device = pick_device()
    rows, wscores, inner_log, test_preds = [], [], [], []
    for fold, test in enumerate(subjects):
        train = [s for s in subjects if s != test]
        idx = subject_index(data, train)
        encoders_from = load_fold_model(model, cfg, run, test, seed) if reuse else None
        oof = []
        gkf = GroupKFold(n_splits=cfg["evaluation"]["inner"]["n_splits"])
        for k, (a, b) in enumerate(gkf.split(idx, groups=data.subject[idx])):
            tr, va = idx[a], idx[b]
            inner_train = sorted(set(data.subject[tr]), key=subjects.index)
            s = seed * 1000 + fold * 10 + k
            set_seed(s)
            ctx = FitContext(resolve(cfg["paths"]["processed_dir"]), inner_train, s, device)
            kwargs = {"encoders_from": encoders_from} if reuse else {}
            m = get_model(model)(cfg, seed).fit(data, tr, parts=("eda", "bvp"), ctx=ctx, **kwargs)
            pr = m.predict(data.eda[va], data.bvp[va])
            oof.append(pd.DataFrame(dict(subject=data.subject[va], y_true=data.y[va],
                                         p_eda=pr["eda"], p_bvp=pr["bvp"])))
            inner_log.append(dict(fold=fold, test_subject=test, inner_fold=k,
                                  inner_train=" ".join(inner_train),
                                  inner_val=" ".join(sorted(set(data.subject[va]), key=subjects.index))))
        sc = score_weights(pd.concat(oof, ignore_index=True), cfg["rq3"]["w_grid"], thr)
        w = choose_weight(sc, cfg["rq3"]["tie_break"])
        wscores.append(sc.assign(fold=fold, test_subject=test))

        t = outer[outer.subject == test].copy()
        t["p_equal"] = 0.5 * t.eda + 0.5 * t.bvp
        t["p_wstar"] = w * t.eda + (1 - w) * t.bvp
        t["w_star"] = w
        test_preds.append(t)
        f_eq = macro_f1(t.y_true, (t.p_equal >= thr).astype(int))
        f_w = macro_f1(t.y_true, (t.p_wstar >= thr).astype(int))
        rows.append(dict(subject=test, w_star=w, f1_equal=f_eq, f1_wstar=f_w))
        print(f"[rq3 {run}] {test}: w*={w:.1f}  F1(0.5)={f_eq:.3f}  F1(w*)={f_w:.3f}")

    for sub in ("predictions", "scores", "splits", "tables"):
        (res / sub).mkdir(parents=True, exist_ok=True)
    pd.concat(test_preds).to_csv(res / "predictions" / f"rq3_{run}.csv", index=False)
    pd.concat(wscores).to_csv(res / "scores" / "rq3_weight_scores.csv", index=False)
    pd.DataFrame(inner_log).to_csv(res / "splits" / f"rq3_inner_{run}.csv", index=False)

    t4 = pd.DataFrame(rows).set_index("subject")
    d = t4.f1_wstar - t4.f1_equal
    table = t4.round(3).astype(object)
    table.loc["mean ± SD"] = [f"{t4[c].mean():.3f} ± {t4[c].std(ddof=1):.3f}" for c in t4.columns]
    table.loc["improved / worse / tie"] = ["—", "—", f"{improved_worse(d)} / {int((d == 0).sum())}"]
    table.to_csv(res / "tables" / "table4_rq3.csv")
    print(table.to_string())


if __name__ == "__main__":
    main()
