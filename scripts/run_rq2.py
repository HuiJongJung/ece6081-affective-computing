"""RQ2: corrupt test-time raw BVP at two SNR levels and re-infer with the saved fold models (no retraining).

EDA input, windows and the LOSO split stay the same. BVP features / representations are recomputed
from the corrupted signal. Requires scripts/run_common.py for the same model first.

Outputs
  results/predictions/rq2_<run>.csv     window x condition x noise level
  results/scores/rq2_per_subject.csv
  results/tables/table3_rq2.csv         results table 3
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.data.wesad import load_windows
from src.eval.loso import load_fold_model, prediction_frame, run_name
from src.eval.metrics import subject_scores, summarize
from src.eval.noise import corrupt_bvp
from src.models.common import CONDITIONS
from src.utils import common_parser, config_from_args, resolve, set_seed


def main():
    p = common_parser(__doc__)
    p.add_argument("--model", default=None, help="default: rq2.model")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()
    cfg = config_from_args(args)
    model = args.model or cfg["rq2"]["model"]
    seed = args.seed if args.seed is not None else cfg["evaluation"]["seed"]
    run = run_name(model, seed)
    noise = cfg["rq2"]["noise"]
    levels = [("clean", None)] + [(k, float(v)) for k, v in noise["levels"].items()]

    data = load_windows(resolve(cfg["paths"]["processed_dir"]))
    frames = []
    for test in cfg["data"]["subjects"]:
        set_seed(seed)
        m = load_fold_model(model, cfg, run, test, seed)
        te = np.flatnonzero(data.subject == test)
        for level, snr in levels:
            bvp = data.bvp[te] if snr is None else corrupt_bvp(
                data.bvp[te], data.subject[te], data.window_id[te], snr, noise["type"], noise["seed"])
            f = prediction_frame(data, te, m.predict(data.eda[te], bvp), model, run,
                                 cfg["evaluation"]["threshold"])
            f["noise_level"], f["snr_db"] = level, snr
            frames.append(f)
        print(f"[rq2 {run}] {test} done")
    pred = pd.concat(frames, ignore_index=True)

    res = resolve(cfg["paths"]["results_dir"])
    for sub in ("predictions", "scores", "tables"):
        (res / sub).mkdir(parents=True, exist_ok=True)
    pred.to_csv(res / "predictions" / f"rq2_{run}.csv", index=False)
    scores = subject_scores(pred, keys=("model", "run", "noise_level", "condition"))
    scores.to_csv(res / "scores" / "rq2_per_subject.csv", index=False)

    summ = summarize(scores, keys=("noise_level", "condition"))
    summ["cell"] = [f"{a:.3f} ± {b:.3f}" for a, b in zip(summ.macro_f1_mean, summ.macro_f1_sd)]
    table = summ.pivot(index="condition", columns="noise_level", values="cell")
    table = table.reindex(list(CONDITIONS))[[lv for lv, _ in levels]]
    table.columns = ["clean"] + [f"{lv} (SNR = {snr:g} dB)" for lv, snr in levels[1:]]
    table.to_csv(res / "tables" / "table3_rq2.csv")
    print(f"noise: {noise['type']}, seed {noise['seed']}, applied to raw test BVP only, no retraining")
    print(table.to_string())


if __name__ == "__main__":
    main()
