"""Common experiment: one model, four conditions (EDA, BVP, feature fusion, prediction fusion), LOSO 15-fold.

  python scripts/run_common.py --model m0
  python scripts/run_common.py --model m2 --seed 43 [--bootstrap]    # second RQ4 member

Outputs
  results/predictions/<run>.csv   one row per window x condition (prob, y_pred)
  results/splits/<run>.csv        LOSO split (test subject, training subjects)
  outputs/models/<run>/<S>.*      fold models, reused by RQ2 / RQ3
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.wesad import load_windows
from src.eval.loso import run_loso
from src.eval.metrics import subject_scores, summarize
from src.utils import common_parser, config_from_args, resolve


def main():
    p = common_parser(__doc__)
    p.add_argument("--model", required=True, choices=["m0", "m1", "m2"])
    p.add_argument("--seed", type=int, default=None, help="default: evaluation.seed")
    p.add_argument("--bootstrap", action="store_true", help="resample training subjects (RQ4 diversity)")
    args = p.parse_args()
    cfg = config_from_args(args)
    seed = args.seed if args.seed is not None else cfg["evaluation"]["seed"]

    data = load_windows(resolve(cfg["paths"]["processed_dir"]))
    t = time.time()
    pred, splits = run_loso(args.model, cfg, data, seed, args.bootstrap)
    run = pred.run.iloc[0]

    res = resolve(cfg["paths"]["results_dir"])
    for sub in ("predictions", "splits"):
        (res / sub).mkdir(parents=True, exist_ok=True)
    pred.to_csv(res / "predictions" / f"{run}.csv", index=False)
    splits.to_csv(res / "splits" / f"{run}.csv", index=False)
    print(summarize(subject_scores(pred)).to_string(index=False))
    print(f"done in {time.time() - t:.0f}s -> {res / 'predictions' / f'{run}.csv'}")


if __name__ == "__main__":
    main()
