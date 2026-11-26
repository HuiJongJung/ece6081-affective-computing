"""Config loading, path handling and seeding shared by all scripts."""
from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(overrides: list[str] | None = None, paths: dict | None = None) -> dict:
    """Merge configs/common.yaml and configs/experiments.yaml, then apply overrides.

    overrides: ["m2.pretrain.epochs=5", ...]; values are parsed as YAML.
    paths: {"processed_dir": ..., "results_dir": ...} replaces cfg["paths"] entries.
    """
    with open(ROOT / "configs" / "common.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(ROOT / "configs" / "experiments.yaml", encoding="utf-8") as f:
        cfg.update(yaml.safe_load(f))
    for item in overrides or []:
        key, _, value = item.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(value)
    for k, v in (paths or {}).items():
        if v is not None:
            cfg["paths"][k] = v
    return cfg


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def common_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="config override, e.g. m2.pretrain.epochs=5")
    p.add_argument("--raw-dir", default=None)
    p.add_argument("--processed-dir", default=None)
    p.add_argument("--results-dir", default=None)
    p.add_argument("--models-dir", default=None)
    return p


def config_from_args(args) -> dict:
    return load_config(args.set, {
        "raw_dir": args.raw_dir,
        "processed_dir": args.processed_dir,
        "results_dir": args.results_dir,
        "models_dir": args.models_dir,
    })


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def deepcopy_cfg(cfg: dict) -> dict:
    return copy.deepcopy(cfg)
