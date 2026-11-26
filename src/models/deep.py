"""Training helpers shared by M1 and M2, and M1 (1D-ResNet + Linear, supervised)."""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from src.models.common import FIT_PARTS, MODALITIES, Standardizer, add_prediction_fusion
from src.models.resnet import build_encoder


def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


class Net(nn.Module):
    """Encoders (one per modality) -> concat pooled embeddings -> Linear -> logit.

    With no encoders the input is an embedding matrix under key "h" (M2 heads).
    """

    def __init__(self, encoders: dict[str, nn.Module], in_dim: int):
        super().__init__()
        self.mods = list(encoders) or ["h"]
        self.encoders = nn.ModuleDict(encoders)
        self.head = nn.Linear(in_dim, 1)

    def forward(self, xs: dict) -> torch.Tensor:
        if len(self.encoders):
            h = torch.cat([self.encoders[m](xs[m]) for m in self.mods], dim=1)
        else:
            h = xs["h"]
        return self.head(h).squeeze(1)


def train_classifier(net: Net, inputs: dict, y: np.ndarray, p: dict, seed: int, device: str,
                     freeze_encoders: bool = False) -> Net:
    """BCE with pos_weight = n_neg / n_pos from the training labels. Fixed epochs, no early stopping."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    net.to(device)
    if freeze_encoders:
        for prm in net.encoders.parameters():
            prm.requires_grad = False
    opt = torch.optim.AdamW([q for q in net.parameters() if q.requires_grad],
                            lr=p["lr"], weight_decay=p["weight_decay"])
    pos = float(np.sum(y == 1))
    pos_weight = torch.tensor((len(y) - pos) / max(pos, 1.0), device=device)
    xs_all = {m: torch.from_numpy(np.ascontiguousarray(inputs[m])) for m in net.mods}
    y_all = torch.from_numpy(y.astype(np.float32))
    n, bs = len(y), p["batch_size"]
    for _ in range(p["epochs"]):
        net.train()
        if freeze_encoders:
            net.encoders.eval()          # keep BatchNorm statistics frozen too
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, bs):
            b = perm[i: i + bs]
            if len(b) < 2:
                continue                 # BatchNorm cannot train on a single sample
            logit = net({m: xs_all[m][b].to(device) for m in net.mods})
            loss = F.binary_cross_entropy_with_logits(logit, y_all[b].to(device), pos_weight=pos_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return net


@torch.no_grad()
def predict_proba(net: Net, inputs: dict, device: str, bs: int = 256) -> np.ndarray:
    net.eval().to(device)
    n = len(next(iter(inputs.values())))
    out = []
    for i in range(0, n, bs):
        xs = {m: torch.from_numpy(np.ascontiguousarray(inputs[m][i: i + bs])).to(device) for m in net.mods}
        out.append(torch.sigmoid(net(xs)).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def embed(encoder: nn.Module, x: np.ndarray, device: str, bs: int = 256) -> np.ndarray:
    encoder.eval().to(device)
    return np.concatenate([encoder(torch.from_numpy(np.ascontiguousarray(x[i: i + bs])).to(device)).cpu().numpy()
                           for i in range(0, len(x), bs)])


class M1:
    """Encoder and Linear trained jointly with labels.

    fusion_mode = joint : feature fusion trains both encoders + Linear on [h_E ; h_B]
    fusion_mode = frozen: feature fusion reuses the single-modality encoders (frozen), trains Linear only
    """
    name = "m1"

    def __init__(self, cfg: dict, seed: int):
        self.cfg, self.seed = cfg, seed
        self.p = cfg["m1"]
        self.norm: dict[str, Standardizer] = {}
        self.nets: dict[str, Net] = {}
        self.device = pick_device()

    def _net(self, part: str) -> Net:
        mods = list(MODALITIES) if part == "feature_fusion" else [part]
        encs = {m: build_encoder(self.cfg, m) for m in mods}
        return Net(encs, sum(e.emb_dim for e in encs.values()))

    def _inputs(self, eda, bvp) -> dict:
        return {"eda": self.norm["eda"].transform(eda), "bvp": self.norm["bvp"].transform(bvp)}

    def fit(self, data, idx, parts=FIT_PARTS, ctx=None):
        self.device = ctx.device if ctx else self.device
        self.norm = {"eda": Standardizer().fit(data.eda[idx]), "bvp": Standardizer().fit(data.bvp[idx])}
        X, y = self._inputs(data.eda[idx], data.bvp[idx]), data.y[idx]
        for k, part in enumerate(parts):
            seed = self.seed * 100 + k
            torch.manual_seed(seed)
            if part == "feature_fusion" and self.p["fusion_mode"] == "frozen":
                net = Net({m: copy.deepcopy(self.nets[m].encoders[m]) for m in MODALITIES},
                          2 * self.cfg["encoder"]["channels"][-1])
                self.nets[part] = train_classifier(net, X, y, self.p, seed, self.device, freeze_encoders=True)
            else:
                self.nets[part] = train_classifier(self._net(part), X, y, self.p, seed, self.device)
        return self

    def predict(self, eda, bvp) -> dict:
        X = self._inputs(eda, bvp)
        return add_prediction_fusion({p: predict_proba(n, X, self.device) for p, n in self.nets.items()})

    def save(self, path):
        torch.save({"norm": {m: s.state() for m, s in self.norm.items()},
                    "nets": {p: n.state_dict() for p, n in self.nets.items()}}, path.with_suffix(".pt"))

    def load(self, path):
        ck = torch.load(path.with_suffix(".pt"), map_location="cpu")
        self.norm = {m: Standardizer.from_state(s) for m, s in ck["norm"].items()}
        self.nets = {}
        for part, state in ck["nets"].items():
            net = self._net(part)
            net.load_state_dict(state)
            self.nets[part] = net
        return self
