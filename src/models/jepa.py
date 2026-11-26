"""M2: cross-modal JEPA pre-training (no labels) -> frozen encoders + Linear.

Stage 1: online f_E, f_B + projectors p_E, p_B; predictors g_E->B, g_B->E predict the
         EMA target branch's latent of the other modality (stop-gradient).
         L = L_pred + mu * L_var + nu * L_cov  (VICReg terms on the online latents).
Stage 2: encoders frozen; h_E / h_B / [h_E ; h_B] -> Linear, trained with labels.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from src.data.wesad import load_continuous
from src.models.common import FIT_PARTS, MODALITIES, Standardizer, add_prediction_fusion
from src.models.deep import Net, embed, pick_device, predict_proba, train_classifier
from src.models.resnet import build_encoder


def mlp(d_in: int, d_hidden: int, d_out: int) -> nn.Module:
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.BatchNorm1d(d_hidden), nn.ReLU(inplace=True),
                         nn.Linear(d_hidden, d_out))


def var_loss(z: torch.Tensor) -> torch.Tensor:
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    return F.relu(1.0 - std).mean()


def cov_loss(z: torch.Tensor) -> torch.Tensor:
    z = z - z.mean(dim=0)
    c = (z.T @ z) / (len(z) - 1)
    off = c - torch.diag(torch.diag(c))
    return off.pow(2).sum() / z.shape[1]


class CrossModalJEPA(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        p = cfg["m2"]["pretrain"]
        emb = cfg["encoder"]["channels"][-1]
        self.enc = nn.ModuleDict({m: build_encoder(cfg, m) for m in MODALITIES})
        self.proj = nn.ModuleDict({m: mlp(emb, p["pred_hidden"], p["proj_dim"]) for m in MODALITIES})
        self.pred = nn.ModuleDict({"e2b": mlp(p["proj_dim"], p["pred_hidden"], p["proj_dim"]),
                                   "b2e": mlp(p["proj_dim"], p["pred_hidden"], p["proj_dim"])})
        self.t_enc = copy.deepcopy(self.enc)
        self.t_proj = copy.deepcopy(self.proj)
        for q in list(self.t_enc.parameters()) + list(self.t_proj.parameters()):
            q.requires_grad = False
        self.mu, self.nu, self.tau = p["var_weight"], p["cov_weight"], p["ema"]

    def loss(self, xe: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
        ze = self.proj["eda"](self.enc["eda"](xe))
        zb = self.proj["bvp"](self.enc["bvp"](xb))
        with torch.no_grad():
            te = self.t_proj["eda"](self.t_enc["eda"](xe))
            tb = self.t_proj["bvp"](self.t_enc["bvp"](xb))
        l_pred = F.mse_loss(self.pred["e2b"](ze), tb) + F.mse_loss(self.pred["b2e"](zb), te)
        l_reg = self.mu * (var_loss(ze) + var_loss(zb)) + self.nu * (cov_loss(ze) + cov_loss(zb))
        return l_pred + l_reg

    @torch.no_grad()
    def ema_update(self):
        for online, target in ((self.enc, self.t_enc), (self.proj, self.t_proj)):
            for q, t in zip(online.parameters(), target.parameters()):
                t.mul_(self.tau).add_(q.detach(), alpha=1 - self.tau)
            for b, tb in zip(online.buffers(), target.buffers()):
                tb.copy_(b)


class CropSampler:
    """Random aligned 60 s EDA/BVP crops from the training subjects' whole recordings (label-free)."""

    def __init__(self, processed_dir, subjects: list[str], n_per_subject: int, len_e: int = 240, ratio: int = 16):
        self.subjects = list(subjects)       # repeats allowed (bootstrap)
        self.signals = {s: load_continuous(processed_dir, s) for s in set(subjects)}
        self.n, self.len_e, self.ratio = n_per_subject, len_e, ratio

    def sample(self, rng: np.random.Generator):
        xe, xb = [], []
        for s in self.subjects:
            eda, bvp = self.signals[s]
            starts = rng.integers(0, len(eda) - self.len_e + 1, size=self.n)
            xe.extend(eda[a: a + self.len_e] for a in starts)
            xb.extend(bvp[a * self.ratio: (a + self.len_e) * self.ratio] for a in starts)
        return np.stack(xe), np.stack(xb)


def pretrain(model: CrossModalJEPA, cfg: dict, norm: dict, ctx, windows=None) -> CrossModalJEPA:
    p = cfg["m2"]["pretrain"]
    device = ctx.device
    torch.manual_seed(ctx.seed)
    rng = np.random.default_rng(ctx.seed)
    model.to(device)
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad],
                            lr=p["lr"], weight_decay=p["weight_decay"])
    sampler = (CropSampler(ctx.processed_dir, ctx.train_subjects, p["crops_per_subject"])
               if p["source"] == "continuous" else None)
    for _ in range(p["epochs"]):
        xe, xb = sampler.sample(rng) if sampler else windows
        xe, xb = norm["eda"].transform(xe), norm["bvp"].transform(xb)
        perm = rng.permutation(len(xe))
        model.train()
        for i in range(0, len(perm), p["batch_size"]):
            b = perm[i: i + p["batch_size"]]
            if len(b) < 2:
                continue
            loss = model.loss(torch.from_numpy(xe[b]).to(device), torch.from_numpy(xb[b]).to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            model.ema_update()
    return model


class HeadInput:
    """Standardize frozen embeddings with training-set statistics before the Linear head."""

    def fit(self, h):
        self.mean, self.std = h.mean(0), h.std(0) + 1e-6
        return self

    def transform(self, h):
        return ((h - self.mean) / self.std).astype(np.float32)


class M2:
    name = "m2"

    def __init__(self, cfg: dict, seed: int):
        self.cfg, self.seed = cfg, seed
        self.norm: dict[str, Standardizer] = {}
        self.enc: dict[str, nn.Module] = {}
        self.heads: dict[str, tuple[HeadInput, Net]] = {}
        self.device = pick_device()

    def _embed(self, eda, bvp) -> dict:
        he = embed(self.enc["eda"], self.norm["eda"].transform(eda), self.device)
        hb = embed(self.enc["bvp"], self.norm["bvp"].transform(bvp), self.device)
        return {"eda": he, "bvp": hb, "feature_fusion": np.hstack([he, hb])}

    def fit(self, data, idx, parts=FIT_PARTS, ctx=None, encoders_from: "M2 | None" = None):
        """encoders_from: reuse another fold model's pre-trained encoders (RQ3 inner_pretrain=reuse_outer)."""
        self.device = ctx.device
        if encoders_from is not None:
            self.norm, self.enc = encoders_from.norm, encoders_from.enc
        else:
            self.norm = {"eda": Standardizer().fit(data.eda[idx]), "bvp": Standardizer().fit(data.bvp[idx])}
            jepa = pretrain(CrossModalJEPA(self.cfg), self.cfg, self.norm, ctx,
                            windows=(data.eda[idx], data.bvp[idx]))
            self.enc = {m: jepa.enc[m].eval() for m in MODALITIES}     # online encoders, frozen from here on
        H, y = self._embed(data.eda[idx], data.bvp[idx]), data.y[idx]
        hp = self.cfg["m2"]["head"]
        for k, part in enumerate(parts):
            scaler = HeadInput().fit(H[part])
            net = Net({}, H[part].shape[1])
            self.heads[part] = (scaler, train_classifier(net, {"h": scaler.transform(H[part])}, y, hp,
                                                         self.seed * 100 + k, self.device))
        return self

    def predict(self, eda, bvp) -> dict:
        H = self._embed(eda, bvp)
        probs = {p: predict_proba(net, {"h": sc.transform(H[p])}, self.device)
                 for p, (sc, net) in self.heads.items()}
        return add_prediction_fusion(probs)

    def save(self, path):
        torch.save({"norm": {m: s.state() for m, s in self.norm.items()},
                    "enc": {m: e.state_dict() for m, e in self.enc.items()},
                    "heads": {p: ({"mean": sc.mean, "std": sc.std}, net.state_dict())
                              for p, (sc, net) in self.heads.items()}}, path.with_suffix(".pt"))

    def load(self, path):
        ck = torch.load(path.with_suffix(".pt"), map_location="cpu", weights_only=False)
        self.norm = {m: Standardizer.from_state(s) for m, s in ck["norm"].items()}
        self.enc = {}
        for m, state in ck["enc"].items():
            e = build_encoder(self.cfg, m)
            e.load_state_dict(state)
            self.enc[m] = e.eval()
        self.heads = {}
        for p, (sc_state, state) in ck["heads"].items():
            sc = HeadInput()
            sc.mean, sc.std = sc_state["mean"], sc_state["std"]
            net = Net({}, len(sc.mean))
            net.load_state_dict(state)
            self.heads[p] = (sc, net)
        return self
