"""1D-ResNet encoder shared by M1 and M2 (f_E for EDA, f_B for BVP)."""
from __future__ import annotations

import torch
from torch import nn


class BasicBlock1d(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int, k: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(c_in, c_out, k, stride, k // 2, bias=False), nn.BatchNorm1d(c_out), nn.ReLU(inplace=True),
            nn.Conv1d(c_out, c_out, k, 1, k // 2, bias=False), nn.BatchNorm1d(c_out),
        )
        self.skip = (nn.Identity() if stride == 1 and c_in == c_out else
                     nn.Sequential(nn.Conv1d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm1d(c_out)))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.body(x) + self.skip(x))


class ResNet1D(nn.Module):
    """(B, L) raw signal -> (B, emb_dim) after global average pooling."""

    def __init__(self, channels=(32, 64, 128), strides=(1, 2, 2), blocks_per_stage=1,
                 kernel_size=7, stem_stride=1, stem_pool=1):
        super().__init__()
        c0 = channels[0]
        stem = [nn.Conv1d(1, c0, kernel_size, stem_stride, kernel_size // 2, bias=False),
                nn.BatchNorm1d(c0), nn.ReLU(inplace=True)]
        if stem_pool > 1:
            stem.append(nn.MaxPool1d(stem_pool))
        self.stem = nn.Sequential(*stem)
        layers, c_in = [], c0
        for c, s in zip(channels, strides):
            for b in range(blocks_per_stage):
                layers.append(BasicBlock1d(c_in, c, s if b == 0 else 1, kernel_size))
                c_in = c
        self.layers = nn.Sequential(*layers)
        self.emb_dim = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.layers(self.stem(x.unsqueeze(1)))
        return h.mean(dim=-1)


def build_encoder(cfg: dict, modality: str) -> ResNet1D:
    e = cfg["encoder"]
    stem = e[f"{modality}_stem"]
    return ResNet1D(tuple(e["channels"]), tuple(e["strides"]), e["blocks_per_stage"],
                    e["kernel_size"], stem["stride"], stem["pool"])
