from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


@dataclass
class FusionConfig:
    proj_dim: int = 256
    hidden_dim: int = 512
    num_classes: int = 60


class FusionTransformer(nn.Module):
    """Late fusion module inspired by the paper's MMFF relation module (Eq.16-17, Fig.6).

    Inputs:
      - gcn_feat: (B, C_s, T, V)
      - rgb_feat_map: (B, C_r, H, W)

    Output:
      - logits: (B, num_classes)

    Implementation details:
      - Convert RGB to (B, C_r, S) with S=H*W
      - Convert GCN to (B, C_s, V) via max over T
      - Compute global vectors f_rgb, f_gcn by GAP
      - Build combined feature F_com: (B, C_s+C_r, S+V)
      - Build relation attention matrix A: (B, S+V, S+V)
      - Apply attention: F_rel = F_com @ A
      - Classify by 1x1 conv + GAP + 2 FC

    This keeps the "cross-spatial relation" idea from the paper while being shape-consistent.
    """

    def __init__(self, c_s: int, c_r: int, cfg: FusionConfig):
        super().__init__()
        self.cfg = cfg
        c = c_s + c_r

        self.q_proj = nn.Conv1d(c, cfg.proj_dim, kernel_size=1)
        self.k_proj = nn.Conv1d(c, cfg.proj_dim, kernel_size=1)
        self.v_proj = nn.Conv1d(c, c, kernel_size=1)

        self.post_conv = nn.Conv1d(c, c, kernel_size=1)
        self.fc1 = nn.Linear(c, cfg.hidden_dim)
        self.fc2 = nn.Linear(cfg.hidden_dim, cfg.num_classes)

    def _build_fcom(
        self, gcn_feat: torch.Tensor, rgb_feat_map: torch.Tensor
    ) -> Tuple[torch.Tensor, int, int]:
        # gcn_feat: (B,Cs,T,V) -> (B,Cs,V)
        gcn_spatial = gcn_feat.max(dim=2).values
        B, Cs, V = gcn_spatial.shape

        # rgb: (B,Cr,H,W) -> (B,Cr,S)
        B2, Cr, H, W = rgb_feat_map.shape
        assert B2 == B
        S = H * W
        rgb_spatial = rgb_feat_map.view(B, Cr, S)

        # global vectors
        f_gcn = gcn_spatial.mean(dim=2)  # (B,Cs)
        f_rgb = rgb_spatial.mean(dim=2)  # (B,Cr)

        # concat vectors to each spatial location
        gcn_to_rgb = f_gcn.unsqueeze(-1).expand(-1, -1, S)  # (B,Cs,S)
        rgb_aug = torch.cat([rgb_spatial, gcn_to_rgb], dim=1)  # (B,Cr+Cs,S)

        rgb_to_gcn = f_rgb.unsqueeze(-1).expand(-1, -1, V)  # (B,Cr,V)
        gcn_aug = torch.cat([gcn_spatial, rgb_to_gcn], dim=1)  # (B,Cs+Cr,V)

        f_com = torch.cat([rgb_aug, gcn_aug], dim=2)  # (B,C,S+V)
        return f_com, S, V

    def forward(self, gcn_feat: torch.Tensor, rgb_feat_map: torch.Tensor) -> torch.Tensor:
        f_com, _, _ = self._build_fcom(gcn_feat, rgb_feat_map)  # (B,C,L)

        q = self.q_proj(f_com)  # (B,d,L)
        k = self.k_proj(f_com)  # (B,d,L)
        v = self.v_proj(f_com)  # (B,C,L)

        # attention over L
        attn = torch.bmm(q.transpose(1, 2), k)  # (B,L,L)
        attn = torch.softmax(attn / (q.shape[1] ** 0.5), dim=-1)

        out = torch.bmm(v, attn)  # (B,C,L)
        out = self.post_conv(out)

        pooled = out.mean(dim=2)  # GAP over L -> (B,C)
        x = torch.relu(self.fc1(pooled))
        logits = self.fc2(x)
        return logits
