from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from models.attention import SelfAttentionModule, SkeletonAttentionModule
from models.fusion import FusionConfig, FusionTransformer
from models.rgb import RGBStream_Base
from models.skeleton import SkeletonStream_STGCN


@dataclass
class MMFFConfig:
    dataset: str = "ntu60"
    num_classes: int = 60

    # backbones
    skeleton_out_channels: int = 256
    rgb_pretrained: bool = True

    # early fusion
    rgb_fused_channels: int = 256

    # late fusion
    fusion_proj_dim: int = 256
    fusion_hidden_dim: int = 512


class MMFF_Net_Advanced(nn.Module):
    """MMFF network (Skeleton sequence + single RGB frame) as in the paper.

    - Skeleton stream: ST-GCN
    - RGB stream: Xception + self-attention + skeleton-attention (early fusion)
    - Late fusion: relation-attention module (Fig.6 / Eq.16-17 inspired)
    """

    def __init__(self, cfg: MMFFConfig):
        super().__init__()
        self.cfg = cfg

        self.skeleton_backbone = SkeletonStream_STGCN(
            dataset=cfg.dataset,
            out_channels=cfg.skeleton_out_channels,
        )

        self.rgb_backbone = RGBStream_Base(pretrained=cfg.rgb_pretrained)
        rgb_c = self.rgb_backbone.out_channels

        self.self_attn = SelfAttentionModule(rgb_c)
        self.ske_attn = SkeletonAttentionModule()

        # concat F_self and F_ske, reduce to rgb_fused_channels
        self.rgb_reduce = nn.Sequential(
            nn.Conv2d(rgb_c * 2, cfg.rgb_fused_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(cfg.rgb_fused_channels),
            nn.ReLU(inplace=True),
        )

        fusion_cfg = FusionConfig(
            proj_dim=cfg.fusion_proj_dim,
            hidden_dim=cfg.fusion_hidden_dim,
            num_classes=cfg.num_classes,
        )
        self.fusion = FusionTransformer(
            c_s=cfg.skeleton_out_channels,
            c_r=cfg.rgb_fused_channels,
            cfg=fusion_cfg,
        )

    def forward(self, skeleton: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        # skeleton: (B,C,T,V,M)
        # rgb: (B,3,H,W)

        gcn_feat = self.skeleton_backbone(skeleton)  # (B,Cs,T',V)

        rgb_feat = self.rgb_backbone(rgb)  # (B,Cr,H',W')
        _, rgb_self = self.self_attn(rgb_feat)
        _, rgb_ske = self.ske_attn(skeleton, rgb_feat)

        rgb_fused = self.rgb_reduce(torch.cat([rgb_self, rgb_ske], dim=1))  # (B,Cf,H',W')

        logits = self.fusion(gcn_feat, rgb_fused)
        return logits
