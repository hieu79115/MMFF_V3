from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class RGBStream_Base(nn.Module):
    """RGB stream backbone using Xception as in the paper.

    Uses `timm` to load Xception (optionally pretrained).

    Output:
      - feature map (B, C, H, W)

    Note: Xception isn't in torchvision by default; timm is the standard source.
    """

    def __init__(
        self,
        pretrained: bool = True,
        out_indices: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__()

        try:
            import timm
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "timm is required for Xception backbone. Install with `pip install timm`."
            ) from e

        # features_only returns a list of feature maps from various stages
        self.model = timm.create_model(
            "xception",
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices or (4,),
        )

        info = self.model.feature_info.get_dicts()
        self.out_channels = info[-1]["num_chs"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.model(x)
        return feats[-1]


class RGBClassifier(nn.Module):
    """RGB-only classifier head (for stage-1 pretraining)."""

    def __init__(self, backbone: RGBStream_Base, num_classes: int, proj_dim: int = 512):
        super().__init__()
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone.out_channels, proj_dim),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(proj_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_map = self.backbone(x)
        feat = self.pool(feat_map)
        feat = self.proj(feat)
        return self.fc(feat)
