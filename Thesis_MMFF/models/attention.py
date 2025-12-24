from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


class SelfAttentionModule(nn.Module):
    """Self-attention module (paper Eq.11-12, Fig.4).

    Given feature map F_in (B,C,H,W), produces a sigmoid mask M_self (B,1,H,W)
    and an attended feature map F_self = M_self ⊙ F_in.

    Note: The paper additionally uses two repeated branches to produce a 512-d embedding.
    For MMFF we primarily need the attended feature map for early fusion.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.mask_conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, feat_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = torch.sigmoid(self.mask_conv(feat_map))
        return mask, feat_map * mask


@dataclass
class SkeletonAttentionConfig:
    square_size: int = 7
    eps: float = 1e-6


class SkeletonAttentionModule(nn.Module):
    """Skeleton attention module (paper Eq.13-15, Fig.5).

    Paper projects skeleton joints to the RGB frame using camera parameters.
    In this repo, we may not have camera intrinsics/extrinsics. We therefore use a
    pragmatic alternative that works with common preprocessed skeletons:

    - Use x/y channels from the skeleton at the middle frame.
    - Normalize x/y to [0,1] per-sample using min/max over joints.
    - Map to feature-map coordinates (H,W).

    This keeps the key behavior: highlight the most-moving joint region.

    Inputs:
      - skeleton: (B,C,T,V,M)
      - feat_map: (B,Cf,H,W)

    Output:
      - mask: (B,1,H,W)
      - attended: (B,Cf,H,W)
    """

    def __init__(self, config: Optional[SkeletonAttentionConfig] = None):
        super().__init__()
        self.cfg = config or SkeletonAttentionConfig()

    @staticmethod
    def _pick_joint_index(skeleton: torch.Tensor) -> torch.Tensor:
        # skeleton: (B,C,T,V,M) -> use mean over persons
        sk = skeleton.float().mean(dim=-1)  # (B,C,T,V)
        B, C, T, V = sk.shape
        mid = T // 2
        # moving distance between frame0 and frame mid for each joint using xyz if available
        if C >= 3:
            p0 = sk[:, 0:3, 0, :]  # (B,3,V)
            pm = sk[:, 0:3, mid, :]
        else:
            p0 = sk[:, 0:2, 0, :]
            pm = sk[:, 0:2, mid, :]
        dist = torch.linalg.norm(p0 - pm, dim=1)  # (B,V)
        j_idx = dist.argmax(dim=1)  # (B,)
        return j_idx

    def forward(self, skeleton: torch.Tensor, feat_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if skeleton.dim() != 5:
            raise ValueError(f"skeleton must be (B,C,T,V,M), got {tuple(skeleton.shape)}")
        if feat_map.dim() != 4:
            raise ValueError(f"feat_map must be (B,C,H,W), got {tuple(feat_map.shape)}")

        B, _, H, W = feat_map.shape
        sk = skeleton.float().mean(dim=-1)  # (B,C,T,V)
        _, C, T, V = sk.shape
        mid = T // 2

        j_idx = self._pick_joint_index(skeleton)  # (B,)

        # Choose 2D coords from channels 0,1
        if C < 2:
            raise ValueError("Skeleton must have at least 2 channels (x,y) for skeleton attention.")

        xy = sk[:, 0:2, mid, :]  # (B,2,V)

        # Normalize to [0,1] per sample using min/max over joints
        min_xy = xy.amin(dim=2, keepdim=True)
        max_xy = xy.amax(dim=2, keepdim=True)
        xy_n = (xy - min_xy) / (max_xy - min_xy + self.cfg.eps)

        # Gather joint coord
        b_idx = torch.arange(B, device=feat_map.device)
        cx = xy_n[b_idx, 0, j_idx]
        cy = xy_n[b_idx, 1, j_idx]

        px = (cx * (W - 1)).round().long().clamp(0, W - 1)
        py = (cy * (H - 1)).round().long().clamp(0, H - 1)

        mask = torch.zeros((B, 1, H, W), device=feat_map.device, dtype=feat_map.dtype)
        half = self.cfg.square_size // 2
        for b in range(B):
            x0 = max(int(px[b]) - half, 0)
            x1 = min(int(px[b]) + half + 1, W)
            y0 = max(int(py[b]) - half, 0)
            y1 = min(int(py[b]) + half + 1, H)
            mask[b, 0, y0:y1, x0:x1] = 1.0

        attended = feat_map * mask
        return mask, attended
