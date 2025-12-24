from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.graph import get_adjacency_matrix


class GraphConv(nn.Module):
    """Spatial graph convolution used by ST-GCN.

    Input:  x (B, C_in, T, V)
    Output: y (B, C_out, T, V)

    A: adjacency matrix with shape (K, V, V)
    """

    def __init__(self, in_channels: int, out_channels: int, K: int, bias: bool = True):
        super().__init__()
        self.K = K
        self.conv = nn.Conv2d(in_channels, out_channels * K, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        assert A.dim() == 3, "A must be (K,V,V)"
        B, C, T, V = x.shape
        x = self.conv(x)  # (B, K*C_out, T, V)
        x = x.view(B, self.K, -1, T, V)  # (B, K, C_out, T, V)
        # einsum over V: sum_v x[..., v] * A[..., v, w]
        x = torch.einsum("bkctv,kvw->bctw", x, A)
        return x.contiguous()


class STGCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A_shape_K: int,
        stride: int = 1,
        residual: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.gcn = GraphConv(in_channels, out_channels, K=A_shape_K)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(9, 1),
                padding=(4, 0),
                stride=(stride, 1),
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        x = self.gcn(x, A)
        x = self.tcn(x)
        x = x + res
        return self.relu(x)


class SkeletonStream_STGCN(nn.Module):
    """Skeleton stream backbone (ST-GCN) producing a feature tensor for fusion.

    Expected input skeleton: (B, C, T, V, M)
      - C: coordinate channels (typically 3: x,y,z)
      - V: joints
      - M: persons (typically 1 or 2)

    Output:
      - feature map F_gcn: (B, C_s, T', V) after pooling over persons.
    """

    def __init__(
        self,
        dataset: str = "ntu60",
        in_channels: int = 3,
        base_channels: int = 64,
        dropout: float = 0.5,
        out_channels: int = 256,
    ):
        super().__init__()

        A = get_adjacency_matrix(dataset)
        self.register_buffer("A", torch.from_numpy(A), persistent=False)
        K, V, _ = A.shape

        self.data_bn = nn.BatchNorm1d(in_channels * V)

        # A standard ST-GCN stack (lightweight but effective)
        self.layers = nn.ModuleList(
            [
                STGCNBlock(in_channels, base_channels, K, residual=False, dropout=dropout),
                STGCNBlock(base_channels, base_channels, K, dropout=dropout),
                STGCNBlock(base_channels, base_channels, K, dropout=dropout),
                STGCNBlock(base_channels, base_channels * 2, K, stride=2, dropout=dropout),
                STGCNBlock(base_channels * 2, base_channels * 2, K, dropout=dropout),
                STGCNBlock(base_channels * 2, base_channels * 2, K, dropout=dropout),
                STGCNBlock(base_channels * 2, out_channels, K, stride=2, dropout=dropout),
                STGCNBlock(out_channels, out_channels, K, dropout=dropout),
                STGCNBlock(out_channels, out_channels, K, dropout=dropout),
            ]
        )

        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,T,V,M)
        if x.dim() != 5:
            raise ValueError(f"Skeleton input must be (B,C,T,V,M), got {tuple(x.shape)}")

        B, C, T, V, M = x.shape

        # pool persons early by sum (paper uses skeleton sequence; common to sum/mean persons)
        x = x.float().permute(0, 4, 1, 2, 3).contiguous()  # (B,M,C,T,V)
        x = x.view(B * M, C, T, V)

        # data bn over (C*V)
        x_bn = x.permute(0, 2, 3, 1).contiguous().view(B * M, T, V * C)
        x_bn = x_bn.permute(0, 2, 1).contiguous()  # (B*M, V*C, T)
        x_bn = self.data_bn(x_bn)
        x = x_bn.permute(0, 2, 1).contiguous().view(B * M, T, V, C)
        x = x.permute(0, 3, 1, 2).contiguous()  # (B*M, C, T, V)

        A = self.A
        for layer in self.layers:
            x = layer(x, A)

        # reshape back and pool persons
        _, C_out, T_out, V_out = x.shape
        x = x.view(B, M, C_out, T_out, V_out)
        x = x.mean(dim=1)  # (B, C_out, T_out, V)
        return x


class SkeletonClassifier(nn.Module):
    """Skeleton-only classifier head (for stage-1 pretraining)."""

    def __init__(self, backbone: SkeletonStream_STGCN, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Linear(backbone.out_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)  # (B,C,T,V)
        feat = feat.mean(dim=[2, 3])  # GAP over T,V -> (B,C)
        return self.fc(feat)
