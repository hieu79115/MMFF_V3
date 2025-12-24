from __future__ import annotations

import argparse
import os
import time
from dataclasses import asdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.mmff_net import MMFFConfig, MMFF_Net_Advanced
from models.rgb import RGBClassifier, RGBStream_Base
from models.skeleton import SkeletonClassifier, SkeletonStream_STGCN
from utils.dataset import DatasetConfig, MMFFDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--train_data", type=str, default="")
    p.add_argument("--train_label", type=str, default="")
    p.add_argument("--val_data", type=str, default="")
    p.add_argument("--val_label", type=str, default="")
    p.add_argument("--images_dir", type=str, default="")
    p.add_argument("--dataset", type=str, default="ntu60", choices=["ntu60", "ut_mhad"])
    p.add_argument("--num_classes", type=int, default=60)

    # Paper-style staged training
    p.add_argument("--init_skeleton_ckpt", type=str, default="")
    p.add_argument("--init_rgb_ckpt", type=str, default="")

    p.add_argument("--stage", type=str, default="all", choices=["skeleton", "rgb", "fusion", "all"])
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--step_size", type=int, default=10)
    p.add_argument("--gamma", type=float, default=0.1)

    p.add_argument("--image_size", type=int, default=299)

    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--resume", type=str, default="")

    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return p.parse_args()


def _load_ckpt_state(path: str, device: str) -> dict:
    ckpt = torch.load(path, map_location=device)
    return ckpt.get("model", ckpt)


def load_skeleton_into_mmff(mmff: nn.Module, skeleton_ckpt: str, device: str) -> None:
    """Load SkeletonClassifier checkpoint into mmff.skeleton_backbone.

    Expects keys like core.backbone.* (from stage=skeleton training).
    """

    state = _load_ckpt_state(skeleton_ckpt, device)
    remap = {}
    for k, v in state.items():
        if k.startswith("core.backbone."):
            remap[k.replace("core.backbone.", "skeleton_backbone.")] = v
        elif k.startswith("backbone."):
            remap[k.replace("backbone.", "skeleton_backbone.")] = v
    missing, unexpected = mmff.load_state_dict(remap, strict=False)
    # We intentionally ignore missing/unexpected since head layers differ
    _ = (missing, unexpected)


def load_rgb_into_mmff(mmff: nn.Module, rgb_ckpt: str, device: str) -> None:
    """Load RGBClassifier checkpoint into mmff.rgb_backbone.

    Expects keys like core.backbone.* (from stage=rgb training).
    """

    state = _load_ckpt_state(rgb_ckpt, device)
    remap = {}
    for k, v in state.items():
        if k.startswith("core.backbone."):
            remap[k.replace("core.backbone.", "rgb_backbone.")] = v
        elif k.startswith("backbone."):
            remap[k.replace("backbone.", "rgb_backbone.")] = v
    missing, unexpected = mmff.load_state_dict(remap, strict=False)
    _ = (missing, unexpected)


def set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    for skeleton, rgb, label, _ in loader:
        skeleton = skeleton.to(device)
        rgb = rgb.to(device)
        label = label.to(device)

        logits = model(skeleton, rgb)
        pred = logits.argmax(dim=1)
        correct += (pred == label).sum().item()
        total += label.numel()
    return correct / max(total, 1)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    criterion: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0
    for skeleton, rgb, label, _ in tqdm(loader, desc="train", leave=False):
        skeleton = skeleton.to(device)
        rgb = rgb.to(device)
        label = label.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(skeleton, rgb)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * label.size(0)
    return total_loss / max(len(loader.dataset), 1)


def save_ckpt(path: str, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_acc: float) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_acc": best_acc,
        },
        path,
    )


def build_mmff(args: argparse.Namespace) -> MMFF_Net_Advanced:
    cfg = MMFFConfig(dataset=args.dataset, num_classes=args.num_classes)
    model = MMFF_Net_Advanced(cfg)
    return model


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train_ds = MMFFDataset(
        DatasetConfig(
            data_dir=args.data_dir,
            split="train",
            data_path=(args.train_data or None),
            label_path=(args.train_label or None),
            images_dir=(args.images_dir or None),
            image_size=args.image_size,
            augment=True,
        )
    )
    val_ds = MMFFDataset(
        DatasetConfig(
            data_dir=args.data_dir,
            split="val",
            data_path=(args.val_data or None),
            label_path=(args.val_label or None),
            images_dir=(args.images_dir or None),
            image_size=args.image_size,
            augment=False,
        )
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(args.device.startswith("cuda")),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device.startswith("cuda")),
    )

    device = args.device

    # Build model according to stage
    if args.stage == "skeleton":
        sk_backbone = SkeletonStream_STGCN(dataset=args.dataset)
        model = SkeletonClassifier(sk_backbone, num_classes=args.num_classes)
        forward = lambda sk, rgb: model(sk)  # noqa: E731
    elif args.stage == "rgb":
        rgb_backbone = RGBStream_Base(pretrained=True)
        model = RGBClassifier(rgb_backbone, num_classes=args.num_classes)
        forward = lambda sk, rgb: model(rgb)  # noqa: E731
    else:
        model = build_mmff(args)

        # Optionally initialize backbones from pretrained single-stream checkpoints
        if args.init_skeleton_ckpt:
            load_skeleton_into_mmff(model, args.init_skeleton_ckpt, device)
        if args.init_rgb_ckpt:
            load_rgb_into_mmff(model, args.init_rgb_ckpt, device)
        forward = lambda sk, rgb: model(sk, rgb)  # noqa: E731

    model = model.to(device)

    # Wrap into a callable nn.Module for unified training loop
    class _Wrapper(nn.Module):
        def __init__(self, core: nn.Module):
            super().__init__()
            self.core = core

        def forward(self, sk, rgb):
            return forward(sk, rgb)

    wrapped = _Wrapper(model).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(wrapped.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    start_epoch = 0
    best_acc = 0.0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        wrapped.load_state_dict(ckpt["model"], strict=False)
        optimizer.load_state_dict(ckpt.get("optimizer", optimizer.state_dict()))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_acc = float(ckpt.get("best_acc", 0.0))

    # For fusion stage, freeze backbones (paper stage-2)
    if args.stage == "fusion":
        for n, p in wrapped.named_parameters():
            if n.startswith("core.skeleton_backbone") or n.startswith("core.rgb_backbone"):
                p.requires_grad = False

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        loss = train_one_epoch(wrapped, train_loader, optimizer, device, criterion)
        val_acc = evaluate(wrapped, val_loader, device)
        scheduler.step()

        is_best = val_acc > best_acc
        best_acc = max(best_acc, val_acc)

        # stage-specific ckpt names to support multi-stage workflow
        ckpt_path = os.path.join(args.checkpoint_dir, f"{args.stage}_last.pt")
        save_ckpt(ckpt_path, wrapped, optimizer, epoch, best_acc)
        if is_best:
            save_ckpt(os.path.join(args.checkpoint_dir, f"{args.stage}_best.pt"), wrapped, optimizer, epoch, best_acc)

        dt = time.time() - t0
        print(
            f"Epoch {epoch+1}/{args.epochs} | loss={loss:.4f} | val_acc={val_acc*100:.2f}% | best={best_acc*100:.2f}% | {dt:.1f}s"
        )


if __name__ == "__main__":
    main()
