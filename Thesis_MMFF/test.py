from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.mmff_net import MMFFConfig, MMFF_Net_Advanced
from models.rgb import RGBClassifier, RGBStream_Base
from models.skeleton import SkeletonClassifier, SkeletonStream_STGCN
from utils.dataset import DatasetConfig, MMFFDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--val_data", type=str, default="")
    p.add_argument("--val_label", type=str, default="")
    p.add_argument("--images_dir", type=str, default="")
    p.add_argument("--dataset", type=str, default="ntu60", choices=["ntu60", "ut_mhad"])
    p.add_argument("--num_classes", type=int, default=60)
    p.add_argument("--stage", type=str, default="all", choices=["all", "fusion", "skeleton", "rgb"])
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--image_size", type=int, default=299)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_cm", type=str, default="")
    return p.parse_args()


def _resolve_checkpoint(path: str, stage: str) -> str:
    """Resolve a checkpoint path.

    Supports:
    - direct file path
    - directory path (pick a '*best.pt' file)
    - legacy names best.pt/last.pt mapped to '{stage}_best.pt'/'{stage}_last.pt'
    """

    if os.path.isdir(path):
        # Prefer stage-specific best
        cand = os.path.join(path, f"{stage}_best.pt")
        if os.path.exists(cand):
            return cand
        # Any best
        bests = [os.path.join(path, f) for f in os.listdir(path) if f.endswith("best.pt")]
        if bests:
            bests.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return bests[0]
        raise FileNotFoundError(f"No '*best.pt' found in checkpoint dir: {path}")

    if os.path.exists(path):
        return path

    # legacy mapping
    base = os.path.basename(path)
    parent = os.path.dirname(path) or "."
    if base in {"best.pt", "last.pt"}:
        mapped = os.path.join(parent, f"{stage}_{base}")
        if os.path.exists(mapped):
            return mapped

        # also try 'all_*' as common default
        mapped2 = os.path.join(parent, f"all_{base}")
        if os.path.exists(mapped2):
            return mapped2

    raise FileNotFoundError(f"Checkpoint not found: {path}")


@torch.no_grad()
def main() -> None:
    args = parse_args()

    ds = MMFFDataset(
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
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Build model for the requested stage
    if args.stage == "skeleton":
        sk_backbone = SkeletonStream_STGCN(dataset=args.dataset)
        model = SkeletonClassifier(sk_backbone, num_classes=args.num_classes)
        forward = lambda sk, rgb: model(sk)  # noqa: E731
    elif args.stage == "rgb":
        rgb_backbone = RGBStream_Base(pretrained=False)
        model = RGBClassifier(rgb_backbone, num_classes=args.num_classes)
        forward = lambda sk, rgb: model(rgb)  # noqa: E731
    else:
        cfg = MMFFConfig(dataset=args.dataset, num_classes=args.num_classes, rgb_pretrained=False)
        model = MMFF_Net_Advanced(cfg)
        forward = lambda sk, rgb: model(sk, rgb)  # noqa: E731

    device = args.device
    ckpt_path = _resolve_checkpoint(args.checkpoint, stage=args.stage)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)

    model = model.to(device)
    model.eval()

    all_true = []
    all_pred = []
    correct = 0
    total = 0

    for skeleton, rgb, label, _ in tqdm(loader, desc="test"):
        skeleton = skeleton.to(device)
        rgb = rgb.to(device)
        label = label.to(device)

        logits = forward(skeleton, rgb)
        pred = logits.argmax(dim=1)

        all_true.append(label.cpu().numpy())
        all_pred.append(pred.cpu().numpy())

        correct += (pred == label).sum().item()
        total += label.numel()

    acc = correct / max(total, 1)
    print(f"Accuracy: {acc*100:.2f}%")

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(args.num_classes)))

    if args.save_cm:
        os.makedirs(os.path.dirname(args.save_cm) or ".", exist_ok=True)
        plt.figure(figsize=(10, 10))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        plt.tight_layout()
        plt.ylabel("True")
        plt.xlabel("Pred")
        plt.savefig(args.save_cm, dpi=200)
        print(f"Saved confusion matrix to {args.save_cm}")


if __name__ == "__main__":
    main()
