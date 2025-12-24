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
from utils.dataset import DatasetConfig, MMFFDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--val_data", type=str, default="")
    p.add_argument("--val_label", type=str, default="")
    p.add_argument("--images_dir", type=str, default="")
    p.add_argument("--dataset", type=str, default="ntu60", choices=["ntu60", "ut_mhad"])
    p.add_argument("--num_classes", type=int, default=60)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--image_size", type=int, default=299)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_cm", type=str, default="")
    return p.parse_args()


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

    cfg = MMFFConfig(dataset=args.dataset, num_classes=args.num_classes)
    model = MMFF_Net_Advanced(cfg)

    device = args.device
    ckpt = torch.load(args.checkpoint, map_location=device)
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

        logits = model(skeleton, rgb)
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
