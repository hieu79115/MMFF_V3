from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _load_label_pkl(path: str) -> Tuple[List[str], List[int]]:
    with open(path, "rb") as f:
        obj = pickle.load(f)

    # Common skeleton-action formats
    if isinstance(obj, dict) and "sample_name" in obj and "label" in obj:
        names = list(obj["sample_name"])
        labels = list(obj["label"])
        return names, labels

    if isinstance(obj, (tuple, list)) and len(obj) == 2:
        names, labels = obj
        return list(names), list(labels)

    if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], (tuple, list)) and len(obj[0]) == 2:
        names = [x[0] for x in obj]
        labels = [int(x[1]) for x in obj]
        return names, labels

    raise ValueError(
        "Unsupported label.pkl format. Expected dict{sample_name,label} or (names,labels) or list[(name,label)]."
    )


def _ensure_skeleton_layout(x: np.ndarray) -> np.ndarray:
    """Convert skeleton array to (N,C,T,V,M)."""

    if x.ndim == 5:
        # Heuristics to detect order
        # Candidate A: (N,C,T,V,M)
        if x.shape[1] in (2, 3) and x.shape[4] in (1, 2, 3):
            return x
        # Candidate B: (N,T,V,C,M)
        if x.shape[3] in (2, 3) and x.shape[4] in (1, 2, 3):
            return np.transpose(x, (0, 3, 1, 2, 4))
        # Candidate C: (N,M,C,T,V)
        if x.shape[2] in (2, 3):
            return np.transpose(x, (0, 2, 3, 4, 1))

    if x.ndim == 4:
        # (N,T,V,C) -> add M=1
        if x.shape[-1] in (2, 3):
            x = np.transpose(x, (0, 3, 1, 2))
            return x[..., None]
        # (N,C,T,V) -> add M=1
        if x.shape[1] in (2, 3):
            return x[..., None]

    raise ValueError(f"Unsupported skeleton npy shape {x.shape}. Expected 4D or 5D array.")


def _augment_skeleton_rot_scale(skeleton: np.ndarray) -> np.ndarray:
    """Rotation (x,y) in [0,30] degrees + scale (x,y in [1,1.2]) as in the paper.

    skeleton: (C,T,V,M)
    Returns same shape.
    """

    if skeleton.ndim != 4:
        return skeleton
    C, T, V, M = skeleton.shape
    if C < 3:
        return skeleton

    # Random rotations
    alpha = np.deg2rad(np.random.uniform(0.0, 30.0))
    beta = np.deg2rad(np.random.uniform(0.0, 30.0))
    gamma = 0.0

    Rx = np.array(
        [[1, 0, 0], [0, np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]],
        dtype=np.float32,
    )
    Ry = np.array(
        [[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]],
        dtype=np.float32,
    )
    Rz = np.array(
        [[np.cos(gamma), -np.sin(gamma), 0], [np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]],
        dtype=np.float32,
    )
    R = (Rz @ Ry @ Rx).astype(np.float32)

    sx = np.random.uniform(1.0, 1.2)
    sy = np.random.uniform(1.0, 1.2)
    sz = 1.0
    S = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, sz]], dtype=np.float32)

    X = skeleton[0:3].copy()  # (3,T,V,M)
    X = X.reshape(3, -1)  # (3, T*V*M)
    X = (S @ (R @ X)).astype(np.float32)
    X = X.reshape(3, T, V, M)
    skeleton = skeleton.copy()
    skeleton[0:3] = X
    return skeleton


def _find_file_recursive(root: str, filename: str) -> Optional[str]:
    for dirpath, _, files in os.walk(root):
        if filename in files:
            return os.path.join(dirpath, filename)
    return None


@dataclass
class DatasetConfig:
    data_dir: str
    split: str  # train|val
    data_path: Optional[str] = None
    label_path: Optional[str] = None
    images_dir: Optional[str] = None
    images_dirname: str = "images"
    image_ext: str = ".jpg"
    image_size: int = 299
    rgb_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    rgb_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    augment: bool = False


class MMFFDataset(Dataset):
    """Dataset that loads skeleton npy + label pkl + single RGB frame image.

    Expected file names inside data_dir:
      - train_data.npy / train_label.pkl
      - val_data.npy / val_label.pkl
      - images/{sample_name}.jpg

    Returns:
      - skeleton: torch.FloatTensor (C,T,V,M)
      - rgb: torch.FloatTensor (3,H,W)
      - label: int
      - name: str
    """

    def __init__(self, cfg: DatasetConfig):
        super().__init__()
        self.cfg = cfg

        if cfg.split not in {"train", "val", "test"}:
            raise ValueError("split must be train|val|test")

        default_data = os.path.join(cfg.data_dir, f"{cfg.split}_data.npy")
        default_label = os.path.join(cfg.data_dir, f"{cfg.split}_label.pkl")

        data_path = cfg.data_path or default_data
        label_path = cfg.label_path or default_label

        # images directory can be explicitly provided, otherwise default to {data_dir}/images
        self.images_dir = cfg.images_dir or os.path.join(cfg.data_dir, cfg.images_dirname)

        # Auto-discovery: if expected default file doesn't exist, try searching inside data_dir
        if not os.path.exists(data_path) and cfg.data_path is None:
            found = _find_file_recursive(cfg.data_dir, os.path.basename(default_data))
            if found:
                data_path = found

        if not os.path.exists(label_path) and cfg.label_path is None:
            found = _find_file_recursive(cfg.data_dir, os.path.basename(default_label))
            if found:
                label_path = found

        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Missing skeleton npy: '{data_path}'. "
                f"Pass correct --data_dir, or set an explicit path (e.g. --train_data /kaggle/input/.../train_data.npy)."
            )
        if not os.path.exists(label_path):
            raise FileNotFoundError(
                f"Missing label pkl: '{label_path}'. "
                f"Pass correct --data_dir, or set an explicit path (e.g. --train_label /kaggle/input/.../train_label.pkl)."
            )

        self._data = _ensure_skeleton_layout(np.load(data_path, mmap_mode=None))
        self.names, self.labels = _load_label_pkl(label_path)

        if len(self._data) != len(self.labels):
            raise ValueError(f"Mismatch: data has {len(self._data)} samples but label has {len(self.labels)}")

        from torchvision import transforms

        t = [
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.rgb_mean, std=cfg.rgb_std),
        ]
        self.rgb_tf = transforms.Compose(t)

    def __len__(self) -> int:
        return len(self.labels)

    def _load_image(self, name: str) -> Image.Image:
        # Names might include extension already
        if name.lower().endswith((".jpg", ".jpeg", ".png")):
            filename = name
        else:
            filename = name + self.cfg.image_ext
        path = os.path.join(self.images_dir, filename)
        if not os.path.exists(path):
            # fallback: try without any extension change
            alt = os.path.join(self.images_dir, name)
            if os.path.exists(alt):
                path = alt
            else:
                raise FileNotFoundError(f"Image not found for sample '{name}': {path}")
        return Image.open(path).convert("RGB")

    def __getitem__(self, idx: int):
        sk = self._data[idx]  # (C,T,V,M)
        if self.cfg.augment:
            sk = _augment_skeleton_rot_scale(sk)
        name = self.names[idx]
        label = int(self.labels[idx])

        img = self._load_image(name)
        rgb = self.rgb_tf(img)

        skeleton = torch.from_numpy(sk).float()
        return skeleton, rgb, label, name
