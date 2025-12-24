# Thesis_MMFF

PyTorch implementation of **MMFF** (Skeleton sequence + single RGB frame) based on the paper:

- *Skeleton Sequence and RGB Frame Based Multi-Modality Feature Fusion Network for Action Recognition* (Zhu et al.)

## Data format (expected)

```
Thesis_MMFF/
  data/
    train_data.npy
    train_label.pkl
    val_data.npy
    val_label.pkl
    images/
      video_name_1.jpg
      ...
```

- `*_data.npy`: skeleton tensor, typically shaped like `(N, C, T, V, M)`.
- `*_label.pkl`: supports common formats used in skeleton action repos:
  - dict with keys `sample_name` and `label`
  - tuple/list `(sample_name_list, label_list)`
  - list of `(sample_name, label)` pairs

## Install

`pip install -r requirements.txt`

## Train (example)

`python train.py --data_dir data --dataset ntu60 --stage all --epochs 60 --batch_size 32 --lr 1e-4`

## Test (example)

`python test.py --data_dir data --dataset ntu60 --checkpoint checkpoints/best.pt`
