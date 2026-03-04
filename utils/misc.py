from __future__ import annotations

import os
import random

import numpy as np
import torch
from PIL import Image


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # to save time
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def rgb_loader(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def append_date_suffix(path: str, date: str) -> str:
    base, ext = os.path.splitext(path)
    if ext == "":
        return f"{base}-{date}"
    return f"{base}-{date}{ext}"


def rand_bbox(H: int, W: int, lam: float) -> tuple[int, int, int, int]:
    # Return (y1, x1, y2, x2) for CutMix given image size and lambda.
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(round(H * cut_ratio))
    cut_w = int(round(W * cut_ratio))
    cy = np.random.randint(H)
    cx = np.random.randint(W)
    y1 = max(0, cy - cut_h // 2)
    x1 = max(0, cx - cut_w // 2)
    y2 = min(H, cy + cut_h // 2)
    x2 = min(W, cx + cut_w // 2)
    return y1, x1, y2, x2
