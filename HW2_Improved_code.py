"""
Install packages:
pip install -r requirements.txt

Run code:
python HW2_Improved_code.py

Run with a specific config:
python HW2_Improved_code.py --config config.json
"""

# Imports (same as baseline code)
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.datasets import DatasetFolder
from tqdm.auto import tqdm

# Our Imports (no extra third-party packages)
import random
import torch.nn.functional as F
from torch.utils.data import Dataset
import argparse
import json
import logging
import sys
from datetime import datetime
import copy
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn


DEFAULT_CONFIG = {
    "seed": 42,
    "data": {
        "train_labeled": "food11/training/labeled",
        "train_unlabeled": "food11/training/unlabeled",
        "valid": "food11/validation",
        "test": "food11/testing",
    },
    "dataloader": {
        "batch_size": 128,
        "num_workers": 0,
        "pin_memory": True,
    },
    "image": {
        "img_size": 128,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
    "augment": {
        "random_resized_crop_scale": [0.7, 1.0],
        "random_resized_crop_ratio": [0.9, 1.1],
        "horizontal_flip_p": 0.5,
        "rotation_deg": 15,
        "color_jitter": {
            "brightness": 0.2,
            "contrast": 0.2,
            "saturation": 0.2,
            "hue": 0.05,
        },
        "random_erasing": {
            "p": 0.25,
            "scale": [0.02, 0.2],
        },
    },
    "train": {
        "n_epochs": 25,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "label_smoothing": 0.1,
        "accum_steps": 1,
        "dropout": 0.5,
        "mix": {
            "enabled": False,
            "alpha": 0.2,
            "mode": "mixup" # mode can be mixup or cutmix
        }
    },
    "tta": {
        "enabled": False,
        "num_augments": 5,
    },
    "swa": {
        "enabled": False,
        "start_epoch_ratio": 0.9,
        "lr": 1e-5,
    },
    "semi": {
        "enabled": True,
        "warmup_epochs": 5,
        "pseudo_threshold": 0.95,
        "unsup_batch_size": 128,
        "lambda_u": 1.0,
        "randaugment_num_ops": 2,
        "randaugment_magnitude": 9,
        "ema": {
          "decay": 0.999
        }
    },
    "output": {
        "best_path": "best-model.pt",
        "predict_path": "predict.csv",
    },
}


def load_config(config_path: str):
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    with open(config_path, "r") as f:
        loaded = json.load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError("config json must be an object/dict")
    return _deep_update(cfg, loaded)

def _deep_update(base, updates):
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base



def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # to save time
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def rgb_loader(path: str):
    return Image.open(path).convert("RGB")


def _append_date_suffix(path: str, date: str) -> str:
    base, ext = os.path.splitext(path)
    if ext == "":
        return f"{base}-{date}"
    return f"{base}-{date}{ext}"


def _build_logger(date: str) -> logging.Logger:
    logger = logging.getLogger("hw2")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)

    fh = logging.FileHandler(f"log-{date}.txt", mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    logger.addHandler(sh)
    logger.addHandler(fh)
    logger.propagate = False
    return logger

# ===========
# CIFAR-style ResNet-18 (BasicBlock) from scratch
# - 3x3 stem, stride=1
# - NO maxpool
# - NO pretrained weights
# ===========

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class SEBlock(nn.Module):
    # Squeeze-and-Excitation block
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class ResNetBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, se_reduction=16):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SEBlock(planes * self.expansion, reduction=se_reduction)

        self.downsample = None
        if stride != 1 or in_planes != planes * self.expansion:
            self.downsample = nn.Sequential(
                conv1x1(in_planes, planes * self.expansion, stride=stride),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = F.relu(out, inplace=True)
        return out


class ResNet(nn.Module):
    """
    CIFAR-style stem:
      - conv3x3 stride=1
      - no maxpool

    For 128x128 input:
      - after stem: 128x128
      - layer2 stride2 -> 64x64
      - layer3 stride2 -> 32x32
      - layer4 stride2 -> 16x16
      - avgpool -> 1x1
    """
    def __init__(self, block, layers, num_classes=11, base_width=64, dropout=0.5):
        super().__init__()
        self.inplanes = base_width

        # CIFAR-style stem
        self.conv1 = nn.Conv2d(3, base_width, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_width)
        self.relu = nn.ReLU(inplace=True)

        # Stages (same 2-2-2-2 for ResNet18)
        self.layer1 = self._make_layer(block, base_width,     layers[0], stride=1)
        self.layer2 = self._make_layer(block, base_width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, base_width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, base_width * 8, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(p=dropout)
        self.fc = nn.Linear(base_width * 8 * block.expansion, num_classes)

        # Kaiming init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _make_layer(self, block, planes, blocks, stride):
        layers = []
        layers.append(block(self.inplanes, planes, stride=stride))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # head
        x = self.avgpool(x).flatten(1)
        x = self.drop(x)
        x = self.fc(x)
        return x


def resnet18(num_classes=11, dropout=0.5):
    return ResNet(ResNetBasicBlock, [2, 2, 2, 2], num_classes=num_classes, base_width=64, dropout=dropout)



class EMA:
    """
    Exponential Moving Average teacher.
    teacher_params = decay * teacher_params + (1-decay) * student_params
    """
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        msd = model.state_dict()
        esd = self.ema.state_dict()
        for k, v in esd.items():
            if k in msd:
                model_v = msd[k].detach()
                if torch.is_floating_point(v):
                    v.mul_(d).add_(model_v, alpha=1.0 - d)
                else:
                    v.copy_(model_v)

    def to(self, device):
        self.ema.to(device)
        return self


class UnlabeledPairDataset(Dataset):
    """
    Return (x_weak, x_strong) for the same image path.
    """
    def __init__(self, root: str, weak_tfm, strong_tfm):
        super().__init__()
        self.root = root
        self.weak_tfm = weak_tfm
        self.strong_tfm = strong_tfm

        tmp = DatasetFolder(
            root,
            loader=rgb_loader,
            extensions=("jpg", "jpeg", "png"),
            transform=None,
        )
        # tmp.samples: List[(path, target_dummy)]
        self.paths = [p for (p, _) in tmp.samples]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        xw = self.weak_tfm(img) if self.weak_tfm is not None else img
        xs = self.strong_tfm(img) if self.strong_tfm is not None else img
        return xw, xs


def rand_bbox(H, W, lam):
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


def train_one_epoch(
    model,
    ema: EMA,
    labeled_loader,
    unlabeled_loader,
    optimizer,
    criterion,
    device,
    pseudo_threshold: float = 0.95,
    lambda_u: float = 1.0,
    mixup_enabled: bool = False,
    mixup_alpha: float = 0.2,
    mixup_mode: str = "mixup",
    accum_steps: int = 1,
):
    model.train()
    ema.ema.eval()

    losses, accs = [], []
    mask_ratios = []

    # unlabeled iterator (cycle), only when unsupervised branch is enabled
    unl_it = None
    if unlabeled_loader is not None and float(lambda_u) > 0.0:
        unl_it = iter(unlabeled_loader)

    optimizer.zero_grad()
    _accum_count = 0

    for imgs_x, labels_x in tqdm(labeled_loader, desc="Train", leave=False):
        imgs_x = imgs_x.to(device, non_blocking=True)
        labels_x = labels_x.to(device, non_blocking=True)

        # supervised
        do_mixup = bool(mixup_enabled) and float(mixup_alpha) > 0.0 and imgs_x.size(0) > 1
        if do_mixup:
            lam = float(np.random.beta(float(mixup_alpha), float(mixup_alpha)))
            perm = torch.randperm(imgs_x.size(0), device=imgs_x.device)
            labels_a = labels_x
            labels_b = labels_x[perm]
            if mixup_mode == "cutmix":
                imgs_mix = imgs_x.clone()
                y1, x1, y2, x2 = rand_bbox(imgs_x.size(2), imgs_x.size(3), lam)
                imgs_mix[:, :, y1:y2, x1:x2] = imgs_x[perm, :, y1:y2, x1:x2]
                lam = 1.0 - float((y2 - y1) * (x2 - x1)) / float(imgs_x.size(2) * imgs_x.size(3))
            else:
                lam = max(lam, 1.0 - lam)
                imgs_mix = lam * imgs_x + (1.0 - lam) * imgs_x[perm]
            logits_x = model(imgs_mix)
            loss_x = lam * criterion(logits_x, labels_a) + (1.0 - lam) * criterion(logits_x, labels_b)
            with torch.no_grad():
                logits_acc = model(imgs_x)
        else:
            logits_x = model(imgs_x)
            loss_x = criterion(logits_x, labels_x)
            logits_acc = logits_x

        if unl_it is not None:
            try:
                xw_u, xs_u = next(unl_it)
            except StopIteration:
                unl_it = iter(unlabeled_loader)
                xw_u, xs_u = next(unl_it)

            xw_u = xw_u.to(device, non_blocking=True)
            xs_u = xs_u.to(device, non_blocking=True)

            # unsupervised (teacher on weak, student on strong)
            with torch.no_grad():
                t_logits = ema.ema(xw_u)
                t_prob = torch.softmax(t_logits, dim=-1)
                t_conf, t_pred = t_prob.max(dim=-1)
                mask = (t_conf >= float(pseudo_threshold)).float()
                mask_ratios.append(mask.mean().item())

            s_logits_u = model(xs_u)
            loss_u_all = torch.nn.functional.cross_entropy(s_logits_u, t_pred, reduction="none")
            # avoid div-by-0
            denom = mask.sum().clamp(min=1.0)
            loss_u = (loss_u_all * mask).sum() / denom
            loss = loss_x + float(lambda_u) * loss_u
        else:
            mask_ratios.append(0.0)
            loss = loss_x

        (loss / accum_steps).backward()
        _accum_count += 1

        if _accum_count % accum_steps == 0:
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            optimizer.zero_grad()
            ema.update(model)

        acc = (logits_acc.argmax(dim=-1) == labels_x).float().mean().item()
        losses.append(loss.item())
        accs.append(acc)

    # flush remaining accumulated gradients
    if _accum_count % accum_steps != 0:
        nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        optimizer.zero_grad()
        ema.update(model)

    mr = float(np.mean(mask_ratios)) if len(mask_ratios) > 0 else 0.0
    return float(np.mean(losses)), float(np.mean(accs)), mr


@torch.no_grad()
def valid_one_epoch(model, loader, criterion, device):
    model.eval()
    losses, accs = [], []
    for imgs, labels in tqdm(loader, desc="Valid", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(imgs)
        loss = criterion(logits, labels)
        acc = (logits.argmax(dim=-1) == labels).float().mean().item()
        losses.append(loss.item())
        accs.append(acc)
    return float(np.mean(losses)), float(np.mean(accs))


@torch.no_grad()
def tta_forward(model, dataset, tta_tfm, device, num_augments=5, batch_size=64):
    # Average logits: 1 base pass (dataset.transform) + N augmented passes (tta_tfm).
    model.eval()
    orig_tfm = dataset.transform

    # base pass (deterministic)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    chunks = []
    for imgs, _ in tqdm(loader, desc="TTA base", leave=False):
        chunks.append(model(imgs.to(device, non_blocking=True)).cpu())
    summed = torch.cat(chunks, dim=0)

    # augmented passes
    dataset.transform = tta_tfm
    for ai in range(num_augments):
        chunks = []
        for imgs, _ in tqdm(DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0),
                            desc=f"TTA {ai+1}/{num_augments}", leave=False):
            chunks.append(model(imgs.to(device, non_blocking=True)).cpu())
        summed += torch.cat(chunks, dim=0)

    dataset.transform = orig_tfm
    return summed / (1 + num_augments)



def _build_transforms(cfg, logger):
    img_size = int(cfg["image"]["img_size"])
    mean = tuple(float(x) for x in cfg["image"]["mean"])
    std = tuple(float(x) for x in cfg["image"]["std"])
    cj = cfg["augment"]["color_jitter"]
    rrc_scale = tuple(float(x) for x in cfg["augment"]["random_resized_crop_scale"])
    rrc_ratio = tuple(float(x) for x in cfg["augment"]["random_resized_crop_ratio"])

    ra_num_ops = int(cfg.get("semi", {}).get("randaugment_num_ops", 2))
    ra_magnitude = int(cfg.get("semi", {}).get("randaugment_magnitude", 9))

    re_cfg = cfg.get("augment", {}).get("random_erasing", {})
    re_p = float(re_cfg.get("p", 0.25))
    re_scale = tuple(float(x) for x in re_cfg.get("scale", [0.02, 0.2]))

    train_tfm = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=rrc_scale, ratio=rrc_ratio),
        transforms.RandomHorizontalFlip(p=float(cfg["augment"]["horizontal_flip_p"])),
        transforms.RandomRotation(float(cfg["augment"]["rotation_deg"])),
        transforms.ColorJitter(
            brightness=float(cj["brightness"]),
            contrast=float(cj["contrast"]),
            saturation=float(cj["saturation"]),
            hue=float(cj["hue"]),
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=re_p, scale=re_scale),
    ])

    if hasattr(transforms, "RandAugment"):
        unlabeled_strong_tfm = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=rrc_scale, ratio=rrc_ratio),
            transforms.RandomHorizontalFlip(p=float(cfg["augment"]["horizontal_flip_p"])),
            # transforms.RandomRotation(float(cfg["augment"]["rotation_deg"])),
            # transforms.ColorJitter(
            #     brightness=float(cj["brightness"]),
            #     contrast=float(cj["contrast"]),
            #     saturation=float(cj["saturation"]),
            #     hue=float(cj["hue"]),
            # ),
            transforms.RandAugment(num_ops=ra_num_ops, magnitude=ra_magnitude),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=re_p, scale=re_scale),
        ])
    else:
        logger.warning("There is no RandAugment in the current torchvision version.")
        unlabeled_strong_tfm = train_tfm

    test_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    tta_tfm = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    weak_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=float(cfg["augment"]["horizontal_flip_p"])),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return {
        "train": train_tfm, "unlabeled_strong": unlabeled_strong_tfm,
        "test": test_tfm, "tta": tta_tfm, "weak": weak_tfm,
    }


def _build_datasets_and_loaders(cfg, tfms, logger):
    batch_size = int(cfg["dataloader"]["batch_size"])
    num_workers = int(cfg["dataloader"]["num_workers"])
    pin_memory = bool(cfg["dataloader"]["pin_memory"])
    do_semi = bool(cfg["semi"]["enabled"])

    train_set = DatasetFolder(
        cfg["data"]["train_labeled"], loader=rgb_loader,
        extensions=("jpg", "jpeg", "png"), transform=tfms["train"],
    )
    valid_set = DatasetFolder(
        cfg["data"]["valid"], loader=rgb_loader,
        extensions=("jpg", "jpeg", "png"), transform=tfms["test"],
    )
    unlabeled_set = None
    if do_semi:
        unlabeled_set = UnlabeledPairDataset(
            cfg["data"]["train_unlabeled"],
            weak_tfm=tfms["weak"], strong_tfm=tfms["unlabeled_strong"],
        )
    test_set = DatasetFolder(
        cfg["data"]["test"], loader=rgb_loader,
        extensions=("jpg", "jpeg", "png"), transform=tfms["test"],
    )

    loader_kwargs = {}
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, **loader_kwargs,
    )
    valid_loader = DataLoader(
        valid_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, **loader_kwargs,
    )
    unlabeled_loader = None
    if do_semi:
        unsup_bs = int(cfg.get("semi", {}).get("unsup_batch_size", batch_size))
        unlabeled_loader = DataLoader(
            unlabeled_set, batch_size=unsup_bs, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
            drop_last=True, **loader_kwargs,
        )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, **loader_kwargs,
    )

    logger.info(f"train labeled: {len(train_set)}")
    if do_semi:
        logger.info(f"train unlabeled: {len(unlabeled_set)}")
    else:
        logger.info("train unlabeled: disabled (semi.enabled=false)")
    logger.info(f"valid: {len(valid_set)}")
    logger.info(f"test: {len(test_set)}")

    return {
        "train_set": train_set, "valid_set": valid_set,
        "unlabeled_set": unlabeled_set, "test_set": test_set,
        "train_loader": train_loader, "valid_loader": valid_loader,
        "unlabeled_loader": unlabeled_loader, "test_loader": test_loader,
        "batch_size": batch_size, "num_workers": num_workers,
        "pin_memory": pin_memory, "do_semi": do_semi,
    }


def _infer_and_save(model, test_set, test_loader, valid_set,
                    tta_tfm, tta_enabled, tta_num, device, batch_size,
                    va_labels, out_path, logger, label="EMA"):
    tta_va_acc = None
    if tta_enabled and tta_num > 0:
        logger.info(f"[{label}] TTA enabled: {tta_num} augmentations")
        tta_logits = tta_forward(model, valid_set, tta_tfm, device, tta_num, batch_size)
        tta_va_acc = (tta_logits.argmax(dim=-1) == va_labels).float().mean().item()
        logger.info(f"[{label}] Valid acc (TTA): {tta_va_acc:.4f}")
        tta_logits = tta_forward(model, test_set, tta_tfm, device, tta_num, batch_size)
        predictions = tta_logits.argmax(dim=-1).numpy().tolist()
    else:
        predictions = []
        with torch.no_grad():
            for imgs, _ in tqdm(test_loader, desc=f"{label} Test"):
                logits = model(imgs.to(device, non_blocking=True))
                pred = logits.argmax(dim=-1).cpu().numpy().tolist()
                predictions.extend(pred)

    with open(out_path, "w") as f:
        f.write("Id,Category\n")
        for i, pred in enumerate(predictions):
            f.write(f"{i},{pred}\n")
    logger.info(f"[{label}] Saved predictions: {out_path} ({len(predictions)} samples)")
    return predictions, tta_va_acc


def _log_summary(logger, n_epochs, best_metrics, best_path, out_path,
                 tta_va_acc, tta_enabled, tta_num,
                 swa_enabled, swa_model, swa_start_epoch,
                 swa_va_loss, swa_va_acc, swa_tta_va_acc, swa_path, swa_out_path):
    sep = "=" * 60
    logger.info("")
    logger.info(sep)
    logger.info("  TRAINING SUMMARY")
    logger.info(sep)

    logger.info("")
    logger.info("  [Best EMA Model]")
    if best_metrics:
        bm = best_metrics
        logger.info(f"    Epoch:       {bm['epoch']}/{n_epochs}")
        logger.info(f"    Train Loss:  {bm['train_loss']:.4f}")
        logger.info(f"    Train Acc:   {bm['train_acc']:.4f}")
        logger.info(f"    Valid Loss:  {bm['valid_loss']:.4f}")
        logger.info(f"    Valid Acc:   {bm['valid_acc']:.4f}")
        tta_va_str = f"{tta_va_acc:.4f}" if tta_va_acc is not None else "N/A"
        logger.info(f"    Valid (TTA): {tta_va_str}")
        logger.info(f"    Mask:        {bm['mask']:.3f}")
        logger.info(f"    Lambda_u:    {bm['lambda_u']:.3f}")
        logger.info(f"    LR:          {bm['lr']:.2e}")
    logger.info(f"    Checkpoint:  {best_path}")
    logger.info(f"    Predictions: {out_path}")

    if swa_enabled and swa_model is not None:
        logger.info("")
        logger.info("  [SWA Model]")
        logger.info(f"    Avg Range:   epoch {swa_start_epoch}~{n_epochs} ({n_epochs - swa_start_epoch} epochs)")
        logger.info(f"    Valid Loss:  {swa_va_loss:.4f}")
        logger.info(f"    Valid Acc:   {swa_va_acc:.4f}")
        swa_tta_str = f"{swa_tta_va_acc:.4f}" if swa_tta_va_acc is not None else "N/A"
        logger.info(f"    Valid (TTA): {swa_tta_str}")
        logger.info(f"    Checkpoint:  {swa_path}")
        logger.info(f"    Predictions: {swa_out_path}")

    logger.info("")
    logger.info(sep)
    logger.info("  Done.")
    logger.info(sep)


def main(config_path: str):
    date = datetime.now().strftime("%Y%m%d-%H%M%S")
    logger = _build_logger(date)

    # load config
    cfg = load_config(config_path)
    logger.info(f"date={date}")
    logger.info("config=\n" + json.dumps(cfg, indent=2, sort_keys=True))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    set_seed(int(cfg["seed"]))

    # ===========
    # Data
    # ===========
    tfms = _build_transforms(cfg, logger)
    test_tfm = tfms["test"]
    tta_tfm = tfms["tta"]

    data = _build_datasets_and_loaders(cfg, tfms, logger)
    batch_size = data["batch_size"]
    num_workers = data["num_workers"]
    pin_memory = data["pin_memory"]
    do_semi = data["do_semi"]
    train_set, valid_set, test_set = data["train_set"], data["valid_set"], data["test_set"]
    train_loader = data["train_loader"]
    valid_loader = data["valid_loader"]
    unlabeled_loader = data["unlabeled_loader"]
    test_loader = data["test_loader"]

    # ===========
    # Model
    # ===========

    dropout = float(cfg.get("train", {}).get("dropout", 0.5))
    model = resnet18(num_classes=11, dropout=dropout).to(device)

    ema_decay = float(cfg.get("semi", {}).get("ema", {}).get("decay", 0.999))
    ema = EMA(model, decay=ema_decay).to(device)

    logger.info(f"#params: {sum(p.numel() for p in model.parameters()) / 1e6:.3f} M")

    # ===========
    # Training & Validation
    # ===========

    n_epochs = int(cfg["train"]["n_epochs"])
    lr = float(cfg["train"]["lr"])
    weight_decay = float(cfg["train"]["weight_decay"])
    label_smoothing = float(cfg["train"]["label_smoothing"])

    mixup_enabled = bool(cfg.get("train", {}).get("mix", {}).get("enabled", False))
    mixup_alpha = float(cfg.get("train", {}).get("mix", {}).get("alpha", 0.2))
    mixup_mode = str(cfg.get("train", {}).get("mix", {}).get("mode", "mixup"))
    accum_steps = int(cfg.get("train", {}).get("accum_steps", 1))

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    warmup_epochs = int(cfg["semi"]["warmup_epochs"])
    pseudo_threshold = float(cfg["semi"]["pseudo_threshold"])
    lambda_u = float(cfg.get("semi", {}).get("lambda_u", 1.0))
    lambda_u_ramp_epochs = int(cfg.get("semi", {}).get("lambda_u_ramp_epochs", 30))

    best_acc = 0.0
    best_path = _append_date_suffix(str(cfg["output"]["best_path"]), date)
    best_metrics = {}  # track best epoch metrics for summary

    # SWA setup
    swa_cfg = cfg.get("swa", {})
    swa_enabled = bool(swa_cfg.get("enabled", False))
    swa_start_epoch = int(n_epochs * float(swa_cfg.get("start_epoch_ratio", 0.9)))
    swa_lr = float(swa_cfg.get("lr", 1e-5))
    swa_model = None
    swa_scheduler = None
    if swa_enabled:
        swa_model = AveragedModel(ema.ema).to(device)
        swa_scheduler = SWALR(optimizer, swa_lr=swa_lr)
        logger.info(f"SWA enabled: start averaging from epoch {swa_start_epoch}, swa_lr={swa_lr:.1e}")

    # do semi-supervised learning (FixMatch-style, no ConcatDataset)
    for epoch in range(1, n_epochs + 1):
        # warmup: can disable unsup loss before warmup_epochs
        use_unsup = do_semi and (epoch > warmup_epochs)

        if use_unsup:
            if lambda_u_ramp_epochs > 0:
                ramp = min(1.0, float(epoch - warmup_epochs) / float(lambda_u_ramp_epochs))
                lambda_u_eff = float(lambda_u) * ramp
            else:
                lambda_u_eff = float(lambda_u)
        else:
            lambda_u_eff = 0.0

        tr_loss, tr_acc, tr_mask = train_one_epoch(
            model=model,
            ema=ema,
            labeled_loader=train_loader,
            unlabeled_loader=unlabeled_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            pseudo_threshold=pseudo_threshold if use_unsup else 1.1,  # effectively keep none in warmup
            lambda_u=lambda_u_eff,
            mixup_enabled=mixup_enabled,
            mixup_alpha=mixup_alpha,
            mixup_mode=mixup_mode,
            accum_steps=accum_steps,
        )

        # Validate with EMA teacher (final inference target)
        va_loss, va_acc = valid_one_epoch(ema.ema, valid_loader, criterion, device)

        if swa_enabled and epoch >= swa_start_epoch:
            swa_scheduler.step()
        else:
            scheduler.step()

        logger.info(
            f"Epoch {epoch:02d}/{n_epochs} | train loss {tr_loss:.4f} acc {tr_acc:.4f} mask {tr_mask:.3f} u {lambda_u_eff:.3f} | "
            f"valid loss {va_loss:.4f} acc {va_acc:.4f} | lr {scheduler.get_last_lr()[0]:.2e}"
        )

        if va_acc > best_acc:
            best_acc = va_acc
            best_metrics = {
                "epoch": epoch,
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "valid_loss": va_loss,
                "valid_acc": va_acc,
                "mask": tr_mask,
                "lambda_u": lambda_u_eff,
                "lr": scheduler.get_last_lr()[0],
            }
            torch.save(
                {"student": model.state_dict(), "ema": ema.ema.state_dict(), "best_acc": best_acc},
                best_path
            )
            logger.info(f"  -> saved best (EMA): {best_acc:.4f} ({best_path})")

        # SWA: accumulate EMA weights in the last phase
        if swa_enabled and epoch >= swa_start_epoch:
            swa_model.update_parameters(ema.ema)

    # ===========
    # SWA finalize
    # ===========
    swa_va_loss, swa_va_acc, swa_path = 0.0, 0.0, ""
    if swa_enabled and swa_model is not None:
        logger.info("SWA: updating batch normalization statistics...")
        # Build a simple loader for BN update (labeled data, no augmentation)
        bn_loader = DataLoader(
            DatasetFolder(cfg["data"]["train_labeled"], loader=rgb_loader,
                          extensions=("jpg", "jpeg", "png"), transform=test_tfm),
            batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory,
        )
        update_bn(bn_loader, swa_model, device=device)
        swa_model.eval()

        # Validate SWA model
        swa_va_loss, swa_va_acc = valid_one_epoch(swa_model, valid_loader, criterion, device)
        logger.info(f"SWA valid acc: {swa_va_acc:.4f} (EMA best: {best_acc:.4f})")

        # Save SWA model
        swa_path = _append_date_suffix("swa-model.pt", date)
        torch.save({"swa": swa_model.module.state_dict(), "swa_acc": swa_va_acc}, swa_path)
        logger.info(f"  -> saved SWA model: {swa_path}")

    # ===========
    # Inference
    # ===========
    ckpt = torch.load(best_path, map_location=device)
    if isinstance(ckpt, dict) and "ema" in ckpt:
        model.load_state_dict(ckpt["ema"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    tta_enabled = bool(cfg.get("tta", {}).get("enabled", False))
    tta_num = int(cfg.get("tta", {}).get("num_augments", 5))

    va_labels = []
    for _, lbl in DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=0):
        va_labels.append(lbl)
    va_labels = torch.cat(va_labels, dim=0)

    out_path = _append_date_suffix(str(cfg["output"]["predict_path"]), date)
    _, tta_va_acc = _infer_and_save(
        model, test_set, test_loader, valid_set,
        tta_tfm, tta_enabled, tta_num, device, batch_size,
        va_labels, out_path, logger, label="EMA",
    )

    swa_tta_va_acc = None
    swa_out_path = ""
    if swa_enabled and swa_model is not None:
        swa_inner = swa_model.module
        swa_inner.eval()
        swa_out_path = _append_date_suffix("swa-predict.csv", date)
        _, swa_tta_va_acc = _infer_and_save(
            swa_inner, test_set, test_loader, valid_set,
            tta_tfm, tta_enabled, tta_num, device, batch_size,
            va_labels, swa_out_path, logger, label="SWA",
        )

    _log_summary(
        logger, n_epochs, best_metrics, best_path, out_path,
        tta_va_acc, tta_enabled, tta_num,
        swa_enabled, swa_model, swa_start_epoch,
        swa_va_loss, swa_va_acc, swa_tta_va_acc, swa_path, swa_out_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json")
    args = parser.parse_args()
    
    main(args.config)
