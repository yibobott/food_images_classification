from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from models.ema import EMA
from utils.misc import rand_bbox


@dataclass
class BestMetrics:
    epoch: int = 0
    train_loss: float = 0.0
    train_acc: float = 0.0
    valid_loss: float = 0.0
    valid_acc: float = 0.0
    mask: float = 0.0
    lambda_u: float = 0.0
    lr: float = 0.0


def train_one_epoch(
    model: nn.Module,
    ema: EMA,
    labeled_loader: DataLoader,
    unlabeled_loader: Optional[DataLoader],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    pseudo_threshold: float = 0.95,
    lambda_u: float = 1.0,
    mixup_enabled: bool = False,
    mixup_alpha: float = 0.2,
    mixup_mode: str = "mixup",
    accum_steps: int = 1,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> tuple[float, float, float]:
    model.train()
    ema.ema.eval()

    losses: list[float] = []
    accs: list[float] = []
    mask_ratios: list[float] = []

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
            loss_u_all = F.cross_entropy(s_logits_u, t_pred, reduction="none")
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
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
            ema.update(model)

        acc = (logits_acc.argmax(dim=-1) == labels_x).float().mean().item()
        losses.append(loss.item())
        accs.append(acc)

    # flush remaining accumulated gradients
    if _accum_count % accum_steps != 0:
        nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()
        ema.update(model)

    mr = float(np.mean(mask_ratios)) if len(mask_ratios) > 0 else 0.0
    return float(np.mean(losses)), float(np.mean(accs)), mr


@torch.no_grad()
def valid_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    model.eval()
    losses: list[float] = []
    accs: list[float] = []
    for imgs, labels in tqdm(loader, desc="Valid", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(imgs)
        loss = criterion(logits, labels)
        acc = (logits.argmax(dim=-1) == labels).float().mean().item()
        losses.append(loss.item())
        accs.append(acc)
    return float(np.mean(losses)), float(np.mean(accs))
