from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder
from tqdm.auto import tqdm

from engine.trainer import BestMetrics


@dataclass
class RunSummary:
    n_epochs: int = 0
    best_metrics: BestMetrics = None  # type: ignore[assignment]
    best_path: str = ""
    predict_path: str = ""
    tta_va_acc: Optional[float] = None
    tta_enabled: bool = False
    tta_num: int = 0
    swa_enabled: bool = False
    swa_start_epoch: int = 0
    swa_va_loss: float = 0.0
    swa_va_acc: float = 0.0
    swa_tta_va_acc: Optional[float] = None
    swa_path: str = ""
    swa_predict_path: str = ""


@torch.no_grad()
def tta_forward(
    model: nn.Module,
    dataset: DatasetFolder,
    tta_tfm,
    device: str,
    num_augments: int = 5,
    batch_size: int = 64,
) -> torch.Tensor:
    # Average logits: 1 base pass (dataset.transform) + N augmented passes (tta_tfm).
    model.eval()
    orig_tfm = dataset.transform

    # base pass (deterministic)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    chunks: list[torch.Tensor] = []
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


def infer_and_save(
    model: nn.Module,
    test_set: DatasetFolder,
    test_loader: DataLoader,
    valid_set: DatasetFolder,
    tta_tfm,
    tta_enabled: bool,
    tta_num: int,
    device: str,
    batch_size: int,
    va_labels: torch.Tensor,
    out_path: str,
    logger: logging.Logger,
    label: str = "EMA",
) -> tuple[list[int], Optional[float]]:
    tta_va_acc: Optional[float] = None
    if tta_enabled and tta_num > 0:
        logger.info(f"[{label}] TTA enabled: {tta_num} augmentations")
        tta_logits = tta_forward(model, valid_set, tta_tfm, device, tta_num, batch_size)
        tta_va_acc = (tta_logits.argmax(dim=-1) == va_labels).float().mean().item()
        logger.info(f"[{label}] Valid acc (TTA): {tta_va_acc:.4f}")
        tta_logits = tta_forward(model, test_set, tta_tfm, device, tta_num, batch_size)
        predictions = tta_logits.argmax(dim=-1).numpy().tolist()
    else:
        predictions: list[int] = []
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


def log_summary(logger: logging.Logger, summary: RunSummary) -> None:
    sep = "=" * 60
    logger.info("")
    logger.info(sep)
    logger.info("  TRAINING SUMMARY")
    logger.info(sep)

    logger.info("")
    logger.info("  [Best EMA Model]")
    bm = summary.best_metrics
    if bm is not None:
        logger.info(f"    Epoch:       {bm.epoch}/{summary.n_epochs}")
        logger.info(f"    Train Loss:  {bm.train_loss:.4f}")
        logger.info(f"    Train Acc:   {bm.train_acc:.4f}")
        logger.info(f"    Valid Loss:  {bm.valid_loss:.4f}")
        logger.info(f"    Valid Acc:   {bm.valid_acc:.4f}")
        tta_va_str = f"{summary.tta_va_acc:.4f}" if summary.tta_va_acc is not None else "N/A"
        logger.info(f"    Valid (TTA): {tta_va_str}")
        logger.info(f"    Mask:        {bm.mask:.3f}")
        logger.info(f"    Lambda_u:    {bm.lambda_u:.3f}")
        logger.info(f"    LR:          {bm.lr:.2e}")
    logger.info(f"    Checkpoint:  {summary.best_path}")
    logger.info(f"    Predictions: {summary.predict_path}")

    if summary.swa_enabled:
        logger.info("")
        logger.info("  [SWA Model]")
        logger.info(f"    Avg Range:   epoch {summary.swa_start_epoch}~{summary.n_epochs} "
                     f"({summary.n_epochs - summary.swa_start_epoch} epochs)")
        logger.info(f"    Valid Loss:  {summary.swa_va_loss:.4f}")
        logger.info(f"    Valid Acc:   {summary.swa_va_acc:.4f}")
        swa_tta_str = f"{summary.swa_tta_va_acc:.4f}" if summary.swa_tta_va_acc is not None else "N/A"
        logger.info(f"    Valid (TTA): {swa_tta_str}")
        logger.info(f"    Checkpoint:  {summary.swa_path}")
        logger.info(f"    Predictions: {summary.swa_predict_path}")

    logger.info("")
    logger.info(sep)
    logger.info("  Done.")
    logger.info(sep)
