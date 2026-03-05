"""
Install packages:
pip install -r requirements.txt

Run code:
python train.py

Run with a specific config:
python train.py --config config.json
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import math
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder

from utils.config import Config, load_config
from utils.logger import build_logger
from utils.misc import set_seed, rgb_loader, append_date_suffix
from models.resnet import resnet18
from models.wrn import wrn28_8
from models.ema import EMA
from data.transforms import build_transforms, Transforms
from data.datasets import build_datasets_and_loaders, DataBundle, PseudoLabeledDataset, UnlabeledPairDataset
from engine.trainer import train_one_epoch, valid_one_epoch, BestMetrics
from engine.inference import tta_forward, infer_and_save, log_summary, RunSummary


def setup(config_path: str) -> tuple[Config, logging.Logger, str, str]:
    """
    Load config, build logger, set seed, detect device.
    """
    date = datetime.now().strftime("%Y%m%d-%H%M%S")
    logger = build_logger(date)

    cfg = load_config(config_path)
    logger.info(f"date={date}")
    logger.info("config=\n" + json.dumps(cfg.to_dict(), indent=2, sort_keys=True))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    set_seed(cfg.seed)

    return cfg, logger, date, device


def build_model(cfg: Config, device: str, logger: logging.Logger) -> tuple[nn.Module, EMA]:
    """
    Create student model and EMA teacher.
    """
    arch = cfg.model.arch
    dropout = cfg.train.dropout

    if arch == "wrn28_8":
        drop_path_rate = cfg.model.drop_path_rate
        model = wrn28_8(num_classes=11, dropout=dropout, drop_path_rate=drop_path_rate).to(device)
    else:
        model = resnet18(num_classes=11, dropout=dropout).to(device)

    ema_decay = cfg.semi.ema.decay
    ema = EMA(model, decay=ema_decay).to(device)

    logger.info(f"Model: {arch} | #params: {sum(p.numel() for p in model.parameters()) / 1e6:.3f} M")
    return model, ema


def build_optimizer(
    model: nn.Module,
    cfg: Config,
    train_loader: DataLoader,
    logger: logging.Logger,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler, nn.Module, str]:
    """
    Create AdamW optimizer and LR scheduler.
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    sched_type = cfg.train.scheduler
    max_lr = cfg.train.max_lr
    n_epochs = cfg.train.n_epochs
    accum_steps = cfg.train.accum_steps

    if sched_type == "onecycle":
        steps_per_epoch = math.ceil(len(train_loader) / max(accum_steps, 1))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lr,
            epochs=n_epochs, steps_per_epoch=steps_per_epoch,
            pct_start=0.3,
        )
        logger.info(f"Scheduler: OneCycleLR (max_lr={max_lr}, steps/epoch={steps_per_epoch})")
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
        logger.info(f"Scheduler: CosineAnnealingLR (T_max={n_epochs})")

    return optimizer, scheduler, criterion, sched_type


def train_loop(
    model: nn.Module,
    ema: EMA,
    data: DataBundle,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    criterion: nn.Module,
    sched_type: str,
    cfg: Config,
    date: str,
    logger: logging.Logger,
    device: str,
    epoch_offset: int = 0,
) -> tuple[BestMetrics, str, Optional[AveragedModel], int]:
    """
    Run the full training loop. Returns best metrics, best path, SWA model, SWA start epoch.
    epoch_offset: added to epoch numbers for logging (used in progressive resizing).
    """
    use_amp = cfg.train.use_amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    if use_amp:
        logger.info("AMP (Automatic Mixed Precision) enabled")

    n_epochs = cfg.train.n_epochs
    accum_steps = cfg.train.accum_steps
    mixup_enabled = cfg.train.mix.enabled
    mixup_alpha = cfg.train.mix.alpha
    mixup_mode = cfg.train.mix.mode

    warmup_epochs = cfg.semi.warmup_epochs
    pseudo_threshold_start = cfg.semi.pseudo_threshold
    pseudo_threshold_end = cfg.semi.pseudo_threshold_end
    lambda_u = cfg.semi.lambda_u
    lambda_u_ramp_epochs = cfg.semi.lambda_u_ramp_epochs

    # progressive threshold: if pseudo_threshold_end < 0, use fixed threshold
    use_progressive_threshold = pseudo_threshold_end >= 0.0
    if use_progressive_threshold:
        logger.info(f"Progressive threshold: {pseudo_threshold_start:.2f} -> {pseudo_threshold_end:.2f}")

    best_acc = 0.0
    best_path = append_date_suffix(cfg.output.best_path, date)
    best_metrics = BestMetrics()

    # SWA setup
    swa_enabled = cfg.swa.enabled
    swa_start_epoch = int(n_epochs * cfg.swa.start_epoch_ratio)
    swa_lr = cfg.swa.lr
    swa_model: Optional[AveragedModel] = None
    swa_scheduler = None
    if swa_enabled:
        swa_model = AveragedModel(ema.ema).to(device)
        swa_scheduler = SWALR(optimizer, swa_lr=swa_lr)
        logger.info(f"SWA enabled: start averaging from epoch {swa_start_epoch}, swa_lr={swa_lr:.1e}")

    # do semi-supervised learning (FixMatch-style, no ConcatDataset)
    for epoch in range(1, n_epochs + 1):
        # warmup: can disable unsup loss before warmup_epochs
        use_unsup = data.do_semi and (epoch > warmup_epochs)

        # compute current pseudo threshold (progressive or fixed)
        if use_progressive_threshold:
            progress = min(1.0, float(epoch - 1) / max(n_epochs - 1, 1))
            current_threshold = pseudo_threshold_start + (pseudo_threshold_end - pseudo_threshold_start) * progress
        else:
            current_threshold = pseudo_threshold_start

        if use_unsup:
            if lambda_u_ramp_epochs > 0:
                ramp = min(1.0, float(epoch - warmup_epochs) / float(lambda_u_ramp_epochs))
                lambda_u_eff = float(lambda_u) * ramp
            else:
                lambda_u_eff = float(lambda_u)
        else:
            lambda_u_eff = 0.0

        in_swa = swa_enabled and epoch >= swa_start_epoch
        # OneCycleLR steps per batch inside train_one_epoch;
        # CosineAnnealingLR steps per epoch below.
        batch_sched = scheduler if (sched_type == "onecycle" and not in_swa) else None

        tr_loss, tr_acc, tr_mask = train_one_epoch(
            model=model,
            ema=ema,
            labeled_loader=data.train_loader,
            unlabeled_loader=data.unlabeled_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            pseudo_threshold=current_threshold if use_unsup else 1.1,  # effectively keep none in warmup
            lambda_u=lambda_u_eff,
            mixup_enabled=mixup_enabled,
            mixup_alpha=mixup_alpha,
            mixup_mode=mixup_mode,
            accum_steps=accum_steps,
            scheduler=batch_sched,
            use_amp=use_amp,
            scaler=scaler,
        )

        # Validate with EMA teacher (final inference target)
        va_loss, va_acc = valid_one_epoch(ema.ema, data.valid_loader, criterion, device)

        if in_swa:
            swa_scheduler.step()
        elif sched_type != "onecycle":
            scheduler.step()

        ge = epoch + epoch_offset
        te = n_epochs + epoch_offset
        logger.info(
            f"Epoch {ge:02d}/{te} | train loss {tr_loss:.4f} acc {tr_acc:.4f} mask {tr_mask:.3f} u {lambda_u_eff:.3f} thr {current_threshold:.3f} | "
            f"valid loss {va_loss:.4f} acc {va_acc:.4f} | lr {optimizer.param_groups[0]['lr']:.2e}"
        )

        if va_acc > best_acc:
            best_acc = va_acc
            best_metrics = BestMetrics(
                epoch=ge,
                train_loss=tr_loss,
                train_acc=tr_acc,
                valid_loss=va_loss,
                valid_acc=va_acc,
                mask=tr_mask,
                lambda_u=lambda_u_eff,
                lr=optimizer.param_groups[0]['lr'],
            )
            torch.save(
                {"student": model.state_dict(), "ema": ema.ema.state_dict(), "best_acc": best_acc},
                best_path
            )
            logger.info(f"  -> saved best (EMA): {best_acc:.4f} ({best_path})")

        # SWA: accumulate EMA weights in the last phase
        if swa_enabled and epoch >= swa_start_epoch:
            swa_model.update_parameters(ema.ema)

    return best_metrics, best_path, swa_model, swa_start_epoch


def finalize_swa(
    swa_model: Optional[AveragedModel],
    cfg: Config,
    data: DataBundle,
    criterion: nn.Module,
    device: str,
    date: str,
    best_acc: float,
    logger: logging.Logger,
) -> tuple[float, float, str]:
    """
    Update SWA batch norm and validate. Returns (swa_va_loss, swa_va_acc, swa_path).
    """
    if not cfg.swa.enabled or swa_model is None:
        return 0.0, 0.0, ""

    logger.info("SWA: updating batch normalization statistics...")
    test_tfm_set = DatasetFolder(
        cfg.data.train_labeled, loader=rgb_loader,
        extensions=("jpg", "jpeg", "png"), transform=None,
    )
    # Reuse the test transform for BN update
    from data.transforms import Transforms
    import torchvision.transforms as T
    bn_tfm = T.Compose([
        T.Resize((cfg.image.img_size, cfg.image.img_size)),
        T.ToTensor(),
        T.Normalize(cfg.image.mean, cfg.image.std),
    ])
    bn_set = DatasetFolder(
        cfg.data.train_labeled, loader=rgb_loader,
        extensions=("jpg", "jpeg", "png"), transform=bn_tfm,
    )
    bn_loader = DataLoader(
        bn_set, batch_size=data.batch_size, shuffle=True,
        num_workers=data.num_workers, pin_memory=data.pin_memory,
    )
    update_bn(bn_loader, swa_model, device=device)
    swa_model.eval()

    # Validate SWA model
    swa_va_loss, swa_va_acc = valid_one_epoch(swa_model, data.valid_loader, criterion, device)
    logger.info(f"SWA valid acc: {swa_va_acc:.4f} (EMA best: {best_acc:.4f})")

    # Save SWA model
    swa_path = append_date_suffix("swa-model.pt", date)
    torch.save({"swa": swa_model.module.state_dict(), "swa_acc": swa_va_acc}, swa_path)
    logger.info(f"  -> saved SWA model: {swa_path}")

    return swa_va_loss, swa_va_acc, swa_path


def run_inference(
    model: nn.Module,
    swa_model: Optional[AveragedModel],
    data: DataBundle,
    tfms: Transforms,
    cfg: Config,
    best_path: str,
    best_metrics: BestMetrics,
    swa_va_loss: float,
    swa_va_acc: float,
    swa_path: str,
    swa_start_epoch: int,
    date: str,
    device: str,
    logger: logging.Logger,
) -> None:
    """
    Load best model, run TTA inference, log summary.
    """
    ckpt = torch.load(best_path, map_location=device)
    if isinstance(ckpt, dict) and "ema" in ckpt:
        model.load_state_dict(ckpt["ema"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    tta_enabled = cfg.tta.enabled
    tta_num = cfg.tta.num_augments

    va_labels_chunks: list[torch.Tensor] = []
    for _, lbl in DataLoader(data.valid_set, batch_size=data.batch_size, shuffle=False, num_workers=0):
        va_labels_chunks.append(lbl)
    va_labels = torch.cat(va_labels_chunks, dim=0)

    out_path = append_date_suffix(cfg.output.predict_path, date)
    _, tta_va_acc = infer_and_save(
        model, data.test_set, data.test_loader, data.valid_set,
        tfms.tta, tta_enabled, tta_num, device, data.batch_size,
        va_labels, out_path, logger, label="EMA",
    )

    swa_tta_va_acc: Optional[float] = None
    swa_out_path = ""
    if cfg.swa.enabled and swa_model is not None:
        swa_inner = swa_model.module
        swa_inner.eval()
        swa_out_path = append_date_suffix("swa-predict.csv", date)
        _, swa_tta_va_acc = infer_and_save(
            swa_inner, data.test_set, data.test_loader, data.valid_set,
            tfms.tta, tta_enabled, tta_num, device, data.batch_size,
            va_labels, swa_out_path, logger, label="SWA",
        )

    summary = RunSummary(
        n_epochs=cfg.train.n_epochs,
        best_metrics=best_metrics,
        best_path=best_path,
        predict_path=out_path,
        tta_va_acc=tta_va_acc,
        tta_enabled=tta_enabled,
        tta_num=tta_num,
        swa_enabled=cfg.swa.enabled,
        swa_start_epoch=swa_start_epoch,
        swa_va_loss=swa_va_loss,
        swa_va_acc=swa_va_acc,
        swa_tta_va_acc=swa_tta_va_acc,
        swa_path=swa_path,
        swa_predict_path=swa_out_path,
    )
    log_summary(logger, summary)



def generate_pseudo_labels(
    model: nn.Module,
    cfg: Config,
    device: str,
    threshold: float,
    logger: logging.Logger,
) -> tuple[list[str], list[int]]:
    """
    Use teacher model to generate pseudo-labels for all unlabeled data.
    Returns (paths, labels) for samples above the confidence threshold.
    """
    import torchvision.transforms as T
    model.eval()

    img_size = cfg.image.img_size
    tfm = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(cfg.image.mean, cfg.image.std),
    ])
    unl_set = DatasetFolder(
        cfg.data.train_unlabeled, loader=rgb_loader,
        extensions=("jpg", "jpeg", "png"), transform=tfm,
    )
    unl_loader = DataLoader(
        unl_set, batch_size=cfg.dataloader.batch_size * 2, shuffle=False,
        num_workers=cfg.dataloader.num_workers, pin_memory=cfg.dataloader.pin_memory,
    )

    all_paths = [p for (p, _) in unl_set.samples]
    all_preds: list[int] = []
    all_confs: list[float] = []

    with torch.no_grad():
        for imgs, _ in unl_loader:
            imgs = imgs.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = model(imgs)
            probs = torch.softmax(logits, dim=-1)
            confs, preds = probs.max(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_confs.extend(confs.cpu().tolist())

    # filter by threshold
    kept_paths: list[str] = []
    kept_labels: list[int] = []
    for path, pred, conf in zip(all_paths, all_preds, all_confs):
        if conf >= threshold:
            kept_paths.append(path)
            kept_labels.append(pred)

    logger.info(
        f"Pseudo-labels: {len(kept_paths)}/{len(all_paths)} samples above threshold {threshold:.2f} "
        f"(avg conf: {sum(all_confs)/len(all_confs):.4f})"
    )
    return kept_paths, kept_labels


def self_train_phase(
    cfg: Config,
    date: str,
    logger: logging.Logger,
    device: str,
    teacher_path: str,
) -> None:
    """
    Phase 2: Self-Training.
    1. Load teacher (EMA) model and generate pseudo-labels for unlabeled data.
    2. Combine labeled + pseudo-labeled data.
    3. Train a new model from scratch (supervised only).
    """
    sep = "=" * 60
    logger.info("")
    logger.info(sep)
    logger.info("  PHASE 2: SELF-TRAINING")
    logger.info(sep)

    # Load teacher
    teacher_model, teacher_ema = build_model(cfg, device, logger)
    ckpt = torch.load(teacher_path, map_location=device)
    if isinstance(ckpt, dict) and "ema" in ckpt:
        teacher_model.load_state_dict(ckpt["ema"])
    else:
        teacher_model.load_state_dict(ckpt)
    teacher_model.eval()

    # Generate pseudo-labels
    st_threshold = cfg.self_training.threshold
    pseudo_paths, pseudo_labels = generate_pseudo_labels(
        teacher_model, cfg, device, st_threshold, logger,
    )
    del teacher_model, teacher_ema
    torch.cuda.empty_cache()

    # Build transforms (full img_size, no progressive resize in Phase 2)
    tfms = build_transforms(cfg, logger)

    # Build labeled dataset
    labeled_set = DatasetFolder(
        cfg.data.train_labeled, loader=rgb_loader,
        extensions=("jpg", "jpeg", "png"), transform=tfms.train,
    )
    pseudo_set = PseudoLabeledDataset(pseudo_paths, pseudo_labels, transform=tfms.train)
    combined_set = torch.utils.data.ConcatDataset([labeled_set, pseudo_set])
    logger.info(f"Self-training data: {len(labeled_set)} labeled + {len(pseudo_set)} pseudo = {len(combined_set)} total")

    # Build loaders
    loader_kwargs: dict = {}
    if cfg.dataloader.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(
        combined_set, batch_size=cfg.dataloader.batch_size, shuffle=True,
        num_workers=cfg.dataloader.num_workers, pin_memory=cfg.dataloader.pin_memory,
        **loader_kwargs,
    )
    valid_set = DatasetFolder(
        cfg.data.valid, loader=rgb_loader,
        extensions=("jpg", "jpeg", "png"), transform=tfms.test,
    )
    valid_loader = DataLoader(
        valid_set, batch_size=cfg.dataloader.batch_size, shuffle=False,
        num_workers=cfg.dataloader.num_workers, pin_memory=cfg.dataloader.pin_memory,
        **loader_kwargs,
    )
    test_set = DatasetFolder(
        cfg.data.test, loader=rgb_loader,
        extensions=("jpg", "jpeg", "png"), transform=tfms.test,
    )
    test_loader = DataLoader(
        test_set, batch_size=cfg.dataloader.batch_size, shuffle=False,
        num_workers=cfg.dataloader.num_workers, pin_memory=cfg.dataloader.pin_memory,
        **loader_kwargs,
    )

    data = DataBundle(
        train_set=labeled_set, valid_set=valid_set,
        unlabeled_set=None, test_set=test_set,
        train_loader=train_loader, valid_loader=valid_loader,
        unlabeled_loader=None, test_loader=test_loader,
        batch_size=cfg.dataloader.batch_size,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory,
        do_semi=False,
    )

    # Build new model from scratch
    model, ema = build_model(cfg, device, logger)

    # Build optimizer with self-training epochs
    cfg_st = copy.deepcopy(cfg)
    cfg_st.train.n_epochs = cfg.self_training.epochs

    optimizer, scheduler, criterion, sched_type = build_optimizer(
        model, cfg_st, train_loader, logger,
    )

    # Train (supervised only, no semi)
    st_date = date + "-ST"
    best_metrics, best_path, swa_model, swa_start_epoch = train_loop(
        model, ema, data, optimizer, scheduler, criterion, sched_type,
        cfg_st, st_date, logger, device,
    )

    # SWA finalize
    swa_va_loss, swa_va_acc, swa_path = finalize_swa(
        swa_model, cfg_st, data, criterion, device, st_date,
        best_metrics.valid_acc, logger,
    )

    # Inference & summary
    run_inference(
        model, swa_model, data, tfms, cfg_st,
        best_path, best_metrics,
        swa_va_loss, swa_va_acc, swa_path, swa_start_epoch,
        st_date, device, logger,
    )


def main(config_path: str) -> None:
    # Setup
    cfg, logger, date, device = setup(config_path)

    # Model
    model, ema = build_model(cfg, device, logger)

    pr = cfg.progressive_resize

    if pr.enabled:
        # ─── Progressive Resize: Stage 1 (small image) ───
        s1_epochs = pr.stage1_epochs
        total_epochs = cfg.train.n_epochs
        s2_epochs = total_epochs - s1_epochs

        logger.info(f"=== Progressive Resize Stage 1: img{pr.stage1_size} x {s1_epochs} epochs ===")
        tfms1 = build_transforms(cfg, logger, img_size_override=pr.stage1_size)
        data1 = build_datasets_and_loaders(cfg, tfms1, logger)

        cfg_s1 = copy.deepcopy(cfg)
        cfg_s1.train.n_epochs = s1_epochs
        cfg_s1.swa.enabled = False

        optimizer1, scheduler1, criterion, sched_type = build_optimizer(
            model, cfg_s1, data1.train_loader, logger,
        )
        best_metrics, best_path, _, _ = train_loop(
            model, ema, data1, optimizer1, scheduler1, criterion, sched_type,
            cfg_s1, date, logger, device,
        )

        # Free Stage 1 resources before Stage 2
        del optimizer1, scheduler1, data1, tfms1, cfg_s1
        torch.cuda.empty_cache()

        # ─── Progressive Resize: Stage 2 (full image) ───
        logger.info(f"=== Progressive Resize Stage 2: img{cfg.image.img_size} x {s2_epochs} epochs ===")
        tfms = build_transforms(cfg, logger)
        data = build_datasets_and_loaders(cfg, tfms, logger)

        cfg_s2 = copy.deepcopy(cfg)
        cfg_s2.train.n_epochs = s2_epochs

        optimizer2, scheduler2, criterion, sched_type = build_optimizer(
            model, cfg_s2, data.train_loader, logger,
        )
        best_metrics, best_path, swa_model, swa_start_epoch = train_loop(
            model, ema, data, optimizer2, scheduler2, criterion, sched_type,
            cfg_s2, date, logger, device, epoch_offset=s1_epochs,
        )
        # adjust swa_start_epoch for correct summary display
        swa_start_epoch = swa_start_epoch + s1_epochs if swa_start_epoch else swa_start_epoch
        n_epochs_display = total_epochs
    else:
        # ─── Standard training (no progressive resize) ───
        tfms = build_transforms(cfg, logger)
        data = build_datasets_and_loaders(cfg, tfms, logger)

        optimizer, scheduler, criterion, sched_type = build_optimizer(
            model, cfg, data.train_loader, logger,
        )
        best_metrics, best_path, swa_model, swa_start_epoch = train_loop(
            model, ema, data, optimizer, scheduler, criterion, sched_type,
            cfg, date, logger, device,
        )
        n_epochs_display = cfg.train.n_epochs

    # SWA finalize
    swa_va_loss, swa_va_acc, swa_path = finalize_swa(
        swa_model, cfg, data, criterion, device, date,
        best_metrics.valid_acc, logger,
    )

    # Inference & summary (use n_epochs_display for correct total)
    cfg_display = copy.deepcopy(cfg)
    cfg_display.train.n_epochs = n_epochs_display
    run_inference(
        model, swa_model, data, tfms, cfg_display,
        best_path, best_metrics,
        swa_va_loss, swa_va_acc, swa_path, swa_start_epoch,
        date, device, logger,
    )

    # ─── Phase 2: Self-Training ───
    if cfg.self_training.enabled:
        self_train_phase(cfg, date, logger, device, best_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json")
    args = parser.parse_args()

    main(args.config)
