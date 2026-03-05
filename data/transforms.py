from __future__ import annotations

import logging
from dataclasses import dataclass

import torchvision.transforms as transforms

from utils.config import Config


@dataclass
class Transforms:
    train: transforms.Compose
    unlabeled_strong: transforms.Compose
    test: transforms.Compose
    tta: transforms.Compose
    weak: transforms.Compose


def build_transforms(cfg: Config, logger: logging.Logger, img_size_override: int | None = None) -> Transforms:
    img_size = img_size_override if img_size_override is not None else cfg.image.img_size
    mean = cfg.image.mean
    std = cfg.image.std
    cj = cfg.augment.color_jitter
    rrc_scale = cfg.augment.random_resized_crop_scale
    rrc_ratio = cfg.augment.random_resized_crop_ratio

    ra_num_ops = cfg.semi.randaugment_num_ops
    ra_magnitude = cfg.semi.randaugment_magnitude

    re_p = cfg.augment.random_erasing.p
    re_scale = cfg.augment.random_erasing.scale
    grayscale_p = cfg.augment.grayscale_p

    train_ops = [
        transforms.RandomResizedCrop(img_size, scale=rrc_scale, ratio=rrc_ratio),
        transforms.RandomHorizontalFlip(p=cfg.augment.horizontal_flip_p),
        transforms.RandomRotation(cfg.augment.rotation_deg),
        transforms.ColorJitter(
            brightness=cj.brightness,
            contrast=cj.contrast,
            saturation=cj.saturation,
            hue=cj.hue,
        ),
    ]
    if grayscale_p > 0:
        train_ops.append(transforms.RandomGrayscale(p=grayscale_p))
    train_ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=re_p, scale=re_scale),
    ]
    train_tfm = transforms.Compose(train_ops)

    if hasattr(transforms, "RandAugment"):
        strong_ops = [
            transforms.RandomResizedCrop(img_size, scale=rrc_scale, ratio=rrc_ratio),
            transforms.RandomHorizontalFlip(p=cfg.augment.horizontal_flip_p),
            transforms.RandAugment(num_ops=ra_num_ops, magnitude=ra_magnitude),
        ]
        if grayscale_p > 0:
            strong_ops.append(transforms.RandomGrayscale(p=grayscale_p))
        strong_ops += [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=re_p, scale=re_scale),
        ]
        unlabeled_strong_tfm = transforms.Compose(strong_ops)
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
        transforms.RandomHorizontalFlip(p=cfg.augment.horizontal_flip_p),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return Transforms(
        train=train_tfm,
        unlabeled_strong=unlabeled_strong_tfm,
        test=test_tfm,
        tta=tta_tfm,
        weak=weak_tfm,
    )
