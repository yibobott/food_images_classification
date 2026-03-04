"""
Configuration dataclasses and loading utilities.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict


def _filter_fields(cls: type, d: dict) -> dict:
    """
    Filter dict to only include valid dataclass field names.
    """
    return {k: v for k, v in d.items() if k in cls.__dataclass_fields__}


# Nested config dataclasses

@dataclass
class DataConfig:
    train_labeled: str = "food11/training/labeled"
    train_unlabeled: str = "food11/training/unlabeled"
    valid: str = "food11/validation"
    test: str = "food11/testing"


@dataclass
class DataLoaderConfig:
    batch_size: int = 128
    num_workers: int = 0
    pin_memory: bool = True


@dataclass
class ImageConfig:
    img_size: int = 128
    mean: tuple[float, ...] = (0.485, 0.456, 0.406)
    std: tuple[float, ...] = (0.229, 0.224, 0.225)

    @classmethod
    def from_dict(cls, d: dict) -> ImageConfig:
        return cls(
            img_size=d.get("img_size", 128),
            mean=tuple(d["mean"]) if "mean" in d else (0.485, 0.456, 0.406),
            std=tuple(d["std"]) if "std" in d else (0.229, 0.224, 0.225),
        )


@dataclass
class ColorJitterConfig:
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    hue: float = 0.05


@dataclass
class RandomErasingConfig:
    p: float = 0.25
    scale: tuple[float, float] = (0.02, 0.2)

    @classmethod
    def from_dict(cls, d: dict) -> RandomErasingConfig:
        return cls(
            p=d.get("p", 0.25),
            scale=tuple(d["scale"]) if "scale" in d else (0.02, 0.2),
        )


@dataclass
class AugmentConfig:
    random_resized_crop_scale: tuple[float, float] = (0.7, 1.0)
    random_resized_crop_ratio: tuple[float, float] = (0.9, 1.1)
    horizontal_flip_p: float = 0.5
    rotation_deg: float = 15.0
    color_jitter: ColorJitterConfig = field(default_factory=ColorJitterConfig)
    random_erasing: RandomErasingConfig = field(default_factory=RandomErasingConfig)

    @classmethod
    def from_dict(cls, d: dict) -> AugmentConfig:
        return cls(
            random_resized_crop_scale=tuple(d["random_resized_crop_scale"])
            if "random_resized_crop_scale" in d else (0.7, 1.0),
            random_resized_crop_ratio=tuple(d["random_resized_crop_ratio"])
            if "random_resized_crop_ratio" in d else (0.9, 1.1),
            horizontal_flip_p=d.get("horizontal_flip_p", 0.5),
            rotation_deg=d.get("rotation_deg", 15.0),
            color_jitter=ColorJitterConfig(**_filter_fields(ColorJitterConfig, d.get("color_jitter", {}))),
            random_erasing=RandomErasingConfig.from_dict(d.get("random_erasing", {})),
        )


@dataclass
class MixConfig:
    enabled: bool = False
    alpha: float = 0.2
    mode: str = "mixup"


@dataclass
class TrainConfig:
    n_epochs: int = 25
    lr: float = 3e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    dropout: float = 0.5
    accum_steps: int = 1
    scheduler: str = "cosine"
    max_lr: float = 3e-3
    mix: MixConfig = field(default_factory=MixConfig)

    @classmethod
    def from_dict(cls, d: dict) -> TrainConfig:
        mix_raw = d.get("mix", {})
        mix_clean = {k: v for k, v in mix_raw.items() if not k.startswith("_")}
        return cls(
            n_epochs=d.get("n_epochs", 25),
            lr=d.get("lr", 3e-4),
            weight_decay=d.get("weight_decay", 1e-4),
            label_smoothing=d.get("label_smoothing", 0.1),
            dropout=d.get("dropout", 0.5),
            accum_steps=d.get("accum_steps", 1),
            scheduler=d.get("scheduler", "cosine"),
            max_lr=d.get("max_lr", 3e-3),
            mix=MixConfig(**_filter_fields(MixConfig, mix_clean)),
        )


@dataclass
class TTAConfig:
    enabled: bool = False
    num_augments: int = 5


@dataclass
class SWAConfig:
    enabled: bool = False
    start_epoch_ratio: float = 0.9
    lr: float = 1e-5


@dataclass
class EMAConfig:
    decay: float = 0.999


@dataclass
class SemiConfig:
    enabled: bool = True
    warmup_epochs: int = 5
    pseudo_threshold: float = 0.95
    unsup_batch_size: int = 128
    lambda_u: float = 1.0
    lambda_u_ramp_epochs: int = 30
    randaugment_num_ops: int = 2
    randaugment_magnitude: int = 9
    ema: EMAConfig = field(default_factory=EMAConfig)

    @classmethod
    def from_dict(cls, d: dict) -> SemiConfig:
        ema_raw = d.get("ema", {})
        return cls(
            enabled=d.get("enabled", True),
            warmup_epochs=d.get("warmup_epochs", 5),
            pseudo_threshold=d.get("pseudo_threshold", 0.95),
            unsup_batch_size=d.get("unsup_batch_size", 128),
            lambda_u=d.get("lambda_u", 1.0),
            lambda_u_ramp_epochs=d.get("lambda_u_ramp_epochs", 30),
            randaugment_num_ops=d.get("randaugment_num_ops", 2),
            randaugment_magnitude=d.get("randaugment_magnitude", 9),
            ema=EMAConfig(**_filter_fields(EMAConfig, ema_raw)),
        )


@dataclass
class OutputConfig:
    best_path: str = "best-model.pt"
    predict_path: str = "predict.csv"


# Top-level Config

@dataclass
class Config:
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    tta: TTAConfig = field(default_factory=TTAConfig)
    swa: SWAConfig = field(default_factory=SWAConfig)
    semi: SemiConfig = field(default_factory=SemiConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_dict(cls, d: dict) -> Config:
        return cls(
            seed=d.get("seed", 42),
            data=DataConfig(**_filter_fields(DataConfig, d.get("data", {}))),
            dataloader=DataLoaderConfig(**_filter_fields(DataLoaderConfig, d.get("dataloader", {}))),
            image=ImageConfig.from_dict(d.get("image", {})),
            augment=AugmentConfig.from_dict(d.get("augment", {})),
            train=TrainConfig.from_dict(d.get("train", {})),
            tta=TTAConfig(**_filter_fields(TTAConfig, d.get("tta", {}))),
            swa=SWAConfig(**_filter_fields(SWAConfig, d.get("swa", {}))),
            semi=SemiConfig.from_dict(d.get("semi", {})),
            output=OutputConfig(**_filter_fields(OutputConfig, d.get("output", {}))),
        )

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(config_path: str) -> Config:
    """
    Load JSON config file and return a Config dataclass.
    """
    with open(config_path, "r") as f:
        raw = json.load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError("config json must be an object/dict")
    return Config.from_dict(raw)
