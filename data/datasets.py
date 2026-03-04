from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import DatasetFolder

from utils.config import Config
from utils.misc import rgb_loader
from data.transforms import Transforms


class UnlabeledPairDataset(Dataset):
    """
    Return (x_weak, x_strong) for the same image path.
    """
    def __init__(self, root: str, weak_tfm, strong_tfm) -> None:
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

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        xw = self.weak_tfm(img) if self.weak_tfm is not None else img
        xs = self.strong_tfm(img) if self.strong_tfm is not None else img
        return xw, xs


@dataclass
class DataBundle:
    train_set: DatasetFolder
    valid_set: DatasetFolder
    unlabeled_set: Optional[UnlabeledPairDataset]
    test_set: DatasetFolder
    train_loader: DataLoader
    valid_loader: DataLoader
    unlabeled_loader: Optional[DataLoader]
    test_loader: DataLoader
    batch_size: int
    num_workers: int
    pin_memory: bool
    do_semi: bool


def build_datasets_and_loaders(cfg: Config, tfms: Transforms, logger: logging.Logger) -> DataBundle:
    batch_size = cfg.dataloader.batch_size
    num_workers = cfg.dataloader.num_workers
    pin_memory = cfg.dataloader.pin_memory
    do_semi = cfg.semi.enabled

    train_set = DatasetFolder(
        cfg.data.train_labeled, loader=rgb_loader,
        extensions=("jpg", "jpeg", "png"), transform=tfms.train,
    )
    valid_set = DatasetFolder(
        cfg.data.valid, loader=rgb_loader,
        extensions=("jpg", "jpeg", "png"), transform=tfms.test,
    )
    unlabeled_set: Optional[UnlabeledPairDataset] = None
    if do_semi:
        unlabeled_set = UnlabeledPairDataset(
            cfg.data.train_unlabeled,
            weak_tfm=tfms.weak, strong_tfm=tfms.unlabeled_strong,
        )
    test_set = DatasetFolder(
        cfg.data.test, loader=rgb_loader,
        extensions=("jpg", "jpeg", "png"), transform=tfms.test,
    )

    loader_kwargs: dict = {}
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
    unlabeled_loader: Optional[DataLoader] = None
    if do_semi:
        unsup_bs = cfg.semi.unsup_batch_size
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

    return DataBundle(
        train_set=train_set, valid_set=valid_set,
        unlabeled_set=unlabeled_set, test_set=test_set,
        train_loader=train_loader, valid_loader=valid_loader,
        unlabeled_loader=unlabeled_loader, test_loader=test_loader,
        batch_size=batch_size, num_workers=num_workers,
        pin_memory=pin_memory, do_semi=do_semi,
    )
