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
    },
    "train": {
        "n_epochs": 25,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "label_smoothing": 0.1,
    },
    "semi": {
        "enabled": True,
        "warmup_epochs": 5,
        "pseudo_threshold": 0.95,
        "semi_weight": 0.5,
        "pseudo_batch_size": 256,
    },
    "output": {
        "best_path": "best_model.pt",
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out

class SmallResNet(nn.Module):
    def __init__(self, num_classes=11):
        super().__init__()
        # from [batch_size, 3, 128, 128] to [batch_size, 64, 128, 128]
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # layer1 output: [batch_size, 64, 128, 128]
        self.layer1 = nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64))
        # layer2 output: [batch_size, 128, 64, 64]
        self.layer2 = nn.Sequential(BasicBlock(64, 128, stride=2), BasicBlock(128, 128))
        # layer3 output: [batch_size, 256, 32, 32]
        self.layer3 = nn.Sequential(BasicBlock(128, 256, stride=2), BasicBlock(256, 256))
        # layer4 output: [batch_size, 512, 16, 16]
        self.layer4 = nn.Sequential(BasicBlock(256, 512, stride=2), BasicBlock(512, 512))
        # pool output: [batch_size, 512, 1, 1]
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # output: [batch_size, 11]
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # output: [batch_size, 512]
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        return x



class PseudoLabelDataset(Dataset):
    def __init__(self, base_dataset, indices, pseudo_labels, transform=None):
        self.base = base_dataset
        self.indices = list(indices)
        self.pseudo_labels = torch.as_tensor(pseudo_labels, dtype=torch.long)
        self.transform = transform

        # base_dataset.samples: List[(path, target)]
        self.paths = [self.base.samples[i][0] for i in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, self.pseudo_labels[idx].item()

@torch.no_grad()
def get_pseudo_labels(
    unlabeled_ds,
    model,
    device,
    img_size,
    mean,
    std,
    train_tfm,
    num_workers,
    pin_memory,
    threshold=0.95,
    batch_size=256,
):
    model.eval()

    # Use a weak transform for more stable pseudo labels (no strong augmentation)
    weak_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Build a lightweight view of the unlabeled dataset with weak transform
    tmp = DatasetFolder(unlabeled_ds.root, loader=lambda x: Image.open(x).convert('RGB'),
                        extensions=("jpg", "jpeg", "png"), transform=weak_tfm)
    loader = DataLoader(tmp, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    all_conf = []
    all_pred = []

    softmax = nn.Softmax(dim=-1)
    for imgs, _ in tqdm(loader, desc="Pseudo-labeling", leave=False):
        logits = model(imgs.to(device))
        probs = softmax(logits)
        conf, pred = probs.max(dim=-1)
        all_conf.append(conf.cpu())
        all_pred.append(pred.cpu())

    conf = torch.cat(all_conf)
    pred = torch.cat(all_pred)

    mask = conf >= threshold
    indices = mask.nonzero(as_tuple=False).squeeze(1).tolist()
    pseudo_labels = pred[mask].tolist()

    pseudo_ds = PseudoLabelDataset(tmp, indices, pseudo_labels, transform=train_tfm)  # train_tfm = strong augmentation
    model.train()
    return pseudo_ds, float(mask.float().mean().item()), len(pseudo_ds)



def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    losses, accs = [], []
    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs = imgs.to(device)
        labels = labels.to(device)

        logits = model(imgs)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        acc = (logits.argmax(dim=-1) == labels).float().mean().item()
        losses.append(loss.item())
        accs.append(acc)

    return float(np.mean(losses)), float(np.mean(accs))

@torch.no_grad()
def valid_one_epoch(model, loader, criterion, device):
    model.eval()
    losses, accs = [], []
    for imgs, labels in tqdm(loader, desc="Valid", leave=False):
        imgs = imgs.to(device)
        labels = labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        acc = (logits.argmax(dim=-1) == labels).float().mean().item()
        losses.append(loss.item())
        accs.append(acc)
    return float(np.mean(losses)), float(np.mean(accs))


def main(config_path: str):
    # load config
    cfg = load_config(config_path)
    print(json.dumps(cfg, indent=2, sort_keys=True))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    set_seed(int(cfg["seed"]))

    # ===========
    # Dataset, Data Loader, and Transforms
    # ===========

    img_size = int(cfg["image"]["img_size"])
    batch_size = int(cfg["dataloader"]["batch_size"])
    num_workers = int(cfg["dataloader"]["num_workers"])
    pin_memory = bool(cfg["dataloader"]["pin_memory"])

    mean = tuple(float(x) for x in cfg["image"]["mean"])
    std = tuple(float(x) for x in cfg["image"]["std"])

    cj = cfg["augment"]["color_jitter"]
    rrc_scale = tuple(float(x) for x in cfg["augment"]["random_resized_crop_scale"])
    rrc_ratio = tuple(float(x) for x in cfg["augment"]["random_resized_crop_ratio"])

    # data augmentation in training
    train_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
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
    ])

    test_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Construct datasets
    train_set = DatasetFolder(
        cfg["data"]["train_labeled"],
        loader=lambda x: Image.open(x).convert('RGB'),
        extensions=("jpg", "jpeg", "png"),
        transform=train_tfm,
    )
    valid_set = DatasetFolder(
        cfg["data"]["valid"],
        loader=lambda x: Image.open(x).convert('RGB'),
        extensions=("jpg", "jpeg", "png"),
        transform=test_tfm,
    )
    unlabeled_set = DatasetFolder(
        cfg["data"]["train_unlabeled"],
        loader=lambda x: Image.open(x).convert('RGB'),
        extensions=("jpg", "jpeg", "png"),
        transform=train_tfm,
    )
    test_set = DatasetFolder(
        cfg["data"]["test"],
        loader=lambda x: Image.open(x).convert('RGB'),
        extensions=("jpg", "jpeg", "png"),
        transform=test_tfm,
    )

    # Construct data loaders
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    print("train labeled:", len(train_set))
    print("train unlabeled:", len(unlabeled_set))
    print("valid:", len(valid_set))
    print("test:", len(test_set))

    # ===========
    # Model
    # ===========

    model = SmallResNet(num_classes=11).to(device)
    print("#params:", sum(p.numel() for p in model.parameters()) / 1e6, "M")

    n_epochs = int(cfg["train"]["n_epochs"])
    lr = float(cfg["train"]["lr"])
    weight_decay = float(cfg["train"]["weight_decay"])
    label_smoothing = float(cfg["train"]["label_smoothing"])

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    # ===========
    # Training & Validation
    # ===========

    do_semi = bool(cfg["semi"]["enabled"])
    warmup_epochs = int(cfg["semi"]["warmup_epochs"])
    pseudo_threshold = float(cfg["semi"]["pseudo_threshold"])
    semi_weight = float(cfg["semi"]["semi_weight"])
    pseudo_batch_size = int(cfg["semi"]["pseudo_batch_size"])

    best_acc = 0.0
    best_path = str(cfg["output"]["best_path"])

    # do semi-supervised learning after warmup epochs
    for epoch in range(1, n_epochs + 1):
        if do_semi and epoch > warmup_epochs:
            pseudo_ds, keep_ratio, keep_n = get_pseudo_labels(
                unlabeled_set,
                model,
                device,
                img_size,
                mean,
                std,
                train_tfm,
                num_workers,
                pin_memory,
                threshold=pseudo_threshold,
                batch_size=pseudo_batch_size,
            )
            print(f"[Semi] epoch {epoch}: keep_ratio={keep_ratio:.3f}, kept={keep_n}")
            concat_ds = ConcatDataset([train_set, pseudo_ds])
            train_loader_epoch = DataLoader(
                concat_ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
        else:
            train_loader_epoch = train_loader

        tr_loss, tr_acc = train_one_epoch(model, train_loader_epoch, optimizer, criterion, device)
        va_loss, va_acc = valid_one_epoch(model, valid_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch {epoch:02d}/{n_epochs} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | valid loss {va_loss:.4f} acc {va_acc:.4f} | lr {scheduler.get_last_lr()[0]:.2e}"
        )

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), best_path)
            print(f"  -> saved best: {best_acc:.4f}")

    # ===========
    # Testing
    # ===========

    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    predictions = []
    with torch.no_grad():
        for imgs, _ in tqdm(test_loader, desc="Test"):
            logits = model(imgs.to(device))
            pred = logits.argmax(dim=-1).cpu().numpy().tolist()
            predictions.extend(pred)

    print("#pred:", len(predictions))

    out_path = str(cfg["output"]["predict_path"])
    with open(out_path, "w") as f:
        f.write("Id,Category\n")
        for i, pred in enumerate(predictions):
            f.write(f"{i},{pred}\n")

    print("Saved:", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json")
    args = parser.parse_args()
    
    main(args.config)
