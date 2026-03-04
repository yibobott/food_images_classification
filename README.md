# HW2 - Food-11 Image Classification

Food-11 dataset (11 classes) image classification with semi-supervised learning.

## Quick Start

```bash
pip install -r requirements.txt
python HW2_Improved_code.py --config config.json
```

## Project Structure

```
kaggle/
├── HW2_Improved_code.py   # Main training script
├── config.json             # Hyperparameter configuration
├── requirements.txt        # Python dependencies
└── food11/                 # Dataset (not included)
    ├── training/
    │   ├── labeled/        # 2,970 labeled images
    │   └── unlabeled/      # 6,786 unlabeled images
    ├── validation/         # 660 images
    └── testing/            # 3,347 images
```

## Differences from Baseline

| Feature | Baseline (`HW2_Baseline code.ipynb`) | Improved (`HW2_Improved_code.py`) |
|---------|--------------------------------------|-----------------------------------|
| **Model** | 3-layer CNN (Conv→BN→ReLU→MaxPool ×3, FC) | CIFAR-style ResNet-18 + SE attention (11.26M params) |
| **Image Size** | 128×128 | 160×160 (configurable) |
| **Data Augmentation** | Resize only | RandomResizedCrop, HorizontalFlip, Rotation, ColorJitter, RandomErasing |
| **Optimizer** | Adam (lr=3e-4, wd=1e-5) | AdamW (lr=7e-4, wd=5e-4) |
| **LR Scheduler** | None | OneCycleLR (max_lr=0.003) |
| **Loss** | CrossEntropyLoss | CrossEntropyLoss + Label Smoothing (0.05) |
| **Semi-supervised** | Skeleton only (not implemented) | FixMatch-style: EMA teacher + pseudo labeling with confidence threshold |
| **EMA** | None | Exponential Moving Average teacher (decay=0.995) |
| **Unlabeled Aug** | None | RandAugment (strong) + weak augmentation pair |
| **CutMix** | None | CutMix on labeled data (alpha=0.2) |
| **Gradient Accumulation** | None | accum_steps=2 (effective batch=64) |
| **TTA** | None | Test-Time Augmentation (10 augmentations, logit averaging) |
| **SWA** | None | Stochastic Weight Averaging (last 10% epochs) |
| **Dropout** | None | 0.2 (before FC layer) |
| **Batch Size** | 128 | 32 (labeled) / 64 (unlabeled) |
| **Epochs** | 1 (placeholder) | 150 |
| **Config** | Hardcoded | External JSON config with defaults |
| **Logging** | print | File + console logging with training summary |

### Key Architectural Improvements

1. **ResNet-18 with SE Blocks**: Replaced the simple 3-layer CNN with a CIFAR-style ResNet-18 (no pretrained weights, no maxpool in stem). Each BasicBlock includes a Squeeze-and-Excitation (SE) module for channel-wise attention recalibration.

2. **FixMatch Semi-supervised Learning**: The baseline only provided an empty `get_pseudo_labels` skeleton. The improved version implements a full FixMatch-style pipeline:
   - EMA teacher generates pseudo labels on weakly-augmented unlabeled data
   - Student learns from strongly-augmented (RandAugment) unlabeled data
   - Confidence threshold (0.85) filters unreliable pseudo labels
   - Lambda_u ramp-up over 30 epochs for stable training

3. **OneCycleLR Scheduler**: Replaced no-scheduler training with OneCycleLR, which provides a warm-up phase followed by cosine annealing, significantly improving convergence efficiency.

4. **CutMix Augmentation**: Applies region-level mixing on labeled data, encouraging the model to learn from partial views — particularly effective for food/texture classification.

5. **Test-Time Augmentation (TTA)**: Averages logits from 10 augmented views + 1 base view at inference, improving prediction robustness.

## Best Run Summary

**Run ID**: `20260304-023200`

### Configuration

| Parameter | Value |
|-----------|-------|
| Image Size | 160×160 |
| Batch Size | 32 (labeled) / 64 (unlabeled) |
| Gradient Accumulation | 2 (effective batch = 64) |
| Epochs | 150 |
| Scheduler | OneCycleLR (max_lr=0.003) |
| EMA Decay | 0.995 |
| Mix Mode | CutMix (alpha=0.2) |
| Pseudo Threshold | 0.85 |
| Lambda_u | 0.4 |
| Label Smoothing | 0.05 |
| Dropout | 0.2 |
| SWA | Enabled (last 10% epochs, lr=1e-5) |
| TTA | 10 augmentations |

### Best Result

| Model | Valid Acc | TTA Valid Acc | Kaggle Test Acc |
|-------|----------|---------------|-----------------|
| **EMA Best** (epoch 139) | **0.8524** | **0.8652** | **0.84886** |
| SWA | 0.8128 | 0.8379 | 0.84587 |

### Output Files

```
Best-Result-20260304-023200/
├── swa-model-20260304-023200.pt        # SWA model checkpoint
├── predict-20260304-023200.csv         # EMA test predictions (3,347 samples)
├── swa-predict-20260304-023200.csv     # SWA test predictions (3,347 samples)
├── log-20260304-023200.txt             # Full training log
└── swa-predict-kaggle-score.png        # Kaggle submission screenshot
```

### Training Summary

```
[Best EMA Model]
  Epoch:       139/150
  Train Loss:  0.9349
  Train Acc:   0.9987
  Valid Loss:  0.7406
  Valid Acc:   0.8524
  Valid (TTA): 0.8652

[SWA Model]
  Avg Range:   epoch 135~150 (15 epochs)
  Valid Loss:  1.0724
  Valid Acc:   0.8128
  Valid (TTA): 0.8379
```

## Dependencies

```
numpy
torch
torchvision
Pillow
tqdm
```

No external packages beyond the baseline requirements. All model architectures are implemented from scratch without pretrained weights.
