# HW2 - Food-11 Image Classification

Food-11 dataset (11 classes) image classification with semi-supervised learning, progressive resizing, and pseudo-label self-training.

## Quick Start

```bash
pip install -r requirements.txt
python train.py --config config.json
```

## Project Structure

```
kaggle/
├── train.py                    # Entry point: progressive resize + self-training pipeline
├── config.json                 # Hyperparameter configuration
├── requirements.txt            # Python dependencies
├── utils/
│   ├── config.py               # Config dataclasses (incl. ProgressiveResizeConfig, SelfTrainingConfig)
│   ├── logger.py               # build_logger
│   └── misc.py                 # set_seed, rgb_loader, rand_bbox
├── models/
│   ├── resnet.py               # CIFAR-style ResNet-18 + SE attention
│   └── ema.py                  # EMA teacher
├── data/
│   ├── transforms.py           # Transforms + build_transforms (supports img_size_override)
│   └── datasets.py             # DataBundle, UnlabeledPairDataset, PseudoLabeledDataset
├── engine/
│   ├── trainer.py              # BestMetrics + train_one_epoch / valid_one_epoch
│   └── inference.py            # RunSummary + TTA + inference
└── food11/                     # Dataset (not included)
    ├── training/
    │   ├── labeled/            # 2,970 labeled images
    │   └── unlabeled/          # 6,786 unlabeled images
    ├── validation/             # 660 images
    └── testing/                # 3,347 images
```

## Training Pipeline

The training runs in two phases:

### Phase 1: Progressive Resizing + Semi-supervised Learning

1. **Stage 1** — Train on 160×160 images for 100 epochs with FixMatch semi-supervised learning
2. **Stage 2** — Switch to 224×224 images for 150 epochs, continuing from Stage 1 weights

Both stages use the EMA teacher to generate online pseudo-labels for unlabeled data, with confidence-based masking and unsupervised loss ramp-up.

### Phase 2: Pseudo-Label Self-Training

1. Use the Phase 1 EMA teacher to generate offline pseudo-labels for all unlabeled samples above a confidence threshold (0.80)
2. Combine labeled data (2,970) with pseudo-labeled data (~5,648) into a single supervised dataset (8,618 total)
3. Train a fresh model from scratch on the combined dataset for 150 epochs (supervised only, no semi-supervised branch)

## Differences from Baseline

| Feature | Baseline (`HW2_Baseline code.ipynb`) | Improved |
|---------|--------------------------------------|----------|
| **Model** | 3-layer CNN (Conv→BN→ReLU→MaxPool ×3, FC) | CIFAR-style ResNet-18 + SE attention (11.26M params) |
| **Image Size** | 128×128 | Progressive: 160→224 |
| **Data Augmentation** | Resize only | RandomResizedCrop, HorizontalFlip, Rotation, ColorJitter, RandomErasing |
| **Optimizer** | Adam (lr=3e-4, wd=1e-5) | AdamW (lr=7e-4, wd=5e-4) |
| **LR Scheduler** | None | OneCycleLR (max_lr=0.003) |
| **Loss** | CrossEntropyLoss | CrossEntropyLoss + Label Smoothing (0.05) |
| **Semi-supervised** | Skeleton only (not implemented) | FixMatch-style: EMA teacher + pseudo labeling with confidence threshold |
| **Self-Training** | None | Phase 2: offline pseudo-labels from EMA teacher, retrain from scratch |
| **EMA** | None | Exponential Moving Average teacher (decay=0.995) |
| **Unlabeled Aug** | None | RandAugment (strong) + weak augmentation pair |
| **CutMix** | None | CutMix on labeled data (alpha=0.2) |
| **Gradient Accumulation** | None | accum_steps=4 (effective batch=64) |
| **TTA** | None | Test-Time Augmentation (10 augmentations, logit averaging) |
| **SWA** | None | Stochastic Weight Averaging (last 10% epochs) |
| **Dropout** | None | 0.2 (before FC layer) |
| **AMP** | None | Automatic Mixed Precision (fp16) |
| **Batch Size** | 128 | 16 (labeled) / 32 (unlabeled) |
| **Epochs** | 1 (placeholder) | 250 (Phase 1) + 150 (Phase 2) = 400 total |
| **Config** | Hardcoded | External JSON config with defaults |
| **Logging** | print | File + console logging with training summary |

### Key Architectural Improvements

1. **ResNet-18 with SE Blocks**: Replaced the simple 3-layer CNN with a CIFAR-style ResNet-18 (no pretrained weights, no maxpool in stem). Each BasicBlock includes a Squeeze-and-Excitation (SE) module for channel-wise attention recalibration.

2. **FixMatch Semi-supervised Learning**: The baseline only provided an empty `get_pseudo_labels` skeleton. The improved version implements a full FixMatch-style pipeline:
   - EMA teacher generates pseudo labels on weakly-augmented unlabeled data
   - Student learns from strongly-augmented (RandAugment) unlabeled data
   - Confidence threshold (0.80) filters unreliable pseudo labels
   - Lambda_u ramp-up over 30 epochs for stable training

3. **Progressive Resizing**: Train first on smaller images (160×160) then switch to full resolution (224×224). This provides faster early convergence and acts as a form of regularization.

4. **Pseudo-Label Self-Training**: After Phase 1, the trained EMA teacher generates high-confidence pseudo-labels for unlabeled data. A fresh model is then trained on the combined labeled + pseudo-labeled dataset, effectively tripling the supervised data.

5. **OneCycleLR Scheduler**: Provides a warm-up phase followed by cosine annealing, significantly improving convergence efficiency.

6. **CutMix Augmentation**: Applies region-level mixing on labeled data, encouraging the model to learn from partial views — particularly effective for food/texture classification.

7. **Test-Time Augmentation (TTA)**: Averages logits from 10 augmented views + 1 base view at inference, improving prediction robustness.

## Best Run Summary

**Run ID**: `20260305-122606`

### Configuration

| Parameter | Value |
|-----------|-------|
| Image Size | 160→224 (progressive) |
| Batch Size | 16 (labeled) / 32 (unlabeled) |
| Gradient Accumulation | 4 (effective batch=64) |
| Phase 1 Epochs | 250 (100 @ img160 + 150 @ img224) |
| Phase 2 Epochs | 150 (supervised only) |
| Scheduler | OneCycleLR (max_lr=0.003) |
| EMA Decay | 0.995 |
| Mix Mode | CutMix (alpha=0.2) |
| Pseudo Threshold | 0.80 |
| Lambda_u | 0.4 |
| Label Smoothing | 0.05 |
| Dropout | 0.2 |
| SWA | Enabled (last 10% epochs, lr=1e-5) |
| TTA | 10 augmentations |
| AMP | Enabled |
| Self-Training Threshold | 0.80 (5,648/6,786 pseudo-labeled) |

### Best Result

| Model | Phase | Valid Acc | TTA Valid Acc | Kaggle Test Acc |
|-------|-------|----------|---------------|-----------------|
| **EMA Best** (ep 103) | **Phase 2** | **0.9048** | **0.9106** | **0.89068** |
| SWA | Phase 2 | 0.8929 | 0.8985 | — |
| EMA Best (ep 211) | Phase 1 | 0.8929 | 0.8894 | — |
| SWA | Phase 1 | 0.8765 | 0.8924 | — |

### Output Files

```
Result-20260305-122606/
├── best-model-20260305-122606.pt           # Phase 1 EMA best checkpoint
├── swa-model-20260305-122606.pt            # Phase 1 SWA checkpoint
├── predict-20260305-122606.csv             # Phase 1 EMA TTA predictions
├── swa-predict-20260305-122606.csv         # Phase 1 SWA TTA predictions
├── best-model-20260305-122606-ST.pt        # Phase 2 EMA best checkpoint
├── swa-model-20260305-122606-ST.pt         # Phase 2 SWA checkpoint
├── predict-20260305-122606-ST.csv          # Phase 2 EMA TTA predictions (best)
├── swa-predict-20260305-122606-ST.csv      # Phase 2 SWA TTA predictions
└── log-20260305-122606.txt                 # Full training log
```

### Training Summary

```
Phase 1 (Progressive Resizing + Semi-supervised):
  [Best EMA Model]
    Epoch:       211/250
    Valid Acc:   0.8929
    Valid (TTA): 0.8894
  [SWA Model]
    Avg Range:   epoch 235~250
    Valid Acc:   0.8765
    Valid (TTA): 0.8924

Phase 2 (Self-Training):
  Pseudo-labels: 5,648/6,786 (83.2%, avg conf 0.8924)
  [Best EMA Model]
    Epoch:       103/150
    Valid Acc:   0.9048
    Valid (TTA): 0.9106
  [SWA Model]
    Avg Range:   epoch 135~150
    Valid Acc:   0.8929
    Valid (TTA): 0.8985
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
