# HW2

## Requirements

- Python
- GPU is optional (CUDA if available)

Install packages:

```bash
pip install -r requirements.txt
```

## Dataset

Expected folder structure (relative to `kaggle/`):

```text
food11/
  training/
    labeled/
    unlabeled/
  validation/
  testing/
```

## Run

Default run

```bash
python HW2_Improved_code.py
```

Run with a specific config:

```bash
python HW2_Improved_code.py --config config.json
```

At startup the script prints:

- The run timestamp `date`
- The merged runtime config (pretty-printed JSON)

## Configuration (`config.json`)

Key fields:

- `seed`: random seed
- `data.*`: dataset paths
- `dataloader.*`: DataLoader settings
  - `batch_size`
  - `num_workers`
  - `pin_memory`
- `image.*`:
  - `img_size`
  - `mean`, `std`
- `augment.*`: training data augmentation parameters
- `train.*`: training hyperparameters
- `semi.*`: semi-supervised settings
  - `enabled`
  - `warmup_epochs`
  - `pseudo_threshold`
  - `pseudo_batch_size`
  - `pseudo_every` (generate pseudo labels every N epochs after warmup)
- `output.*`:
  - `best_path`
  - `predict_path`

You can edit `config.json` to tune hyperparameters without changing code.

## Outputs

Each run is stamped with a `date` (format: `YYYYmmdd-HHMMSS`).

- Log file:
  - `log-<date>.txt`
- Checkpoint:
  - `<best_path without extension>-<date>.pt`
- Prediction CSV:
  - `<predict_path without extension>-<date>.csv`

Example:

```text
log-20260301-001530.txt
best-model-20260301-001530.pt
predict-20260301-001530.csv
```

## Differences from Baseline Code

Compared with the provided baseline implementation, our improved version introduces several enhancements in model design, training strategy, and engineering structure:

### 1. Improved Model Architecture

- The baseline model uses a relatively simple CNN structure.
- Our implementation replaces it with ResNet18, a deeper convolutional network with residual connections.
- Residual blocks help alleviate the vanishing gradient problem and enable more stable training.
- The final fully connected layer is modified to match the target number of classes.

In particular, we implement a CIFAR-style ResNet-18 from scratch (no pretrained weights):

- A 3x3 convolution stem with stride=1 and **no max-pooling**.
- Four residual stages with `BasicBlock` in a `2-2-2-2` layout (ResNet-18).
- Downsampling is done by using stride=2 at the beginning of stage 2/3/4.
- For `128x128` inputs, the spatial resolution changes as:
  - after stem: `128x128`
  - after stage2: `64x64`
  - after stage3: `32x32`
  - after stage4: `16x16`
- An `AdaptiveAvgPool2d((1, 1))` reduces features to `1x1`, followed by a linear classifier for 11 classes.

This significantly improves feature extraction capability and model expressiveness.

### 2. Configurable Training Pipeline

- The improved version supports a JSON-based configuration system (`config.json`).
- Key hyperparameters (learning rate, batch size, augmentation settings, semi-supervised options, etc.) can be modified without changing the source code.
- This makes experiments more flexible and reproducible.

### 3. Semi-Supervised Learning (Pseudo-Labeling)

- The baseline only uses labeled data.
- Our version optionally incorporates unlabeled data through pseudo-labeling:
  - Warm-up epochs before pseudo-label generation
  - Confidence threshold filtering
  - Periodic pseudo-label updates
- This allows better utilization of the unlabeled dataset and improves generalization performance.

### 4. Enhanced Data Augmentation

- Additional data augmentation techniques are applied during training, such as:
  - Random horizontal flipping
  - Random cropping and resizing
  - Color jitter (brightness / contrast / saturation adjustments)
  - Random rotation (if enabled in config)
- Augmentation parameters are configurable via `config.json`.
- These techniques increase data diversity, reduce overfitting, and improve model generalization performance.

### 5. Logging and Checkpoint Management

- Each run automatically:
  - Prints and logs the merged runtime configuration
  - Saves timestamped checkpoints
  - Saves prediction results with unique filenames
- This improves experiment tracking and reproducibility.


