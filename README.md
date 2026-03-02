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

## Configuration Overview

The training behavior is fully controlled by `config.json`.
Below is a description of the main configuration fields.

Training Settings:

| Key                     | Description                                  |
| ----------------------- | -------------------------------------------- |
| `train.n_epochs`        | Total number of training epochs              |
| `train.lr`              | Initial learning rate                        |
| `train.weight_decay`    | Weight decay coefficient for AdamW optimizer |
| `train.label_smoothing` | Label smoothing factor in `CrossEntropyLoss` |
| `train.mixup.enabled`   | Enable MixUp for labeled batches             |
| `train.mixup.alpha`     | Beta distribution parameter for MixUp        |

Semi-Supervised Settings (FixMatch-style):

| Key                          | Description                                             |
| ---------------------------- | ------------------------------------------------------- |
| `semi.enabled`               | Enable semi-supervised learning                         |
| `semi.warmup_epochs`         | Number of epochs trained with labeled data only         |
| `semi.pseudo_threshold`      | Confidence threshold for pseudo-label filtering         |
| `semi.lambda_u`              | Weight for unsupervised loss                            |
| `semi.lambda_u_ramp_epochs`  | Epochs to gradually ramp up `lambda_u`                  |
| `semi.unsup_batch_size`      | Batch size for unlabeled data                           |
| `semi.ema.decay`             | Exponential Moving Average decay rate for teacher model |
| `semi.randaugment_num_ops`   | Number of operations used in RandAugment                |
| `semi.randaugment_magnitude` | Magnitude of RandAugment transformations                |

Data & Dataloader Settings:

| Key                      | Description                                  |
| ------------------------ | -------------------------------------------- |
| `data.train_labeled`     | Path to labeled training dataset             |
| `data.train_unlabeled`   | Path to unlabeled training dataset           |
| `data.valid`             | Path to validation dataset                   |
| `data.test`              | Path to test dataset                         |
| `dataloader.batch_size`  | Batch size for labeled data                  |
| `dataloader.num_workers` | Number of worker processes for data loading  |
| `dataloader.pin_memory`  | Enable pinned memory for faster GPU transfer |


Image & Augmentation Settings:

| Key                                 | Description                                              |
| ----------------------------------- | -------------------------------------------------------- |
| `image.img_size`                    | Input image size                                         |
| `image.mean`                        | Normalization mean                                       |
| `image.std`                         | Normalization standard deviation                         |
| `augment.random_resized_crop_scale` | Scale range for `RandomResizedCrop`                      |
| `augment.random_resized_crop_ratio` | Aspect ratio range for `RandomResizedCrop`               |
| `augment.horizontal_flip_p`         | Probability of horizontal flip                           |
| `augment.rotation_deg`              | Maximum rotation angle                                   |
| `augment.color_jitter.*`            | Parameters for brightness, contrast, saturation, and hue |

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

## Key Differences from Baseline

### 1. Stronger Model Backbone (CIFAR-style ResNet-18, from scratch)
- Replaces the baseline CNN with a **custom ResNet-18 (BasicBlock)** implementation.
- Uses a **CIFAR-style stem**:
  - `3x3 conv, stride=1`
  - **no maxpool**
- No pre-trained weights are used.

### 2. Semi-Supervised Learning (FixMatch-style)
- Uses both labeled and unlabeled images during training.
- For each unlabeled image, the dataset returns:
  - **weakly-augmented** view (for teacher prediction)
  - **strongly-augmented** view (for student training)
- The teacher produces pseudo labels on weak views; the student is trained to match them on strong views.
- A confidence threshold (`pseudo_threshold`) filters pseudo labels.

### 3. EMA Teacher (Exponential Moving Average)
- Maintains a teacher model as an EMA of the student parameters:
  - `teacher = decay * teacher + (1 - decay) * student`
- Validation is performed using the **EMA teacher**, which is often more stable than the raw student.

### 4. Additional Data Augmentation Techniques
- **Labeled training augmentation** includes:
  - RandomResizedCrop
  - HorizontalFlip
  - Rotation
  - ColorJitter
- **Unlabeled strong augmentation** uses:
  - RandomResizedCrop + HorizontalFlip
  - **RandAugment** (if supported by current torchvision)
- **Unlabeled weak augmentation** is kept mild to stabilize pseudo labels.

### 5. MixUp on Labeled Data Only
- MixUp is optionally enabled **only on labeled batches** (does not modify unlabeled branch).
- This reduces extra variables in semi-supervised training while improving robustness on labeled supervision.

### 6. Improved Training Engineering
- JSON-based configuration system (`config.json`) for reproducible experiments.
- Cosine learning rate schedule with AdamW optimizer.
- Automatic best checkpoint saving (EMA weights) based on validation accuracy.
- Detailed logging:
  - console output
  - saved log file `log-<date>.txt`
- Prediction output is saved with a date suffix to avoid overwriting.
