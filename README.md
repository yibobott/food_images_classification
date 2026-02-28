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

## Notes (macOS DataLoader)

- On macOS, `DataLoader(num_workers > 0)` uses multiprocessing with a start method that requires objects to be picklable.
- The script uses a top-level `rgb_loader` (not lambdas) to avoid pickling errors.
- If you still encounter multiprocessing issues, try setting `num_workers` to `0` in `config.json`.
