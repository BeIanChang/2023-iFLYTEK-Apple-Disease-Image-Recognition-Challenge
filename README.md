# 2023 iFLYTEK Apple Disease Image Recognition Challenge

End-to-end PyTorch pipeline for the 2023 iFLYTEK Apple Disease Image Recognition Challenge (<https://challenge.xfyun.cn/topic/info?type=apple-diseases&option=ssgy>). Achieved ≈90% validation accuracy with ResNet-34 backbone using OneCycleLR and data augmentation. Designed configurable training and inference framework with modular code and reproducible YAML configs.

## Repository Layout
- `train.py`: full training pipeline (data loading, augmentation, training loop, checkpoints, plots, predictions).
- `test_net.py`: quick utility that prints the model summary for a given backbone.
- `config/config0.1.yaml`: default experiment settings loaded automatically by `train.py`.
- `module/`: reusable components
  - `dataset.py`: dataset split utilities and device-aware data loaders.
  - `network.py`: wrapper that instantiates a selectable ResNet backbone with task-specific heads.
  - `resnet.py`: custom lightweight ResNet and torch ResNet variants.
  - `evaluation.py`: validation helpers, accuracy metric, and LR scheduler utilities.
- `data_method/`: experiment helpers (config loading, checkpointing, plotting, CSV export).
- `data_check.ipynb`: exploratory notebook (optional).

## Setup
### Prerequisites
- Python ≥ 3.8
- CUDA-capable GPU recommended (training falls back to CPU if unavailable).

### Install Dependencies
Create and activate a virtual environment, then install the required packages:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # pick the wheel that matches your CUDA version
pip install torchsummary tqdm matplotlib pyyaml numpy pandas scikit-learn imbalanced-learn
```

Adjust the first line if you are using CPU-only PyTorch.

## Data Preparation
The training script expects ImageFolder-compatible directory structures:

```
train/
  class_0/
    img001.jpg
    ...
  class_1/
    ...
test/
  test/
    0001.jpg
    0002.jpg
    ...
```

- Training images must be sorted into per-class folders under `train/`.
- The provided submission helper assumes the challenge test set lives under `test/test/` with image filenames matching the competition UUIDs.

## Configuring Experiments
Default hyperparameters are stored in `config/config0.1.yaml`. Every key in the YAML file is exposed as a command-line flag. You can either edit the YAML directly or override values when launching training:

```bash
python train.py \
  --train_dir ./train \
  --test_dir ./test \
  --ratio 0.85 \
  --resnet_type ResNet34 \
  --epochs 60 \
  --batch_size 128
```

Key options:
- `ratio`: fraction of `train/` used for training versus validation.
- `resnet_type`: `myResNet`, `ResNet18`, `ResNet34`, or `ResNet50`.
- `learning_rate`, `weight_decay`, `grad_clip`: optimizer settings.
- `model_path` and `plt_path`: output directories for checkpoints and plots.
- `reload`: set to `True` and adjust `start_epoch` to resume from checkpoints.

## Running Training
1. Ensure data and config are prepared.
2. Launch training:
   ```bash
   python train.py
   ```
   By default, parameters from `config/config0.1.yaml` are used.
3. The script will:
   - Split the dataset into train/validation according to `ratio`.
   - Apply random resized crops (size `picture_size`) and horizontal flips.
   - Train with OneCycleLR scheduling and optional gradient clipping.
   - Periodically save checkpoints to `model_save/<model_name>/checkpoint_<epoch>.tar`.
   - Refresh the data loaders every five epochs to reshuffle the split.

## Outputs and Evaluation
- Validation metrics per epoch are printed from `Network.training_epoch_end` in `module/network.py`.
- PNG plots (`acc.png`, `loss.png`, `lr.png`) are written to `plt_save/<model_name>/`.
- After training, `data_method/result_to_file.py` generates `results_<model_name>.csv` containing the test predictions (`uuid,label`).

## Inference on New Images
To run inference on additional images without retraining, load a checkpoint and feed preprocessed tensors through `Network.forward`. The helper `data_method/result_to_file.predict` demonstrates the expected transform pipeline and CSV export.

## Utilities
- `test_net.py` prints a `torchsummary` overview of the configured backbone. Update the `resnet_type` and `class_num` inside the script to match your experiment.
- `data_check.ipynb` can be used for dataset inspection or custom visualizations before training.

